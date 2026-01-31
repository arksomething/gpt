#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import sentencepiece as spm
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Tee:
    """Duplicate stdout/stderr to a log file."""

    def __init__(self, log_path, stream_name="stdout"):
        self.log_path = log_path
        self.stream_name = stream_name
        self.original = getattr(sys, stream_name)
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8")
        setattr(sys, stream_name, self)

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def close(self):
        setattr(sys, self.stream_name, self.original)
        self.log_file.close()


def setup_file_logging(logging_cfg, output_dir):
    """Set up file logging if enabled. Returns cleanup function."""
    log_file = logging_cfg.get("log_file")
    if not log_file:
        return lambda: None

    if log_file == "auto":
        log_path = os.path.join(output_dir, "train.log")
    else:
        log_path = log_file

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    stdout_tee = Tee(log_path, "stdout")
    stderr_tee = Tee(log_path, "stderr")

    print(f"[logging] Writing to {log_path}")

    def cleanup():
        stdout_tee.close()
        stderr_tee.close()

    return cleanup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def maybe_launch_screen(enabled, session_name):
    if not enabled:
        return False
    if os.environ.get("STY") or os.environ.get("TMUX"):
        return False
    if os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or os.environ.get("WORLD_SIZE"):
        print(
            "Distributed launch detected; skipping screen auto-launch. "
            "Start the accelerate command inside screen instead."
        )
        return False
    session = session_name or f"train-{time.strftime('%Y%m%d-%H%M%S')}"
    command = [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]]

    if shutil.which("screen") is not None:
        subprocess.check_call(["screen", "-dmS", session, *command])
        print(f"Started screen session '{session}'. Attach with: screen -r {session}")
        return True

    if shutil.which("tmux") is not None:
        subprocess.check_call(["tmux", "new-session", "-d", "-s", session, *command])
        print(f"Started tmux session '{session}'. Attach with: tmux attach -t {session}")
        return True

    print("screen/tmux not found; running in foreground.")
    return False


def load_memmap(path, dtype):
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, batch_size, block_size, rng, device):
    max_idx = len(data) - block_size - 1
    if max_idx <= 0:
        raise SystemExit("Data file too small for block_size.")
    idx = rng.integers(0, max_idx, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in idx])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x = torch.from_numpy(x).long().to(device, non_blocking=True)
    y = torch.from_numpy(y).long().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(model, data, batch_size, block_size, rng, device, batches, accelerator):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, block_size, rng, device)
        outputs = model(input_ids=x, labels=y)
        losses.append(outputs.loss.detach())
    loss_tensor = torch.stack(losses)
    loss_tensor = accelerator.gather(loss_tensor)
    return loss_tensor.mean().item()


def maybe_apply_budget_guard(budget, tokens_per_step):
    if not budget:
        return None
    throughput_path = budget.get("throughput_path")
    if not throughput_path or not os.path.exists(throughput_path):
        return None
    with open(throughput_path, "r", encoding="utf-8") as f:
        throughput = json.load(f)
    tokens_per_sec = throughput.get("tokens_per_sec")
    if not tokens_per_sec:
        return None
    target_tokens = budget.get("target_tokens", 0)
    hourly_rate = budget.get("hourly_rate", 0.0)
    max_cost = budget.get("max_cost", 0.0)
    if target_tokens <= 0 or hourly_rate <= 0 or max_cost <= 0:
        return None
    total_hours = target_tokens / tokens_per_sec / 3600.0
    projected_cost = total_hours * hourly_rate
    if projected_cost > max_cost:
        raise SystemExit(
            f"Projected cost ${projected_cost:.2f} exceeds max_cost ${max_cost:.2f}. "
            "Reduce target_tokens or pick cheaper GPUs."
        )
    return math.ceil(target_tokens / tokens_per_step)


def rotate_checkpoints(output_dir, limit, protected=None):
    """Keep at most `limit` total checkpoints, deleting oldest unprotected first."""
    if limit <= 0:
        return
    protected = set(protected or [])
    entries = [d for d in os.listdir(output_dir) if d.startswith("step_")]
    total_count = len(entries)
    if total_count <= limit:
        return
    # Delete oldest unprotected checkpoints until we have at most `limit` total
    unprotected = sorted([e for e in entries if e not in protected])
    to_remove_count = total_count - limit
    to_remove = unprotected[:to_remove_count]
    for name in to_remove:
        path = os.path.join(output_dir, name)
        shutil.rmtree(path)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml_or_json(path):
    if path.endswith(".json"):
        return _load_json(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_nested(cfg, dotted_path, value):
    parts = dotted_path.split(".")
    current = cfg
    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[parts[-1]] = value


def _get_nested(cfg, dotted_path, default=None):
    parts = dotted_path.split(".")
    current = cfg
    for key in parts:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_prompt_list(path):
    if not path or not os.path.exists(path):
        return []
    data = _load_yaml_or_json(path)
    if isinstance(data, dict):
        data = data.get("prompts", [])
    if not isinstance(data, list):
        return []
    return [str(prompt) for prompt in data if prompt is not None]


def _collect_prompts(fixed_cfg):
    prompts = []
    prompt_list_path = fixed_cfg.get("prompt_list_path")
    prompts.extend(_load_prompt_list(prompt_list_path))
    prompts.extend(fixed_cfg.get("prompt_list", []) or [])
    if not prompts:
        prompts = [fixed_cfg.get("prompt", "The quick brown fox")]
    return [str(prompt) for prompt in prompts]


def _format_sample_block(step, prompt, sample, tag=None):
    header = f"step {step}"
    if tag:
        header = f"{header} [{tag}]"
    return f"{header}\nprompt: {prompt}\n{sample}\n"


def _read_command_queue(path, start_offset):
    if not path or not os.path.exists(path):
        return [], start_offset
    commands = []
    with open(path, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                commands.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        new_offset = f.tell()
    return commands, new_offset


class RuntimeControl:
    def __init__(self, cfg, output_dir):
        self.enabled = bool(cfg.get("enabled", False))
        self.poll_interval_steps = max(1, int(cfg.get("poll_interval_steps", 50)))
        self.control_path = cfg.get("control_path") or os.path.join(output_dir, "runtime_control.yaml")
        self.command_path = cfg.get("command_path") or os.path.join(output_dir, "commands.jsonl")
        self.allowed_updates = set(cfg.get("allowed_updates", []))
        self._last_control_mtime = None
        self._command_offset = 0

    def poll(self, step):
        if not self.enabled or step % self.poll_interval_steps != 0:
            return {}, []
        updates = {}
        if self.control_path and os.path.exists(self.control_path):
            mtime = os.path.getmtime(self.control_path)
            if self._last_control_mtime is None or mtime > self._last_control_mtime:
                payload = _load_yaml_or_json(self.control_path) or {}
                if isinstance(payload, dict):
                    updates.update(payload.get("updates", {}))
                    if "prompts" in payload:
                        updates["checks.fixed_prompt.prompt_list"] = payload["prompts"]
                    if "prompt_list_path" in payload:
                        updates["checks.fixed_prompt.prompt_list_path"] = payload["prompt_list_path"]
                self._last_control_mtime = mtime
        commands, self._command_offset = _read_command_queue(self.command_path, self._command_offset)
        return updates, commands


class MetricsLogger:
    def __init__(self, cfg, output_dir, is_main_process):
        self.enabled = bool(cfg.get("enabled", True)) and is_main_process
        self.console_summary = bool(cfg.get("console_summary", True))
        self.tb = None
        self.wandb = None
        if not self.enabled:
            return
        tb_cfg = cfg.get("tensorboard", {})
        if tb_cfg.get("enabled", False):
            log_dir = tb_cfg.get("log_dir") or os.path.join(output_dir, "tb")
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise SystemExit("TensorBoard enabled but not installed.") from exc
            self.tb = SummaryWriter(log_dir=log_dir)
        wandb_cfg = cfg.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            try:
                import wandb
            except ImportError as exc:
                raise SystemExit("wandb enabled but not installed.") from exc
            self.wandb = wandb.init(
                project=wandb_cfg.get("project"),
                name=wandb_cfg.get("name"),
                entity=wandb_cfg.get("entity"),
                tags=wandb_cfg.get("tags"),
                group=wandb_cfg.get("group"),
                config=wandb_cfg.get("config"),
            )

    def log_metrics(self, step, metrics):
        if not self.enabled:
            return
        if self.tb:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb.add_scalar(key, value, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_text(self, step, key, text):
        if not self.enabled:
            return
        if self.tb:
            self.tb.add_text(key, text, step)
        if self.wandb:
            self.wandb.log({key: text}, step=step)

    def maybe_print(self, text):
        if self.enabled and self.console_summary:
            print(text)


def _get_gpu_stats():
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "gpu_mem_free_gb": free_bytes / (1024**3),
        "gpu_mem_used_gb": used_bytes / (1024**3),
        "gpu_mem_total_gb": total_bytes / (1024**3),
        "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def _load_checkpoint_manifest(output_dir):
    manifest_path = os.path.join(output_dir, "checkpoint_manifest.json")
    if not os.path.exists(manifest_path):
        return {"last": None, "best": [], "good_slots": {}, "steps": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint_manifest(output_dir, manifest):
    manifest_path = os.path.join(output_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _update_best_slots(manifest, step_dir, val_loss, max_slots):
    if val_loss is None:
        return
    best = manifest.get("best", [])
    best.append({"step": step_dir, "val_loss": val_loss})
    best = sorted(best, key=lambda item: item["val_loss"])[: max_slots]
    manifest["best"] = best


def _protected_steps(manifest):
    protected = set()
    if manifest.get("last"):
        protected.add(manifest["last"])
    for entry in manifest.get("best", []):
        if entry.get("step"):
            protected.add(entry["step"])
    for step_dir in (manifest.get("good_slots") or {}).values():
        if step_dir:
            protected.add(step_dir)
    return protected


def _resolve_slot_step(manifest, slot):
    if slot == "last":
        return manifest.get("last")
    if slot == "best":
        best = manifest.get("best", [])
        if best:
            return best[0].get("step")
    good_slots = manifest.get("good_slots") or {}
    if slot in good_slots:
        return good_slots.get(slot)
    return None


def _apply_runtime_updates(optim_cfg, checks_cfg, updates):
    applied = {}
    for path, value in updates.items():
        if path.startswith("training."):
            key = path.replace("training.", "", 1)
            _set_nested(optim_cfg, key, value)
            applied[path] = value
        elif path.startswith("checks."):
            key = path.replace("checks.", "", 1)
            _set_nested(checks_cfg, key, value)
            applied[path] = value
    return applied


def _filter_allowed_updates(updates, allowed):
    if not allowed:
        return updates
    return {path: value for path, value in updates.items() if path in allowed}


def _normalize_runtime_values(optim_cfg):
    for key in ("eval_interval", "save_interval", "log_interval"):
        if key in optim_cfg:
            optim_cfg[key] = max(1, int(optim_cfg[key]))
    if "learning_rate" in optim_cfg:
        optim_cfg["learning_rate"] = float(optim_cfg["learning_rate"])


def _parse_runtime_commands(commands, good_slots):
    updates = {}
    actions = {
        "sample_prompts": [],
        "pin_slots": [],
        "stop_training": False,
        "save_now": False,
    }
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        cmd_type = cmd.get("cmd")
        if cmd_type == "set":
            path = cmd.get("path")
            if path:
                updates[path] = cmd.get("value")
        elif cmd_type == "sample_prompt":
            actions["sample_prompts"].append(cmd)
        elif cmd_type == "pin_checkpoint":
            slot = cmd.get("slot", "good_1")
            if slot in good_slots:
                actions["pin_slots"].append(cmd)
        elif cmd_type == "stop_training":
            actions["stop_training"] = True
            actions["save_now"] = bool(cmd.get("save", True))
    return updates, actions

def _truncate_at_eos(token_ids, eos_id):
    if eos_id is None:
        return token_ids
    try:
        eos_idx = token_ids.index(eos_id)
    except ValueError:
        return token_ids
    return token_ids[:eos_idx]


def _decode_tokens(sp, token_ids, special_ids, eos_id):
    token_ids = _truncate_at_eos(token_ids, eos_id)
    if special_ids:
        token_ids = [t for t in token_ids if t not in special_ids]
    return sp.decode(token_ids)


def _capture_rng_state():
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state


def _restore_rng_state(state):
    if state is None:
        return
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def sample_generate(
    model,
    sp,
    prompt,
    device,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    min_new_tokens,
    special_ids,
    eos_id,
    deterministic,
    seed,
):
    model.eval()
    input_ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_ids], device=device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
    }
    if not deterministic:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k
    if repetition_penalty and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if min_new_tokens and min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = min_new_tokens

    rng_state = None
    if seed is not None and not deterministic:
        rng_state = _capture_rng_state()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    outputs = model.generate(input_ids=input_ids, **gen_kwargs)

    if rng_state is not None:
        _restore_rng_state(rng_state)

    return _decode_tokens(sp, outputs[0].tolist(), special_ids, eos_id)


def run_overfit_microset(
    model,
    train_data,
    data_cfg,
    optim_cfg,
    accelerator,
    check_cfg,
):
    if not check_cfg or not check_cfg.get("enabled", False):
        return

    block_size = data_cfg["block_size"]
    micro_batch = check_cfg.get("micro_batch_size", optim_cfg["micro_batch_size"])
    grad_accum = check_cfg.get("grad_accum_steps", optim_cfg["grad_accum_steps"])
    steps = check_cfg.get("steps", 100)
    eval_batches = check_cfg.get("eval_batches", 20)
    log_interval = max(1, check_cfg.get("log_interval", 10))
    learning_rate = check_cfg.get("learning_rate", optim_cfg["learning_rate"])
    max_grad_norm = check_cfg.get("max_grad_norm", optim_cfg["max_grad_norm"])
    warmup_steps = check_cfg.get("warmup_steps", 0)
    min_drop = check_cfg.get("min_drop", 0.3)
    target_loss = check_cfg.get("target_loss")

    microset_tokens = int(check_cfg.get("tokens", 5_000_000))
    microset_tokens = min(microset_tokens, len(train_data))
    if microset_tokens <= block_size + 1:
        raise SystemExit("Overfit microset too small for block_size.")

    microset_data = train_data[:microset_tokens]
    if accelerator.is_main_process:
        print(
            f"[overfit check] tokens={microset_tokens} steps={steps} "
            f"micro_batch={micro_batch} grad_accum={grad_accum}"
        )

    eval_seed = check_cfg.get("eval_seed", optim_cfg["seed"] + 12345)
    initial_loss = evaluate(
        model,
        microset_data,
        micro_batch,
        block_size,
        np.random.default_rng(eval_seed),
        accelerator.device,
        eval_batches,
        accelerator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(optim_cfg["betas"]),
        weight_decay=optim_cfg["weight_decay"],
    )
    lr_scheduler = None
    if warmup_steps and warmup_steps > 0:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps,
        )

    rng = np.random.default_rng(optim_cfg["seed"] + 4242 + accelerator.process_index)
    for step in range(1, steps + 1):
        model.train()
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(microset_data, micro_batch, block_size, rng, accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()
            step_loss += loss.item()

        if accelerator.is_main_process and step % log_interval == 0:
            avg_loss = step_loss / grad_accum
            print(f"[overfit check] step {step}/{steps} loss={avg_loss:.4f}")

    final_loss = evaluate(
        model,
        microset_data,
        micro_batch,
        block_size,
        np.random.default_rng(eval_seed),
        accelerator.device,
        eval_batches,
        accelerator,
    )
    if accelerator.is_main_process:
        drop = (initial_loss - final_loss) / max(1e-6, initial_loss)
        print(
            f"[overfit check] initial_loss={initial_loss:.4f} "
            f"final_loss={final_loss:.4f} drop={drop:.2%}"
        )
        drop_ok = min_drop is None or min_drop <= 0 or drop >= min_drop
        loss_ok = target_loss is None or final_loss <= target_loss
        if not (drop_ok or loss_ok):
            raise SystemExit(
                "Overfit microset check failed. Loss did not drop enough; "
                "verify data pipeline, labels, or masking."
            )

    accelerator.wait_for_everyone()


def setup_fixed_prompt_sampler(fixed_cfg, model_cfg, output_dir, smoke):
    if not fixed_cfg or not fixed_cfg.get("enabled", False):
        return None
    if smoke and not fixed_cfg.get("run_on_smoke", False):
        return None
    tokenizer_model = fixed_cfg.get("tokenizer_model")
    if not tokenizer_model:
        raise SystemExit("checks.fixed_prompt.tokenizer_model is required.")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)
    special_ids = {
        model_cfg["bos_token_id"],
        model_cfg["eos_token_id"],
        model_cfg["pad_token_id"],
        sp.unk_id(),
    }
    special_ids = {token_id for token_id in special_ids if token_id is not None}
    output_path = fixed_cfg.get("output_path")
    if not output_path:
        output_path = os.path.join(output_dir, "fixed_prompt_samples.txt")
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return {
        "sp": sp,
        "special_ids": special_ids,
        "output_path": output_path,
    }


def run_fixed_prompt_sample(
    model,
    accelerator,
    fixed_cfg,
    sampler_state,
    step,
    device,
    logger=None,
    prompts_override=None,
    tag="fixed_prompt",
):
    if sampler_state is None or not fixed_cfg.get("enabled", False):
        return
    if not accelerator.is_main_process:
        return
    max_new_tokens = fixed_cfg.get("max_new_tokens", 120)
    min_new_tokens = fixed_cfg.get("min_new_tokens", 16)
    temperature = fixed_cfg.get("temperature", 0.7)
    top_p = fixed_cfg.get("top_p", 0.9)
    top_k = fixed_cfg.get("top_k", 50)
    repetition_penalty = fixed_cfg.get("repetition_penalty", 1.1)
    deterministic = fixed_cfg.get("deterministic", True)
    seed = fixed_cfg.get("seed")

    prompts = prompts_override or _collect_prompts(fixed_cfg)
    if not prompts:
        return

    unwrapped = accelerator.unwrap_model(model)
    output_blocks = []
    for idx, prompt in enumerate(prompts, start=1):
        sample = sample_generate(
            unwrapped,
            sampler_state["sp"],
            prompt,
            device,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_new_tokens,
            sampler_state["special_ids"],
            unwrapped.config.eos_token_id,
            deterministic,
            seed,
        )
        print(f"[{tag}] step {step} prompt={prompt!r}")
        print(sample)
        output_blocks.append(_format_sample_block(step, prompt, sample, tag=tag))
        if logger is not None:
            logger.log_text(step, f"samples/{tag}/{idx}", _format_sample_block(step, prompt, sample))

    output_path = sampler_state["output_path"]
    with open(output_path, "a", encoding="utf-8") as f:
        for block in output_blocks:
            f.write(block)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Train a 100M Llama-style model.")
    parser.add_argument("--model_config", default="configs/model_100m.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--resume_from_slot", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--launch_screen", action="store_true")
    parser.add_argument("--screen_name", default=None)
    args = parser.parse_args()

    if maybe_launch_screen(args.launch_screen, args.screen_name):
        return

    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.train_config)
    data_cfg = train_cfg["data"]
    optim_cfg = train_cfg["training"]
    budget_cfg = train_cfg.get("budget", {})
    checks_cfg = train_cfg.get("checks", {})
    logging_cfg = train_cfg.get("logging", {})
    runtime_cfg = train_cfg.get("runtime_control", {})
    checkpoint_cfg = train_cfg.get("checkpoint_slots", {})
    overfit_cfg = checks_cfg.get("overfit_microset", {})
    fixed_prompt_cfg = checks_cfg.setdefault("fixed_prompt", {})

    _normalize_runtime_values(optim_cfg)

    set_seed(optim_cfg["seed"])
    if optim_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    grad_accum = int(optim_cfg["grad_accum_steps"])
    accelerator = Accelerator(
        mixed_precision=optim_cfg.get("precision", "bf16"),
        gradient_accumulation_steps=grad_accum,
    )
    device = accelerator.device

    dtype = np.dtype(data_cfg["dtype"])
    train_data = load_memmap(data_cfg["train_bin"], dtype)
    val_data = load_memmap(data_cfg["val_bin"], dtype)

    config = LlamaConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg["num_key_value_heads"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rms_norm_eps=model_cfg["rms_norm_eps"],
        rope_theta=model_cfg["rope_theta"],
        hidden_act=model_cfg["hidden_act"],
        attention_bias=model_cfg["attention_bias"],
        mlp_bias=model_cfg["mlp_bias"],
        tie_word_embeddings=model_cfg["tie_word_embeddings"],
        pad_token_id=model_cfg["pad_token_id"],
        bos_token_id=model_cfg["bos_token_id"],
        eos_token_id=model_cfg["eos_token_id"],
    )
    model = LlamaForCausalLM(config)
    if optim_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    param_count = count_parameters(model)
    if accelerator.is_main_process:
        print(f"Model parameters: {param_count/1e6:.2f}M")

    block_size = data_cfg["block_size"]
    micro_batch = optim_cfg["micro_batch_size"]
    world_size = accelerator.num_processes
    tokens_per_step = micro_batch * block_size * grad_accum * world_size
    target_steps = maybe_apply_budget_guard(budget_cfg, tokens_per_step)

    max_steps = optim_cfg["max_steps"]
    if target_steps:
        max_steps = min(max_steps, target_steps)
    if args.smoke:
        max_steps = min(max_steps, 50)

    output_dir = optim_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Set up file logging (only on main process)
    cleanup_logging = lambda: None
    if accelerator.is_main_process:
        cleanup_logging = setup_file_logging(logging_cfg, output_dir)

    if not runtime_cfg.get("allowed_updates"):
        runtime_cfg["allowed_updates"] = [
            "training.eval_interval",
            "training.save_interval",
            "training.log_interval",
            "training.learning_rate",
            "checks.fixed_prompt.enabled",
            "checks.fixed_prompt.deterministic",
            "checks.fixed_prompt.max_new_tokens",
            "checks.fixed_prompt.min_new_tokens",
            "checks.fixed_prompt.temperature",
            "checks.fixed_prompt.top_p",
            "checks.fixed_prompt.top_k",
            "checks.fixed_prompt.repetition_penalty",
            "checks.fixed_prompt.prompt",
            "checks.fixed_prompt.prompt_list",
            "checks.fixed_prompt.prompt_list_path",
        ]

    runtime_control = RuntimeControl(runtime_cfg, output_dir)
    metrics_logger = MetricsLogger(logging_cfg, output_dir, accelerator.is_main_process)
    manifest = _load_checkpoint_manifest(output_dir)
    best_slots = max(0, int(checkpoint_cfg.get("best", 2)))
    good_slots = list(checkpoint_cfg.get("good", ["good_1", "good_2"]) or [])

    model_prepared = False
    if overfit_cfg.get("enabled", False) and (not args.smoke or overfit_cfg.get("run_on_smoke", False)):
        model = accelerator.prepare(model)
        model_prepared = True
        run_overfit_microset(model, train_data, data_cfg, optim_cfg, accelerator, overfit_cfg)
    elif accelerator.is_main_process and overfit_cfg.get("enabled", False) and args.smoke:
        print("[overfit check] skipped on smoke run.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg["learning_rate"],
        betas=tuple(optim_cfg["betas"]),
        weight_decay=optim_cfg["weight_decay"],
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optim_cfg["warmup_steps"],
        num_training_steps=max_steps,
    )

    if model_prepared:
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        model_prepared = True

    fixed_prompt_sampler = setup_fixed_prompt_sampler(
        fixed_prompt_cfg,
        model_cfg,
        output_dir,
        args.smoke,
    )

    resume_path = args.resume_from
    if args.resume_from_slot:
        slot_step = _resolve_slot_step(manifest, args.resume_from_slot)
        if not slot_step:
            raise SystemExit(f"resume_from_slot '{args.resume_from_slot}' not found in manifest.")
        resume_path = os.path.join(output_dir, slot_step)
    if resume_path:
        accelerator.load_state(resume_path)

    rng = np.random.default_rng(optim_cfg["seed"] + accelerator.process_index)
    tokens_processed = 0
    start_time = time.time()
    last_val_loss = None
    last_grad_norm = None
    stop_training = False

    for step in range(1, max_steps + 1):
        updates, commands = runtime_control.poll(step)
        cmd_updates, actions = _parse_runtime_commands(commands, good_slots)
        updates.update(cmd_updates)
        updates = _filter_allowed_updates(updates, runtime_control.allowed_updates)
        applied = _apply_runtime_updates(optim_cfg, checks_cfg, updates)
        if applied:
            _normalize_runtime_values(optim_cfg)
            if "training.learning_rate" in applied:
                new_lr = float(optim_cfg["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr
                if hasattr(lr_scheduler, "base_lrs"):
                    lr_scheduler.base_lrs = [new_lr for _ in lr_scheduler.base_lrs]
            if accelerator.is_main_process:
                metrics_logger.maybe_print(f"[runtime] applied updates: {applied}")

        save_requested = False
        if actions["sample_prompts"]:
            for cmd in actions["sample_prompts"]:
                prompts = cmd.get("prompts")
                if prompts is None and cmd.get("prompt") is not None:
                    prompts = [cmd.get("prompt")]
                if not prompts:
                    continue
                temp_cfg = dict(fixed_prompt_cfg)
                temp_cfg.update(cmd.get("params", {}) or {})
                run_fixed_prompt_sample(
                    model,
                    accelerator,
                    temp_cfg,
                    fixed_prompt_sampler,
                    step,
                    device,
                    logger=metrics_logger,
                    prompts_override=prompts,
                    tag=cmd.get("tag", "ad_hoc"),
                )
        if actions["pin_slots"] and accelerator.is_main_process:
            for cmd in actions["pin_slots"]:
                slot = cmd.get("slot", "good_1")
                target = cmd.get("step", "last")
                if isinstance(target, int):
                    step_dir = f"step_{target:07d}"
                elif isinstance(target, str) and target.isdigit():
                    step_dir = f"step_{int(target):07d}"
                elif target == "last":
                    step_dir = manifest.get("last")
                else:
                    step_dir = target
                if not step_dir:
                    continue
                if not os.path.isdir(os.path.join(output_dir, step_dir)):
                    continue
                manifest.setdefault("good_slots", {})[slot] = step_dir
                _save_checkpoint_manifest(output_dir, manifest)
                metrics_logger.maybe_print(f"[checkpoint] pinned {slot} -> {step_dir}")
        if actions["stop_training"]:
            stop_training = True
            save_requested = actions["save_now"]

        model.train()
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, micro_batch, block_size, rng, device)
            with accelerator.accumulate(model):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = None
                    if optim_cfg["max_grad_norm"] > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), optim_cfg["max_grad_norm"]
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if grad_norm is not None:
                        last_grad_norm = float(grad_norm)
            step_loss += loss.item()

        tokens_processed += tokens_per_step
        if accelerator.is_main_process and step % optim_cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            tps = tokens_processed / max(1e-6, elapsed)
            avg_loss = step_loss / grad_accum
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            metrics = {
                "train/loss": avg_loss,
                "train/lr": current_lr,
                "train/tokens_per_sec": tps,
                "train/tokens": tokens_processed,
            }
            if last_grad_norm is not None:
                metrics["train/grad_norm"] = last_grad_norm
            metrics.update(_get_gpu_stats())
            metrics_logger.log_metrics(step, metrics)
            metrics_logger.maybe_print(
                f"step {step}/{max_steps} loss={avg_loss:.4f} "
                f"lr={current_lr:.2e} tps={tps:.0f}"
            )

        if step % optim_cfg["eval_interval"] == 0:
            val_loss = evaluate(
                model,
                val_data,
                micro_batch,
                block_size,
                rng,
                device,
                batches=20 if args.smoke else 100,
                accelerator=accelerator,
            )
            last_val_loss = val_loss
            ppl = math.exp(min(20, val_loss))
            if accelerator.is_main_process:
                print(f"eval loss={val_loss:.4f} ppl={ppl:.2f}")
                metrics_logger.log_metrics(
                    step,
                    {
                        "eval/loss": val_loss,
                        "eval/ppl": ppl,
                    },
                )
            run_fixed_prompt_sample(
                model,
                accelerator,
                fixed_prompt_cfg,
                fixed_prompt_sampler,
                step,
                device,
                logger=metrics_logger,
            )

        if step % optim_cfg["save_interval"] == 0 or save_requested:
            ckpt_dir_name = f"step_{step:07d}"
            ckpt_dir = os.path.join(output_dir, ckpt_dir_name)
            accelerator.wait_for_everyone()
            accelerator.save_state(ckpt_dir)
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                manifest["last"] = ckpt_dir_name
                manifest.setdefault("steps", {})[ckpt_dir_name] = {
                    "step": step,
                    "val_loss": last_val_loss,
                    "timestamp": time.time(),
                }
                if best_slots > 0:
                    _update_best_slots(manifest, ckpt_dir_name, last_val_loss, best_slots)
                _save_checkpoint_manifest(output_dir, manifest)
                rotate_checkpoints(
                    output_dir,
                    optim_cfg["checkpoint_limit"],
                    protected=_protected_steps(manifest),
                )

        if stop_training:
            break

    accelerator.wait_for_everyone()
    final_dir = os.path.join(output_dir, "final")
    accelerator.save_state(final_dir)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "model.pt"))
        print(f"Saved final checkpoint to {final_dir}")
        cleanup_logging()


if __name__ == "__main__":
    main()
