#!/usr/bin/env python3
import argparse
import hashlib
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
from huggingface_hub import HfApi, login as hf_login


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_tokenizer_path(train_cfg):
    checks_cfg = train_cfg.get("checks", {})
    fixed_cfg = checks_cfg.get("fixed_prompt", {})
    data_prep_cfg = train_cfg.get("data_prep", {})
    eval_cfg = train_cfg.get("eval", {})
    inference_cfg = train_cfg.get("inference", {})
    candidates = [
        fixed_cfg.get("tokenizer_model"),
        data_prep_cfg.get("tokenizer_model"),
        eval_cfg.get("tokenizer_model"),
        inference_cfg.get("tokenizer"),
    ]
    for path in candidates:
        if path:
            return path
    return None


def _verify_tokenizer_compat(model_cfg, train_cfg, data_cfg):
    tokenizer_path = _resolve_tokenizer_path(train_cfg)
    if not tokenizer_path:
        return
    if not os.path.exists(tokenizer_path):
        raise SystemExit(f"Tokenizer model not found: {tokenizer_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    mismatches = []
    expected_vocab = int(model_cfg["vocab_size"])
    actual_vocab = int(sp.vocab_size())
    if actual_vocab != expected_vocab:
        mismatches.append(
            f"vocab_size mismatch: model={expected_vocab} tokenizer={actual_vocab}"
        )

    tokenizer_ids = {
        "bos_token_id": sp.bos_id(),
        "eos_token_id": sp.eos_id(),
        "pad_token_id": sp.pad_id(),
    }
    for key, tok_id in tokenizer_ids.items():
        model_id = model_cfg.get(key)
        if model_id is None or tok_id is None:
            continue
        if int(model_id) != int(tok_id):
            mismatches.append(f"{key} mismatch: model={model_id} tokenizer={tok_id}")

    if mismatches:
        details = "\n- ".join(mismatches)
        raise SystemExit(
            "Tokenizer/model config mismatch. Fix before training:\n"
            f"- {details}"
        )

    train_bin = data_cfg.get("train_bin")
    if not train_bin:
        return
    meta_path = os.path.join(os.path.dirname(train_bin) or ".", "data_meta.json")
    meta = _load_json_if_exists(meta_path)
    if not meta:
        if data_cfg.get("hf_repo"):
            print(
                "[data] data_meta.json not found for train data. "
                "If you use pre-tokenized HF data, ensure tokenizer matches that dataset."
            )
        return

    meta_vocab = meta.get("tokenizer_vocab_size")
    if meta_vocab is not None and int(meta_vocab) != actual_vocab:
        raise SystemExit(
            "Tokenizer/data mismatch: data_meta tokenizer_vocab_size "
            f"is {meta_vocab}, current tokenizer vocab_size is {actual_vocab}."
        )

    meta_special = meta.get("tokenizer_special_ids") or {}
    if isinstance(meta_special, dict):
        key_pairs = (
            ("bos_token_id", "bos"),
            ("eos_token_id", "eos"),
            ("pad_token_id", "pad"),
        )
        for model_key, meta_key in key_pairs:
            if meta_key not in meta_special:
                continue
            meta_id = meta_special.get(meta_key)
            model_id = model_cfg.get(model_key)
            if meta_id is None or model_id is None:
                continue
            if int(meta_id) != int(model_id):
                raise SystemExit(
                    "Tokenizer/data mismatch: "
                    f"data_meta {meta_key}={meta_id}, model {model_key}={model_id}."
                )

    meta_hash = meta.get("tokenizer_sha256")
    if meta_hash:
        current_hash = _file_sha256(tokenizer_path)
        if current_hash != meta_hash:
            raise SystemExit(
                "Tokenizer file does not match data tokenization. "
                f"Expected sha256={meta_hash}, got {current_hash}. "
                "Re-run prepare-data with this tokenizer or switch to the matching tokenizer."
            )
    else:
        print(
            "[data] data_meta.json has no tokenizer_sha256; cannot fully verify "
            "tokenizer/data consistency. Re-run prepare-data to add tokenizer fingerprinting."
        )


class CheckpointUploader:
    """Uploads checkpoints to HuggingFace Hub."""
    
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.repo_id = config.get("repo_id")
        self.upload_interval = config.get("upload_interval", 1000)
        self.delete_after_upload = config.get("delete_local_after_upload", False)
        self.upload_optimizer = config.get("upload_optimizer", False)
        self.token = config.get("token") or os.environ.get("HF_TOKEN")
        self._api = None
        self._last_upload_step = 0
        
        if self.enabled and not self.repo_id:
            print("[checkpoint_upload] Warning: enabled but no repo_id set, disabling")
            self.enabled = False
        
        if self.enabled and self.token:
            try:
                hf_login(token=self.token, add_to_git_credential=False)
            except Exception as e:
                print(f"[checkpoint_upload] Warning: HF login failed: {e}")
    
    @property
    def api(self):
        if self._api is None:
            self._api = HfApi()
            # Create repo if it doesn't exist
            if self.enabled:
                try:
                    self._api.create_repo(self.repo_id, repo_type="model", exist_ok=True)
                except Exception as e:
                    print(f"[checkpoint_upload] Warning: Could not create repo: {e}")
        return self._api
    
    def should_upload(self, step: int) -> bool:
        """Check if we should upload at this step."""
        if not self.enabled:
            return False
        if step - self._last_upload_step >= self.upload_interval:
            return True
        return False
    
    def upload(self, checkpoint_dir: str, step: int, is_final: bool = False):
        """Upload checkpoint to HuggingFace Hub."""
        if not self.enabled:
            return
        
        try:
            folder_name = "final" if is_final else f"step_{step:07d}"
            
            # Determine which files to upload
            files_to_upload = ["model.pt", "model.safetensors"]
            if self.upload_optimizer:
                files_to_upload.extend(["optimizer.bin", "scheduler.bin", "random_states_0.pkl"])
            
            uploaded_files = []
            for filename in files_to_upload:
                filepath = os.path.join(checkpoint_dir, filename)
                if os.path.exists(filepath):
                    self.api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=f"{folder_name}/{filename}",
                        repo_id=self.repo_id,
                        repo_type="model",
                    )
                    uploaded_files.append(filename)
            
            print(f"[checkpoint_upload] Uploaded {folder_name} to {self.repo_id} ({', '.join(uploaded_files)})")
            self._last_upload_step = step
            
            # Delete local checkpoint if configured
            if self.delete_after_upload and not is_final:
                shutil.rmtree(checkpoint_dir)
                print(f"[checkpoint_upload] Deleted local checkpoint {checkpoint_dir}")
                
        except Exception as e:
            print(f"[checkpoint_upload] Error uploading checkpoint: {e}")
    
    def upload_logs(self, output_dir: str):
        """Upload training logs to HuggingFace Hub."""
        if not self.enabled:
            return
        
        try:
            log_files = ["train.log", "fixed_prompt_samples.txt"]
            for filename in log_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    self.api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=self.repo_id,
                        repo_type="model",
                    )
            print(f"[checkpoint_upload] Uploaded logs to {self.repo_id}")
        except Exception as e:
            print(f"[checkpoint_upload] Error uploading logs: {e}")


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


def load_memmap(path, dtype, hf_repo=None):
    """Load a memmap file, optionally downloading from HuggingFace first."""
    if not os.path.exists(path) and hf_repo:
        # Download from HuggingFace if not present locally
        from huggingface_hub import hf_hub_download
        filename = os.path.basename(path)
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                repo_type="dataset",
                local_dir=os.path.dirname(path) or ".",
            )
            path = local_path
            print(f"[data] Downloaded missing file {filename} from {hf_repo}")
        except Exception as e:
            print(f"[data] Warning: Could not download from HF: {e}, trying local path")
    
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, batch_size, block_size, rng, device):
    # HF CausalLM loss already shifts labels internally (token < n predicts n),
    # so labels must be aligned with input_ids, not pre-shifted.
    max_idx = len(data) - block_size + 1
    if max_idx <= 0:
        raise SystemExit("Data file too small for block_size.")
    idx = rng.integers(0, max_idx, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in idx])
    x = torch.from_numpy(x).long().to(device, non_blocking=True)
    y = x.clone()
    return x, y


class StreamingDataLoader:
    """Streams tokenized data from HuggingFace with prefetching."""
    
    def __init__(
        self,
        hf_repo: str,
        split: str,
        block_size: int,
        batch_size: int,
        device,
        buffer_size: int = 10000,
        prefetch_batches: int = 10,
        seed: int = 1337,
    ):
        from datasets import load_dataset
        from threading import Thread
        from queue import Queue
        
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.buffer_size = buffer_size
        self.prefetch_batches = prefetch_batches
        self.rng = np.random.default_rng(seed)
        
        # Load streaming dataset
        print(f"[streaming] Loading {hf_repo} split={split} (streaming mode)")
        self.dataset = load_dataset(
            hf_repo,
            split=split,
            streaming=True,
        ).shuffle(seed=seed, buffer_size=buffer_size)
        
        # Token buffer - accumulates tokens from streamed examples
        self._token_buffer = np.array([], dtype=np.uint16)
        self._dataset_iter = iter(self.dataset)
        self._min_buffer_tokens = block_size * batch_size * 4  # Keep buffer well-stocked
        
        # Prefetch queue
        self._batch_queue = Queue(maxsize=prefetch_batches)
        self._stop_prefetch = False
        self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        
        print(f"[streaming] Started with buffer_size={buffer_size}, prefetch={prefetch_batches}")
    
    def _fill_buffer(self):
        """Fill token buffer from streaming dataset."""
        tokens_needed = self._min_buffer_tokens - len(self._token_buffer)
        if tokens_needed <= 0:
            return
        
        new_tokens = []
        tokens_collected = 0
        
        while tokens_collected < tokens_needed:
            try:
                example = next(self._dataset_iter)
                # Expect pre-tokenized data with 'input_ids' or 'tokens' field
                if "input_ids" in example:
                    toks = example["input_ids"]
                elif "tokens" in example:
                    toks = example["tokens"]
                else:
                    # Skip examples without tokens
                    continue
                
                if isinstance(toks, list):
                    new_tokens.extend(toks)
                    tokens_collected += len(toks)
                    
            except StopIteration:
                # Dataset exhausted, restart
                self._dataset_iter = iter(self.dataset)
        
        if new_tokens:
            self._token_buffer = np.concatenate([
                self._token_buffer,
                np.array(new_tokens, dtype=np.uint16)
            ])
    
    def _get_batch_from_buffer(self):
        """Extract a batch from the token buffer."""
        self._fill_buffer()
        
        # HF CausalLM loss shifts labels internally, so no manual +1 offset.
        batch_tokens = self.block_size
        total_needed = self.batch_size * batch_tokens
        
        if len(self._token_buffer) < total_needed:
            raise RuntimeError("Token buffer too small - this shouldn't happen")
        
        # Random starting positions within buffer
        max_start = len(self._token_buffer) - batch_tokens + 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        
        x_list = []
        for start in starts:
            x_list.append(self._token_buffer[start:start + self.block_size])
        
        x = torch.from_numpy(np.stack(x_list)).long().to(self.device, non_blocking=True)
        y = x.clone()
        
        # Trim buffer occasionally to prevent unbounded growth
        if len(self._token_buffer) > self._min_buffer_tokens * 2:
            self._token_buffer = self._token_buffer[-self._min_buffer_tokens:]
        
        return x, y
    
    def _prefetch_worker(self):
        """Background thread that prefetches batches."""
        while not self._stop_prefetch:
            try:
                batch = self._get_batch_from_buffer()
                self._batch_queue.put(batch, timeout=1.0)
            except Exception as e:
                if not self._stop_prefetch:
                    print(f"[streaming] Prefetch error: {e}")
                    import time
                    time.sleep(0.1)
    
    def get_batch(self):
        """Get next batch (from prefetch queue)."""
        return self._batch_queue.get(timeout=30.0)
    
    def stop(self):
        """Stop prefetch thread."""
        self._stop_prefetch = True
        self._prefetch_thread.join(timeout=2.0)


def create_data_loader(data_cfg, batch_size, block_size, device, seed):
    """Create either memmap or streaming data loader based on config."""
    streaming_cfg = data_cfg.get("streaming", {})
    
    if streaming_cfg.get("enabled", False):
        hf_repo = streaming_cfg.get("hf_repo")
        if not hf_repo:
            raise SystemExit("streaming.hf_repo required when streaming.enabled=true")
        
        train_loader = StreamingDataLoader(
            hf_repo=hf_repo,
            split=streaming_cfg.get("train_split", "train"),
            block_size=block_size,
            batch_size=batch_size,
            device=device,
            buffer_size=streaming_cfg.get("buffer_size", 10000),
            prefetch_batches=streaming_cfg.get("prefetch_batches", 10),
            seed=seed,
        )
        
        # For validation, still use local or download
        # (val is small, streaming overhead not worth it)
        val_data = None
        if data_cfg.get("val_bin"):
            dtype = np.dtype(data_cfg["dtype"])
            hf_data_repo = data_cfg.get("hf_repo")
            val_data = load_memmap(data_cfg["val_bin"], dtype, hf_repo=hf_data_repo)
        
        return {"mode": "streaming", "train": train_loader, "val": val_data}
    else:
        # Original memmap mode
        dtype = np.dtype(data_cfg["dtype"])
        hf_data_repo = data_cfg.get("hf_repo")
        train_data = load_memmap(data_cfg["train_bin"], dtype, hf_repo=hf_data_repo)
        val_data = load_memmap(data_cfg["val_bin"], dtype, hf_repo=hf_data_repo)
        return {"mode": "memmap", "train": train_data, "val": val_data}


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
    checkpoint_upload_cfg = train_cfg.get("checkpoint_upload", {})
    overfit_cfg = checks_cfg.get("overfit_microset", {})
    fixed_prompt_cfg = checks_cfg.setdefault("fixed_prompt", {})

    _verify_tokenizer_compat(model_cfg, train_cfg, data_cfg)
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

    # Create data loader (memmap or streaming)
    data_loader = create_data_loader(
        data_cfg,
        batch_size=optim_cfg["micro_batch_size"],
        block_size=data_cfg["block_size"],
        device=device,
        seed=optim_cfg["seed"],
    )
    streaming_mode = data_loader["mode"] == "streaming"
    train_data = data_loader["train"]  # Either np.memmap or StreamingDataLoader
    val_data = data_loader["val"]  # np.memmap (or None)

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
    checkpoint_uploader = CheckpointUploader(checkpoint_upload_cfg) if accelerator.is_main_process else None

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
            # Get batch from streaming or memmap loader
            if streaming_mode:
                x, y = train_data.get_batch()
            else:
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
                
                # Upload checkpoint to HuggingFace if configured
                if checkpoint_uploader and checkpoint_uploader.should_upload(step):
                    checkpoint_uploader.upload(ckpt_dir, step)
                
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
        
        # Upload final checkpoint and logs to HuggingFace
        if checkpoint_uploader and checkpoint_uploader.enabled:
            checkpoint_uploader.upload(final_dir, step, is_final=True)
            checkpoint_uploader.upload_logs(output_dir)
        
        # Stop streaming data loader if active
        if streaming_mode:
            train_data.stop()
        
        cleanup_logging()


if __name__ == "__main__":
    main()
