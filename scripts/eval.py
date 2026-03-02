#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os

import numpy as np
import torch
import yaml
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import LlamaConfig, LlamaForCausalLM


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


def _find_artifact_manifest_path(checkpoint_dir):
    parent = os.path.dirname(os.path.normpath(checkpoint_dir))
    candidates = [
        os.path.join(checkpoint_dir, "artifacts_manifest.json"),
        os.path.join(checkpoint_dir, "artifacts", "artifacts_manifest.json"),
        os.path.join(parent, "artifacts_manifest.json"),
        os.path.join(parent, "artifacts", "artifacts_manifest.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_artifact_manifest(checkpoint_dir, repo_id):
    local_manifest = _find_artifact_manifest_path(checkpoint_dir)
    if local_manifest:
        return _load_json_if_exists(local_manifest), local_manifest
    if not repo_id:
        return None, None

    cache_dir = os.path.join(os.path.dirname(os.path.normpath(checkpoint_dir)), ".hf_artifacts")
    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename="artifacts/artifacts_manifest.json",
            repo_type="model",
            local_dir=cache_dir,
        )
    except Exception:
        return None, None
    return _load_json_if_exists(manifest_path), manifest_path


def _validate_artifacts_or_die(artifact_manifest, tokenizer_model_path, model_config_path, data_cfg):
    errors = []
    tokenizer_info = (artifact_manifest.get("tokenizer") or {})
    expected_tokenizer_sha = tokenizer_info.get("sha256")
    if expected_tokenizer_sha:
        if not os.path.exists(tokenizer_model_path):
            errors.append(f"tokenizer model missing at '{tokenizer_model_path}'")
        else:
            actual_tokenizer_sha = _file_sha256(tokenizer_model_path)
            if actual_tokenizer_sha != expected_tokenizer_sha:
                errors.append(
                    "tokenizer sha256 mismatch: "
                    f"expected={expected_tokenizer_sha}, got={actual_tokenizer_sha}"
                )

    model_cfg_info = artifact_manifest.get("model_config") or {}
    expected_model_cfg_sha = model_cfg_info.get("sha256")
    if expected_model_cfg_sha:
        if not os.path.exists(model_config_path):
            errors.append(f"model config missing at '{model_config_path}'")
        else:
            actual_model_cfg_sha = _file_sha256(model_config_path)
            if actual_model_cfg_sha != expected_model_cfg_sha:
                errors.append(
                    "model config sha256 mismatch: "
                    f"expected={expected_model_cfg_sha}, got={actual_model_cfg_sha}"
                )

    data_meta_expected = (((artifact_manifest.get("data") or {}).get("data_meta") or {}).get("tokenizer_sha256"))
    val_bin = data_cfg.get("val_bin")
    data_meta_path = os.path.join(os.path.dirname(val_bin) or ".", "data_meta.json") if val_bin else None
    data_meta = _load_json_if_exists(data_meta_path)
    if data_meta_expected and isinstance(data_meta, dict):
        local_data_tokenizer_sha = data_meta.get("tokenizer_sha256")
        if local_data_tokenizer_sha and local_data_tokenizer_sha != data_meta_expected:
            errors.append(
                "data_meta tokenizer sha256 mismatch: "
                f"expected={data_meta_expected}, got={local_data_tokenizer_sha}"
            )

    if errors:
        detail = "\n- ".join(errors)
        raise SystemExit(
            "Artifact validation failed. Refusing to evaluate with mismatched artifacts.\n"
            f"- {detail}\n"
            "Use the checkpoint's matching tokenizer/config, or rerun with --skip_artifact_validation "
            "only if you intentionally accept invalid comparisons."
        )


def load_memmap(path, dtype):
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, batch_size, block_size, rng, device):
    max_idx = len(data) - block_size + 1
    if max_idx <= 0:
        raise SystemExit("Validation data file too small for block_size.")
    idx = rng.integers(0, max_idx, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in idx])
    x = torch.from_numpy(x).long().to(device, non_blocking=True)
    return x, x


@torch.no_grad()
def eval_loss(model, data, batch_size, block_size, device, batches, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, block_size, rng, device)
        outputs = model(input_ids=x, labels=y)
        losses.append(outputs.loss.detach().cpu().item())
    return sum(losses) / len(losses) if losses else float("nan")


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
):
    model.eval()
    input_ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_ids], device=device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k and top_k > 0:
        gen_kwargs["top_k"] = top_k
    if repetition_penalty and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if min_new_tokens and min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = min_new_tokens
    outputs = model.generate(input_ids=input_ids, **gen_kwargs)
    return _decode_tokens(sp, outputs[0].tolist(), special_ids, eos_id)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and sample.")
    parser.add_argument("--model_config", default="configs/model_100m.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer_model", default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--skip_artifact_validation",
        action="store_true",
        help="Skip tokenizer/config/data hash validation against artifacts manifest (not recommended).",
    )
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.train_config)

    # Load defaults from config, allow CLI overrides
    eval_cfg = train_cfg.get("eval", {})
    args.checkpoint = args.checkpoint or eval_cfg.get("checkpoint", "runs/llama-100m/final")
    args.tokenizer_model = args.tokenizer_model or eval_cfg.get("tokenizer_model", "tokenizer/spm.model")
    args.batches = args.batches if args.batches is not None else eval_cfg.get("batches", 200)
    args.prompt = args.prompt or eval_cfg.get("prompt", "The quick brown fox")
    args.min_new_tokens = args.min_new_tokens if args.min_new_tokens is not None else eval_cfg.get("min_new_tokens", 16)
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else eval_cfg.get("max_new_tokens", 100)
    args.temperature = args.temperature if args.temperature is not None else eval_cfg.get("temperature", 0.7)
    args.top_p = args.top_p if args.top_p is not None else eval_cfg.get("top_p", 0.9)
    args.top_k = args.top_k if args.top_k is not None else eval_cfg.get("top_k", 50)
    args.repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else eval_cfg.get("repetition_penalty", 1.1)
    data_cfg = train_cfg["data"]
    optim_cfg = train_cfg["training"]
    checkpoint_upload_cfg = train_cfg.get("checkpoint_upload", {})

    if not args.skip_artifact_validation:
        repo_id = checkpoint_upload_cfg.get("repo_id")
        artifact_manifest, manifest_source = _load_artifact_manifest(args.checkpoint, repo_id)
        if artifact_manifest is None:
            raise SystemExit(
                "Missing artifacts manifest for checkpoint validation. "
                "Expected local artifacts_manifest.json or remote "
                "'artifacts/artifacts_manifest.json' in checkpoint_upload.repo_id."
            )
        _validate_artifacts_or_die(artifact_manifest, args.tokenizer_model, args.model_config, data_cfg)
        print(f"[artifact] validated with {manifest_source}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM(config).to(device)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    model_path = os.path.join(args.checkpoint, "model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise SystemExit("No model.pt or pytorch_model.bin found in checkpoint.")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)
    special_ids = {
        config.bos_token_id,
        config.eos_token_id,
        config.pad_token_id,
        sp.unk_id(),
    }
    special_ids = {token_id for token_id in special_ids if token_id is not None}

    val_data = load_memmap(data_cfg["val_bin"], np.dtype(data_cfg["dtype"]))
    batch_size = args.batch_size if args.batch_size is not None else optim_cfg["micro_batch_size"]
    loss = eval_loss(
        model,
        val_data,
        batch_size,
        data_cfg["block_size"],
        device,
        args.batches,
        optim_cfg["seed"],
    )
    ppl = float("nan") if math.isnan(loss) else math.exp(min(20, loss))

    print(f"val_loss={loss:.4f} ppl={ppl:.2f}")
    print(
        sample_generate(
            model,
            sp,
            args.prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.min_new_tokens,
            special_ids,
            config.eos_token_id,
        )
    )


if __name__ == "__main__":
    main()
