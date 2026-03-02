#!/usr/bin/env python3
"""Simple inference script for text generation."""

import argparse
import hashlib
import json
import os
import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM, LlamaConfig
import yaml


DEFAULT_CONFIG = "configs/train.yaml"


def load_config(config_path):
    """Load full train config yaml file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f) or {}
    return full_cfg


def load_model(checkpoint_path, model_config_path, device):
    """Load model from checkpoint."""
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)["model"]
    
    config = LlamaConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"]),
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        hidden_act=model_cfg.get("hidden_act", "silu"),
        attention_bias=model_cfg.get("attention_bias", False),
        mlp_bias=model_cfg.get("mlp_bias", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        bos_token_id=model_cfg.get("bos_token_id", 1),
        eos_token_id=model_cfg.get("eos_token_id", 2),
        pad_token_id=model_cfg.get("pad_token_id", 3),
    )
    
    model = LlamaForCausalLM(config)
    
    # Load weights
    state_dict = torch.load(f"{checkpoint_path}/model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config


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


def _validate_artifacts_or_die(artifact_manifest, tokenizer_path, model_config_path):
    errors = []
    tokenizer_info = artifact_manifest.get("tokenizer") or {}
    expected_tokenizer_sha = tokenizer_info.get("sha256")
    if expected_tokenizer_sha:
        if not os.path.exists(tokenizer_path):
            errors.append(f"tokenizer model missing at '{tokenizer_path}'")
        else:
            actual_tokenizer_sha = _file_sha256(tokenizer_path)
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

    if errors:
        detail = "\n- ".join(errors)
        raise SystemExit(
            "Artifact validation failed. Refusing to run inference with mismatched artifacts.\n"
            f"- {detail}\n"
            "Use the checkpoint's matching tokenizer/config, or rerun with --skip_artifact_validation "
            "only if you intentionally accept invalid outputs."
        )


def _truncate_at_eos(token_ids, eos_id):
    if eos_id is None:
        return token_ids
    try:
        eos_idx = token_ids.index(eos_id)
    except ValueError:
        return token_ids
    return token_ids[:eos_idx]


def _decode_tokens(tokenizer, token_ids, special_ids, eos_id):
    token_ids = _truncate_at_eos(token_ids, eos_id)
    if special_ids:
        token_ids = [t for t in token_ids if t not in special_ids]
    return tokenizer.decode(token_ids)


def validate_tokenizer_model_config(tokenizer, config):
    """Fail fast on tokenizer/model special-id mismatches."""
    mismatches = []
    if tokenizer.vocab_size() != config.vocab_size:
        mismatches.append(
            f"vocab_size mismatch: tokenizer={tokenizer.vocab_size()} model={config.vocab_size}"
        )
    id_pairs = (
        ("bos_token_id", tokenizer.bos_id()),
        ("eos_token_id", tokenizer.eos_id()),
        ("pad_token_id", tokenizer.pad_id()),
    )
    for attr, tok_id in id_pairs:
        model_id = getattr(config, attr, None)
        if model_id is None or tok_id is None or tok_id < 0:
            continue
        if int(model_id) != int(tok_id):
            mismatches.append(f"{attr} mismatch: tokenizer={tok_id} model={model_id}")
    if mismatches:
        detail = "\n- ".join(mismatches)
        raise SystemExit(
            "Tokenizer/model mismatch detected. This can cause gibberish outputs.\n"
            f"- {detail}"
        )


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.8, top_p=0.95, top_k=50, repetition_penalty=1.2):
    """Generate text from prompt."""
    input_token_ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_token_ids], device=device)

    do_sample = temperature is not None and temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k
    if repetition_penalty and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    outputs = model.generate(input_ids=input_ids, **gen_kwargs)
    generated = outputs[0].tolist()
    continuation = generated[len(input_token_ids) :]
    special_ids = {
        model.config.bos_token_id,
        model.config.eos_token_id,
        model.config.pad_token_id,
        tokenizer.unk_id(),
    }
    special_ids = {token_id for token_id in special_ids if token_id is not None}
    return _decode_tokens(tokenizer, continuation, special_ids, model.config.eos_token_id)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a prompt")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--model_config", type=str, default=None, help="Model config")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (1.0=none, 1.2=moderate)")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cuda/cpu)")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL mode (keeps model loaded)")
    parser.add_argument(
        "--skip_artifact_validation",
        action="store_true",
        help="Skip tokenizer/config hash validation against artifacts manifest (not recommended).",
    )
    args = parser.parse_args()
    
    # Load config file
    full_cfg = load_config(args.config)
    inference_cfg = full_cfg.get("inference", {})
    checkpoint_upload_cfg = full_cfg.get("checkpoint_upload", {})

    # Merge config with CLI args (CLI takes precedence)
    checkpoint = args.checkpoint or inference_cfg.get("checkpoint", "runs/llama-100m/final")
    model_config = args.model_config or inference_cfg.get("model_config", "configs/model_100m.yaml")
    tokenizer_path = args.tokenizer or inference_cfg.get("tokenizer", "tokenizer/spm.model")
    prompt = args.prompt or inference_cfg.get("prompt", "The quick brown fox")
    max_tokens = args.max_tokens if args.max_tokens is not None else inference_cfg.get("max_tokens", 100)
    temperature = args.temperature if args.temperature is not None else inference_cfg.get("temperature", 0.8)
    top_p = args.top_p if args.top_p is not None else inference_cfg.get("top_p", 0.95)
    top_k = args.top_k if args.top_k is not None else inference_cfg.get("top_k", 50)
    repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else inference_cfg.get("repetition_penalty", 1.2)
    device_cfg = args.device or inference_cfg.get("device", "auto")
    
    # Auto-detect device
    if device_cfg == "auto" or device_cfg is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg
    print(f"Using device: {device}")

    if not args.skip_artifact_validation:
        repo_id = checkpoint_upload_cfg.get("repo_id")
        artifact_manifest, manifest_source = _load_artifact_manifest(checkpoint, repo_id)
        if artifact_manifest is None:
            raise SystemExit(
                "Missing artifacts manifest for checkpoint validation. "
                "Expected local artifacts_manifest.json or remote "
                "'artifacts/artifacts_manifest.json' in checkpoint_upload.repo_id."
            )
        _validate_artifacts_or_die(artifact_manifest, tokenizer_path, model_config)
        print(f"[artifact] validated with {manifest_source}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    
    # Load model
    print("Loading model...")
    model, _ = load_model(checkpoint, model_config, device)
    validate_tokenizer_model_config(tokenizer, model.config)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    if args.repl:
        # Interactive REPL mode
        print("\n" + "=" * 50)
        print("REPL mode - model loaded and ready")
        print("Commands: /quit, /temp <val>, /tokens <val>, /topk <val>, /topp <val>, /rep <val>")
        print("=" * 50 + "\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                    print("Exiting...")
                    break
                elif cmd == "/temp" and len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                elif cmd == "/tokens" and len(parts) > 1:
                    try:
                        max_tokens = int(parts[1])
                        print(f"Max tokens set to {max_tokens}")
                    except ValueError:
                        print("Invalid max_tokens value")
                elif cmd == "/topk" and len(parts) > 1:
                    try:
                        top_k = int(parts[1])
                        print(f"Top-k set to {top_k}")
                    except ValueError:
                        print("Invalid top_k value")
                elif cmd == "/topp" and len(parts) > 1:
                    try:
                        top_p = float(parts[1])
                        print(f"Top-p set to {top_p}")
                    except ValueError:
                        print("Invalid top_p value")
                elif cmd == "/rep" and len(parts) > 1:
                    try:
                        repetition_penalty = float(parts[1])
                        print(f"Repetition penalty set to {repetition_penalty}")
                    except ValueError:
                        print("Invalid repetition_penalty value")
                elif cmd == "/settings":
                    print(f"temperature={temperature}, max_tokens={max_tokens}, top_k={top_k}, top_p={top_p}, rep={repetition_penalty}")
                elif cmd == "/help":
                    print("Commands: /quit, /temp <val>, /tokens <val>, /topk <val>, /topp <val>, /rep <val>, /settings, /help")
                else:
                    print(f"Unknown command: {cmd}")
                continue
            
            # Generate from prompt
            output = generate(
                model, tokenizer, user_input, device,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            print(output)
            print()
    else:
        # Single generation mode
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        output = generate(
            model, tokenizer, prompt, device,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        
        print(output)


if __name__ == "__main__":
    main()
