#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
import yaml
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokens/sec.")
    parser.add_argument("--model_config", default="configs/model_100m.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output_path", default="runs/throughput.json")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.train_config)
    data_cfg = train_cfg["data"]
    optim_cfg = train_cfg["training"]

    accelerator = Accelerator(mixed_precision=optim_cfg.get("precision", "bf16"))
    device = accelerator.device

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

    model = LlamaForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["learning_rate"])

    micro_batch = optim_cfg["micro_batch_size"]
    block_size = data_cfg["block_size"]
    grad_accum = optim_cfg["grad_accum_steps"]
    world_size = accelerator.num_processes
    tokens_per_step = micro_batch * block_size * grad_accum * world_size

    def run_step():
        loss_total = 0.0
        for _ in range(grad_accum):
            input_ids = torch.randint(
                0,
                model_cfg["vocab_size"],
                (micro_batch, block_size),
                device=device,
            )
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            loss_total += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        return loss_total / grad_accum

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(args.warmup_steps):
        run_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.steps):
        run_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = tokens_per_step * args.steps / max(1e-6, elapsed)
    result = {
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_step": tokens_per_step,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "block_size": block_size,
        "world_size": world_size,
        "elapsed_sec": elapsed,
        "steps": args.steps,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if accelerator.is_main_process:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
