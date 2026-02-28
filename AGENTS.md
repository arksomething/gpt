# Agent Instructions

This file provides context for AI coding assistants working on this project.

## Project Overview

This repo trains a ~100M parameter Llama-style language model from scratch. The pipeline includes:
- Tokenizer training (32k SentencePiece BPE)
- Data preparation (streaming from C4 + Wikipedia with filtering)
- Training with checkpointing
- Evaluation

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Main training script with model definition |
| `scripts/prepare_data.py` | Data streaming, filtering, tokenization |
| `scripts/train_tokenizer.py` | SentencePiece tokenizer training |
| `scripts/eval.py` | Model evaluation |
| `scripts/filters.py` | Text quality filters |
| `configs/model_100m.yaml` | Model architecture config |
| `configs/train.yaml` | Training hyperparameters |

## Available Commands (pyproject.toml scripts)

All commands are run with `uv run <command>`:

| Command | Description |
|---------|-------------|
| `uv run train` | Train the model |
| `uv run train --launch_screen` | Train in detached screen session |
| `uv run train --smoke` | Quick smoke test |
| `uv run train --resume_from <path>` | Resume from checkpoint |
| `uv run eval` | Evaluate a trained model |
| `uv run prepare-data` | Prepare tokenized training data |
| `uv run train-tokenizer` | Train the SentencePiece tokenizer |
| `uv run benchmark` | Benchmark throughput (tokens/sec) |
| `uv run tokenizer-repl` | Interactive tokenizer REPL |
| `uv run scan-bins` | Inspect .bin data files |
| `uv run ls` | List project files/info |

### Train command flags

```
--model_config PATH      Model config (default: configs/model_100m.yaml)
--train_config PATH      Training config (default: configs/train.yaml)
--resume_from PATH       Resume from checkpoint directory
--resume_from_slot NAME  Resume from named checkpoint slot
--smoke                  Run quick smoke test
--launch_screen          Run training in detached screen session
--screen_name NAME       Custom screen session name
```

## Data

Pre-tokenized data is hosted on HuggingFace:
- Repo: `ark296/gpt-training-data`
- Files: `train.bin` (1B tokens), `val.bin` (10M tokens), `data_meta.json`
- Format: numpy memmap of uint16 token IDs
- Tokenizer: `tokenizer/spm.model` (32k vocab)

Download:
```bash
huggingface-cli download ark296/gpt-training-data --repo-type dataset --local-dir ./data
```

## Model Architecture

- ~100M parameters
- Llama-style (RMSNorm, RoPE, SwiGLU)
- See `configs/model_100m.yaml` for full config

## Quick Start

```bash
# Setup
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Download pre-tokenized data
huggingface-cli download ark296/gpt-training-data --repo-type dataset --local-dir ./data

# Train (in screen for long runs)
uv run train --launch_screen

# Attach to watch training
screen -r train

# Detach: Ctrl+A, then D
```

## Cloud Training (Google Colab)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env

# Clone and setup
git clone https://github.com/arksomething/gpt.git && cd gpt
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Download data
huggingface-cli download ark296/gpt-training-data --repo-type dataset --local-dir ./data

# Train
uv run train --launch_screen
```

### GPU Performance Reference

| GPU | FP16 TFLOPS | Training Time (100M/1B) |
|-----|-------------|------------------------|
| T4 | 65 | ~8-10 hrs |
| P100 | 19 | ~12 hrs |
| V100 | 125 | ~4-5 hrs |
| A100 | 312 | ~2 hrs |
| RTX 4090 | 330 | ~1.5 hrs |

## Common Tasks

### Resume from checkpoint
```bash
uv run train --resume_from runs/llama-100m/checkpoint-XXXXX
```

### Run smoke test
```bash
uv run train --smoke
```

### Evaluate model
```bash
uv run eval
```

### View training log
```bash
less +F runs/llama-100m/train.log
```

## Dependencies

Managed with `uv`. Key packages:
- torch (CUDA-enabled)
- sentencepiece
- accelerate
- huggingface_hub
- numpy, tqdm, pyyaml

## File Formats

- `.bin` files: numpy memmap, dtype=uint16, token IDs
- `data_meta.json`: dataset statistics and filter config
- `tokenizer_meta.json`: vocab size, special tokens
