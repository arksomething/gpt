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

## Training

Uses PyTorch with accelerate. Key training features:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Checkpointing every N steps
- Cosine learning rate schedule with warmup

Run training:
```bash
accelerate launch scripts/train.py \
  --model_config configs/model_100m.yaml \
  --train_config configs/train.yaml
```

## Cloud Training Notes

### Google Colab SSH Setup
1. Enable SSH with `colab-ssh` package
2. Use cloudflared tunnel for connection
3. Data persists only during session - save checkpoints to HuggingFace

### GPU Performance Reference

| GPU | FP16 TFLOPS | Training Time (100M/1B) |
|-----|-------------|------------------------|
| T4 | 65 | ~8-10 hrs |
| P100 | 19 | ~12 hrs |
| V100 | 125 | ~4-5 hrs |
| A100 | 312 | ~2 hrs |
| RTX 4090 | 330 | ~1.5 hrs |

## Common Tasks

### Resuming from checkpoint
```bash
accelerate launch scripts/train.py \
  --model_config configs/model_100m.yaml \
  --train_config configs/train.yaml \
  --resume runs/llama-100m/checkpoint-XXXXX
```

### Uploading results to HuggingFace
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="runs/llama-100m", repo_id="username/model-name", repo_type="model")
```

### Running smoke test
```bash
accelerate launch scripts/train.py --smoke
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
