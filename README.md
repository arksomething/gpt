# 100M LLM Training (RTX 4090)

This repo trains a ~100M parameter Llama-style model from scratch on a public
data mix using rented RTX 4090s. It includes tokenizer training, data prep,
throughput benchmarking, training, and evaluation.

## Setup

1. Create a virtualenv and install deps with `uv`:

```
uv venv
uv sync
```

2. Install a CUDA-enabled PyTorch build for your GPU from
https://pytorch.org/get-started/locally/ (example below uses CUDA 12.4):

```
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Tokenizer

Train a 32k SentencePiece BPE:

```
python scripts/train_tokenizer.py --output_dir tokenizer
```

Outputs: `tokenizer/spm.model`, `tokenizer/spm.vocab`, `tokenizer/tokenizer_meta.json`

## Data preparation

Stream C4 + Wikipedia, tokenize, and write memmaps:

```
python scripts/prepare_data.py \
  --tokenizer_model tokenizer/spm.model \
  --out_dir data \
  --train_tokens 2000000000 \
  --val_tokens 20000000
```

Outputs: `data/train.bin`, `data/val.bin`, `data/data_meta.json`

## Throughput benchmark

Measure tokens/sec and write `runs/throughput.json`:

```
python scripts/benchmark_throughput.py --output_path runs/throughput.json
```

Update `configs/train.yaml` with your **actual** `hourly_rate` and desired
`target_tokens`. Training will refuse to start if projected cost > `max_cost`
and `runs/throughput.json` exists.

## Training

Single GPU:

```
accelerate launch scripts/train.py \
  --model_config configs/model_100m.yaml \
  --train_config configs/train.yaml
```

Sanity checks are configured under `checks` in `configs/train.yaml`:
- `overfit_microset` runs before the main loop and stops if loss won't drop.
- `fixed_prompt` appends a deterministic sample each eval interval to
  `runs/llama-100m/fixed_prompt_samples.txt` by default.

Run detached in screen (useful for long runs):

```
screen -dmS train accelerate launch scripts/train.py \
  --model_config configs/model_100m.yaml \
  --train_config configs/train.yaml
```

If you launch `scripts/train.py` directly, you can add `--launch_screen` to
auto-detach into `screen` (or `tmux` if screen isn't available).

Smoke test:

```
accelerate launch scripts/train.py --smoke
```

Checkpoints are written to `runs/llama-100m/`.

## Evaluation

```
python scripts/eval.py \
  --checkpoint runs/llama-100m/final \
  --tokenizer_model tokenizer/spm.model
```

## Dataset licensing notes

- C4: ODC-By 1.0 + Common Crawl Terms of Use
- Wikipedia: CC-BY-SA

Review dataset terms before commercial use.
# gpt
