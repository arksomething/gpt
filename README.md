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

Stream from C4 (30%), Wikipedia (45%), and Project Gutenberg (25%), apply
aggressive filtering, deduplicate, tokenize, and write memmaps:

```
python scripts/prepare_data.py \
  --tokenizer_model tokenizer/spm.model \
  --out_dir data \
  --train_tokens 2000000000 \
  --val_tokens 20000000
```

The pipeline applies:
- **Global filters**: length bounds (300-80k chars), alpha ratio, repeated chars,
  weird symbols, line structure, boilerplate blacklist
- **C4-specific**: stricter alpha ratio (0.70), punctuation spam, web junk keywords,
  nav patterns, paragraph quality, entropy bounds
- **Wikipedia-specific**: strip tables/refs/external links, filter list-heavy content
- **Gutenberg-specific**: strip Project Gutenberg boilerplate, normalize line wrapping
- **Deduplication**: MinHash near-duplicate removal (85% similarity threshold)

Configure weights and filter params in `configs/train.yaml` under `data_prep`.

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

## Pre-tokenized Data

Pre-tokenized training data (1B tokens) is available on HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download ark296/gpt-training-data --repo-type dataset --local-dir ./data
```

Dataset contents:
- `train.bin` - 1,000,000,231 tokens (2.1 GB)
- `val.bin` - 10,001,221 tokens (21 MB)
- `data_meta.json` - metadata and filter statistics

Data mix: 40% C4, 60% Wikipedia (filtered and deduplicated).

## Cloud Training (Google Colab)

### Setup SSH Access to Colab

1. Create a new Colab notebook and select GPU runtime (`Runtime → Change runtime type → GPU`)

2. Run this cell to enable SSH:
```python
!pip install colab-ssh --upgrade
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_secret_password")
```

3. Install cloudflared on your local machine (one-time):
```bash
# Linux
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/
```

4. Add to `~/.ssh/config`:
```
Host *.trycloudflare.com
    HostName %h
    User root
    Port 22
    ProxyCommand /usr/local/bin/cloudflared access ssh --hostname %h
```

5. SSH in using the hostname from Colab output:
```bash
ssh your-random-hostname.trycloudflare.com
```

### Training on Colab

```bash
# Setup
pip install torch numpy sentencepiece huggingface_hub tqdm pyyaml accelerate

# Clone repo
git clone https://github.com/arksomething/gpt.git && cd gpt

# Download pre-tokenized data
huggingface-cli download ark296/gpt-training-data --repo-type dataset --local-dir ./data

# Check GPU
nvidia-smi

# Train
accelerate launch scripts/train.py \
  --model_config configs/model_100m.yaml \
  --train_config configs/train.yaml
```

### Other Cloud Options

| Platform | GPU | Cost | Notes |
|----------|-----|------|-------|
| Google Colab Pro (Education) | T4/V100/A100 | Free (students) | 100 compute units/month |
| Kaggle | P100/T4 | Free | 30 hrs/week |
| Vast.ai | RTX 4090 | ~$0.40/hr | Consumer GPUs, cheap |
| Azure for Students | T4/V100 | $100 credits | .edu email required |
| Google Cloud | Any | $300 credits | New accounts |

### Estimated Training Times

| GPU | Time for 100M model / 1B tokens |
|-----|--------------------------------|
| T4 | ~8-10 hours |
| V100 | ~4-5 hours |
| RTX 4090 | ~1.5 hours |
| A100 | ~2 hours |

## Dataset licensing notes

- C4: ODC-By 1.0 + Common Crawl Terms of Use
- Wikipedia: CC-BY-SA
- Project Gutenberg (pg19): Public domain (US)

Review dataset terms before commercial use.
