# Conversational post-training and evaluation

This is the operating contract for turning a base checkpoint into a chat model.
It does not authorize a paid run. The base checkpoint, corpus, tokenizer,
throughput, stop criteria, and cost cap must still pass their normal gates.

## Training data contract

Input is JSONL with one conversation per line:

```json
{
  "id": "stable-record-id",
  "source_id": "qwen_synthetic_v1",
  "source_name": "Qwen synthetic conversation pilot",
  "license": "Apache-2.0",
  "license_evidence": "URL or policy snapshot identifier",
  "synthetic": true,
  "generator": {
    "provider": "fireworks",
    "model": "exact model identifier",
    "prompt_template_sha256": "..."
  },
  "quality_score": 0.9,
  "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain compound interest."},
    {"role": "assistant", "content": "It is interest earned on prior interest..."}
  ]
}
```

Records must end in an assistant turn. System messages are optional and may
appear only first. User and assistant turns alternate. Invalid conversations
fail preparation rather than being silently repaired.

Generate a small Fireworks pilot without allowing evaluation prompts into the
generator:

```bash
uv run generate-chat-data \
  --output data/chat/raw/fireworks-pilot-v1.jsonl \
  --count_per_seed 10 \
  --max_records 120 \
  --confirm_spend
```

The generator is inert without `--confirm_spend`. It records the exact model,
prompt-template hash, scenario recipe, license evidence, sampling parameters,
and response hash. Twelve-word overlap against the frozen conversational suite
is rejected. Use `--dry_run` to inspect planned scenario prompts without making
API calls. Generated records still require random manual inspection; generator
acceptance is only structural and decontamination validation.

Create a scrollable review pack and export only explicit keeps:

```bash
uv run chat-review create \
  --input data/chat/raw/fireworks-pilot-v1.jsonl \
  --output_dir data/chat/review/fireworks-pilot-v1

# After filling review.csv:
uv run chat-review apply \
  --input data/chat/raw/fireworks-pilot-v1.jsonl \
  --review_csv data/chat/review/fireworks-pilot-v1/review.csv \
  --output data/chat/accepted/fireworks-pilot-v1.jsonl \
  --require_complete
```

Blank decisions never enter the accepted export. Scores and notes are retained
as record metadata so later quality analyses can be tied to the exact review.

Prepare immutable indexed shards:

```bash
uv run prepare-chat-data \
  --input data/chat/accepted/fireworks-pilot-v1.jsonl \
  --output_dir data/chat/pilot-v1 \
  --tokenizer tokenizer/spm.model \
  --validation_fraction 0.02 \
  --max_tokens 2048 \
  --require_human_keep
```

The output records:

- the exact input and tokenizer hashes;
- a versioned chat template;
- deterministic content-hash train/validation assignment;
- source and license metadata;
- assistant supervision spans and supervised-token counts;
- immutable indexed-shard and recipe hashes.

The trainer treats documents without supervision spans as ordinary pretraining
documents. Documents with spans compute loss only on assistant content and the
assistant end marker. User, system, and assistant-header tokens remain context
but never become prediction targets.

## SFT runs

`configs/train_350m_chat_sft.yaml` is a non-runnable template until its explicit
base checkpoint and corpus paths exist. A new post-training run loads base
weights while starting a fresh optimizer:

```bash
uv run train \
  --model_config configs/model_350m.yaml \
  --train_config configs/train_350m_chat_sft.yaml \
  --initialize_from runs/gpt-350m-base/final
```

Initialization requires `model.pt` and an artifact manifest. Model-config and
tokenizer hashes must match. The child artifact manifest stores the parent
model hash and manifest hash. Exact post-training resume continues to use
`--resume_from`.

The first SFT bakeoff uses equal supervised-assistant-token budgets:

1. inspected public-data control;
2. inspected Fireworks/Qwen synthetic conversations;
3. a mixed public and synthetic arm.

Compare checkpoints around 10M, 30M, and 100M supervised tokens. These are
measurement points, not commitments to consume all available data.

## Frozen evaluation

`evals/conversation/v1.jsonl` is evaluation-only and must be excluded from
training-data generation, retrieval, and prompt seeding. It covers:

- clarification under ambiguity;
- context corrections and reference tracking;
- instruction retention across turns;
- disagreement and false-premise correction;
- natural empathy without canned claims;
- format switching and exact constraints;
- uncertainty and non-fabrication;
- style adjustment and concise follow-ups.

Generate a reproducible local run:

```bash
uv run conversation-eval generate-local \
  --checkpoint runs/gpt-350m-chat-sft/final \
  --model_config configs/model_350m.yaml \
  --tokenizer tokenizer/spm.model \
  --output runs/conversation-eval/sft.jsonl \
  --temperature 0
```

Inspect deterministic results:

```bash
uv run conversation-eval score \
  --input runs/conversation-eval/sft.jsonl
```

Create a reproducibly blinded human comparison:

```bash
uv run conversation-eval build-review \
  --left runs/conversation-eval/base.jsonl \
  --right runs/conversation-eval/sft.jsonl \
  --output_dir runs/conversation-eval/review-base-vs-sft
```

The review CSV includes hidden source mappings so results can be decoded after
judgment. Reviewers mark `A`, `B`, `tie`, or `both_bad`.

## Fireworks judging

Automated pairwise judging is position-swapped. A judgment is accepted only
when both presentation orders agree; otherwise it is recorded as
`position_unstable`.

No API call occurs without explicit spend confirmation:

```bash
uv run conversation-eval judge-fireworks \
  --left runs/conversation-eval/base.jsonl \
  --right runs/conversation-eval/sft.jsonl \
  --output runs/conversation-eval/qwen-judge.jsonl \
  --max_comparisons 10 \
  --confirm_spend
```

Use at least two unrelated judge families for a release comparison and
calibrate them against blinded human labels. Results produced with an alternate
judge are internal variants and must not be presented as official AlpacaEval or
MT-Bench scores.

## Go/no-go rule

Preference training is earned only after SFT:

- improves deterministic conversation checks;
- wins blinded human comparisons;
- wins with both calibrated judge families;
- does not materially regress IFEval or the frozen base-capability suite.

Start with DPO or APO on inspected pairs. Do not begin subjective conversational
GRPO/RFT until the evaluator survives adversarial tests for verbosity,
sycophancy, format-only rewards, and position bias.
