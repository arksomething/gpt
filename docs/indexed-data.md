# Indexed training data

New experiments should use the indexed corpus format in
`scripts/indexed_shards.py`. The historical flat `.bin` loader remains
available only for reproducing old runs.

## Corpus layout

An immutable corpus directory contains:

```text
manifest.json
documents.jsonl
tokens-00000.bin
tokens-00001.bin
...
```

The manifest fingerprints the tokenizer, preparation recipe, document index,
and every token shard. Each document record preserves its source ID, token
range, content hash, optional quality score, and arbitrary JSON metadata.
Documents never span shard files.

`IndexedShardReader` validates the manifest fingerprint, file hashes, byte
sizes, source IDs, document ordering, and token ranges before exposing data.
`IndexedShardWriter` builds in a temporary sibling directory and refuses to
overwrite an existing corpus.

## Sampling and packing

`IndexedCorpusSampler` is deterministic given a NumPy random generator:

1. choose one source for each batch row using the configured mixture weights;
2. sample documents uniformly within that source;
3. fill the row with document segments without padding;
4. restart position IDs at every document boundary;
5. set the label at every segment start to `-100`.

With Transformers 5, reset position IDs and `use_cache=False` produce
block-diagonal causal attention. Tokens cannot attend across document
boundaries, and ignored boundary labels prevent cross-document next-token
targets. A regression test verifies that changing one packed document leaves
the other document's logits unchanged.

The sampler holds no private random state. The trainer's existing per-rank
NumPy RNG sidecar is enough to reproduce the exact next indexed batch after a
checkpoint resume.

## Trainer configuration

Indexed mode takes precedence over flat memmap and Hugging Face streaming:

```yaml
data:
  block_size: 2048
  indexed:
    enabled: true
    train_dir: data/indexed/train
    val_dir: data/indexed/validation
    tokenizer_sha256: null
    recipe_sha256: null
    verify_hashes: true
    source_weights:
      fineweb_edu: 0.50
      wikipedia: 0.20
      writing: 0.30
    validation_source_weights: null
```

When a tokenizer path is configured elsewhere in the training configuration,
its actual SHA-256 must match both corpus manifests. Optional explicit hashes
above add another fail-fast expectation.

Every run artifact manifest records both indexed corpus fingerprints, manifest
file hashes, document/token counts, recipe hashes, and mixture weights.
Checkpoint progress sidecars bind the model config, training config, tokenizer,
corpora, and mixture weights into a compatibility fingerprint. Exact resume
fails before training if any of those inputs differ.

## Producing indexed corpora

Set `data_prep.output_format: indexed` or pass the CLI override:

```bash
uv run prepare-data \
  --output_format indexed \
  --out_dir data/probe-v1 \
  --train_tokens 500000000 \
  --val_tokens 5000000 \
  --overwrite
```

This produces `data/probe-v1/train`, `data/probe-v1/validation`, and
`data/probe-v1/data_meta.json`. Source identity, chunk hashes, parent-document
hashes, chunk positions, and EOS boundaries survive filtering, shuffling, and
parallel tokenization.

Raw documents are assigned to train or validation by a seeded content hash
before filtering and chunking. Restarting the same upstream stream therefore
cannot put the same raw text in both splits. The validation fraction is part of
the preparation-recipe fingerprint.

Indexed construction is transactional and resumable. During preparation,
completed shards and the committed prefix of `documents.jsonl` live under
`train.staging` or `validation.staging` with a fully valid manifest. A restart
with `--resume` replays and verifies the committed source IDs, content hashes,
and token sequences before appending. Uncommitted tail bytes are discarded;
committed shard files are reused byte-for-byte. Completion atomically renames
the staging directory to the immutable final corpus.

Do not edit a staging directory manually. If both a final and staging directory
exist for one split, preparation stops rather than guessing which is valid.
