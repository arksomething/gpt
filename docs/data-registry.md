# High-Quality Writing Source Registry

This registry is the intake boundary for the project's curated writing corpus.
It favors broad discovery while preventing public availability from being
mistaken for permission, provenance, or quality.

The machine-readable source of truth is
[`data/sources.yaml`](../data/sources.yaml). Generated text stays ignored by
Git; each acquisition run emits a reviewable manifest with document hashes,
source revisions, policy snapshots, and retrieval details.

## Operating model

Candidates are unlimited, but at most 25 sources may be under active inspection
at once.

| Tier | Meaning | Permitted action |
|---|---|---|
| A | Open or public-domain source with documented redistribution terms | Acquire a capped sample and inspect |
| B | Plausibly usable for private research training, but raw redistribution is not cleared | Use only through an official export/API after recording the risk |
| C | Copyright or platform terms require permission | Record metadata and request permission; do not ingest |
| D | Explicit block, unacceptable provenance, or incompatible terms | Exclude |

Redistribution, training, and model-weight release are separate questions.
Tier A means evidence supports redistribution of the source text under the
recorded conditions; it is not a conclusion about every jurisdiction or about
the license of trained weights.

## First acquisition batch

The initial run deliberately mixes writing modes:

| Source | Primary signal |
|---|---|
| Gwern | Long technical and analytical essays |
| Standard Ebooks | Carefully proofread public-domain books |
| Public Domain Review | Edited contemporary essays |
| Our World in Data | Data-grounded explanatory writing |
| Congressional Research Service | Professional, nonpartisan policy analysis |
| PLOS | Contemporary open scientific prose |
| DOAB through Common Pile | Modern open-access monographs |
| Project Gutenberg through Common Pile | Broad public-domain long-form prose |
| Filtered Wikimedia through Common Pile | Factual reference writing |
| Open-license peS2o through Common Pile | Academic prose |
| Stack Exchange through Common Pile | Expert question-and-answer explanations |

The default acquisition is only 12 documents per source. It is an inspection
set, not training data and not authorization to fetch each source in full.

Run it with:

```bash
uv run python -m scripts.acquire_writing_samples
```

List or select sources:

```bash
uv run python -m scripts.acquire_writing_samples --list
uv run python -m scripts.acquire_writing_samples \
  --sources gwern,crs_reports,plos \
  --docs-per-source 12
```

## Provenance requirements

Every acquired document must retain:

- source ID and stable document ID;
- canonical URL or dataset identifier;
- source revision or retrieval timestamp;
- author and publication date when available;
- document-level license metadata when available;
- normalized-text SHA-256;
- the immutable acquisition-run manifest.

For websites, archive the contemporaneous `robots.txt`, relevant terms page,
AI content signal or RSL document when present, and the HTTP acquisition
result. Prefer official APIs, exports, dumps, repositories, and bulk-download
channels over crawling.

Raw sources remain in separate shards. Do not concatenate sources before
inspection, licensing review, cross-source deduplication, and evaluation
decontamination.

The preparation pipeline assigns raw documents to train or validation using a
seeded content hash before filtering or chunking. This prevents the previous
failure mode where separately restarted streaming datasets could place the same
opening documents in both splits.

## Inspection funnel

### Quick screen

Review 12 seeded-random documents per source. Reject immediately when the
sample is dominated by extraction artifacts, non-prose, OCR corruption,
template repetition, SEO filler, or synthetic text.

### Deep review

Review 40 documents for a boutique source and 100 for a potential trunk source.
Two readers independently grade a shared calibration subset using 0–2 scores:

- sentence-level clarity;
- argument or narrative structure;
- factual discipline, including two spot-checked claims;
- extraction and OCR artifacts;
- voice distinctness;
- synthetic-text suspicion;
- ideological or stylistic concentration;
- incremental value over the general web baseline.

Readers should reach Cohen's kappa of at least 0.6 on 20 shared calibration
documents before their independent labels are used as gates.

### Acceptance gates

A source advances only when:

- at least 80% of reviewed documents are artifact-clean;
- at least 70% pass the writing-quality threshold;
- serious synthetic suspicion is no more than 5%;
- license and policy evidence is archived;
- overlap with other sources and evaluations is measured;
- its intended role in the mixture is explicit.

Any provenance gap, unresolved required permission, machine-readable training
prohibition, or systematic contamination rejects or quarantines the source.

## Small-corpus controls

Small excellent sources primarily seed quality classifiers, rerank the web
corpus, support annealing experiments, and provide held-out writing probes.
They must not silently become a repeated ideological or stylistic template.

- Repeat a document at most four times.
- Keep the entire boutique lane at or below 10% of annealing tokens.
- Keep one author below 20% of a boutique source.
- Keep the combined rationalist cluster below 25% of the boutique lane.
- Run memorization checks on anything repeated more than twice.

Mixture weights are earned through the small-model data ablations in
[`PLAN.md`](../PLAN.md), not assigned merely because reviewers enjoy a source.
