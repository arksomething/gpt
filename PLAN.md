# 1B-Class Model Research and Execution Plan

**Status:** working plan

**Updated:** 2026-07-27

**North star:** build a credible, reproducible 1B-class language model without spending the available cloud credits on an unproven data or training recipe.

## Executive decision

The next expensive run should not be a 1B run.

The project should use a ladder of smaller models as instrumented scaling
experiments, culminating in a genuinely useful 350M release. The 1B run happens
only after those experiments establish:

1. the data mixture and tokenizer are better than the current versions;
2. training, checkpoint resume, export, and evaluation are trustworthy;
3. quality and throughput scale predictably across at least three model sizes;
4. the predicted 1B run fits the remaining compute budget with recovery margin.

This is not giving up on 1B. It is the shortest defensible route to it. Recent
work shows that small-model experiments can predict larger-model behavior, while
MobileLLM, SmolLM, and IMU-1 show that sub-billion models can be worthwhile
artifacts in their own right.

The realistic goal is **not** to beat frontier general-purpose 1B models trained
on trillions of tokens. The goal is to:

- prove an excellent end-to-end model-building process;
- release a competitive small base model and a useful post-trained variant;
- learn enough from controlled experiments to make one high-confidence 1B run;
- preserve checkpoints, data lineage, evaluations, and findings so the result
  is credible even where it does not win.

## The situation today

### What already exists

- A Llama-style implementation built on Transformers and Accelerate.
- A 32k SentencePiece BPE tokenizer.
- A roughly 102M-parameter configuration using RMSNorm, RoPE, SwiGLU, and tied
  embeddings.
- Streaming preparation from C4, Wikipedia, and FineWeb into `uint16` memmaps.
- Checkpoint upload, exact-resume artifacts, inference, and basic evaluation.
- A corrected causal-language-model label path. Earlier experiments affected by
  the double-shift bug are historical evidence, not quality baselines.

### What prevents an expensive run now

- The configured corpus is only 2B unique tokens, consumed for about two epochs.
- The tokenizer was trained on an older C4/Wikipedia mixture rather than the
  current or future data mixture.
- The historical flat token memmap does not preserve document boundaries or
  source identity; new probes now use indexed, isolated documents.
- FineWeb currently receives no source-specific quality treatment.
- The correctness suite and probe safety gates now exist, but they have not yet
  been exercised on the final paid-probe corpus and provider.
- Evaluation is too narrow to support model-selection claims.
- There are no calibrated 350M or 1B configurations and no measured
  loss-versus-compute scaling curve.
- The useful historical checkpoint is an exported artifact, while current local
  training data and full-resume state need to be restored or regenerated.

These are all cheaper to fix at 25M–150M scale than during a 1B run.

## What current research changes

### 1. Small models are architecture-sensitive

[MobileLLM](https://arxiv.org/abs/2402.14905) found that deep-and-thin
architectures, embedding sharing, and grouped-query attention materially improve
125M and 350M models. This repo already ties embeddings, but its current model is
comparatively shallow and uses full multi-head attention. Depth is the ablation
expected to actually move loss and should be tested before scaling. GQA mostly
buys inference KV-cache savings rather than training quality — adopt it by
default for the release sizes and expect a null result on loss at probe scale.

[MobileLLM-Pro](https://arxiv.org/abs/2511.06719) extends that direction to 1B,
using 30 layers at hidden size 1280 plus data-mixture optimization,
distillation/merging, and quantization-aware post-training. Its advanced recipe
is a useful research target, not a safe first baseline.

### 2. Data quality is a first-order model feature

[FineWeb](https://arxiv.org/abs/2406.17557) and
[DataComp-LM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/19e4ea30dded58259665db375885e412-Abstract-Datasets_and_Benchmarks_Track.html)
show large gains from curation and model-based quality selection at fixed
compute. [SmolLM](https://huggingface.co/blog/smollm) combined educational web,
synthetic textbooks, and educational code; its code-data ablation reported much
faster convergence from curated educational Python than from unfiltered code.

The lesson is not “use one quality classifier.” A
[2026 audit of FineWeb-Edu-style classifiers](https://arxiv.org/abs/2605.23721)
shows sensitivity to superficial formatting. The practical consequence for this
budget: do not build a bespoke filtering stack. Lean on corpora that already
ship filtering and deduplication (FineWeb-Edu, DCLM-baseline, Dolma subsets)
and spend effort where this project can actually differentiate — mixture
composition, document-aware packing, decontamination, and blind human sample
review of what the upstream filters accepted and rejected.

### 3. Small proxy runs can protect the large run

[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
shows that controlled small-scale runs can forecast loss and downstream
performance of substantially larger runs. The project should fit its own curves
rather than copy a universal token count.

[Beyond Chinchilla-Optimal](https://proceedings.mlr.press/v235/sardana24a.html)
also supports training smaller models longer when inference cost matters. That
makes a strong 350M model a useful outcome, not merely a discarded prototype.

### 4. Sequence construction matters

[Analysing the Impact of Sequence Composition on Language Model
Pre-Training](https://arxiv.org/abs/2402.13991) reports gains from
intra-document causal masking and better sequence construction. The current unindexed flat memmap makes those controls difficult
and should be replaced before the main corpus is generated.

### 5. Modern 1B baselines use far more data than this budget permits

- [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) reports trillions
  of pretraining tokens for its small models.
- [OLMo 2 1B](https://allenai.org/blog/olmo2) used a multi-stage, multi-trillion
  token recipe with explicit stability work and high-quality annealing data.
- [Qwen3](https://qwenlm.github.io/blog/qwen3/) describes a family-scale corpus
  in the tens of trillions of tokens.
- [SmolLM3](https://huggingface.co/blog/smollm3) used 384 H100s for 24 days for
  its 3B run.

The available credits cannot reproduce those generalist recipes. A 1B model
trained on roughly 100B–200B well-selected tokens can still be a credible
research and engineering result, especially with a focused post-trained use
case, but it should not be marketed as frontier general-purpose quality.

## Definition of success

The project needs two kinds of proof.

### Research/engineering proof

- Reproducible data, tokenizer, model, and training artifacts.
- Exact checkpoint resume and HF export equivalence.
- Stable loss curves with no unexplained spikes or silent sample loss.
- Transparent comparison against public models in the same evaluation harness.
- A documented scaling prediction made before the 1B run and compared with the
  eventual result.

### Workable-model proof

The base model should produce coherent continuations, but “workable” should be
demonstrated by one narrow post-trained model, not judged only from cherry-picked
base-model prompts.

Choose one task whose success can be measured automatically, such as:

- structured extraction to a strict JSON schema;
- short-form summarization with factuality checks;
- constrained tool selection and argument generation;
- a narrow code or educational assistant tied to the selected data emphasis.

Before post-training, define a held-out test set and target. A reasonable initial
bar is at least 90% schema-valid output plus a task-specific quality metric that
beats a prompt-only public small-model baseline. The exact task should be chosen
after the 150M data experiments reveal the model's strongest capability.

## The model ladder

Token counts below are planning ranges, not commitments. Every stage may stop
early when its decision is already clear.

| Stage | Target size | Training tokens | Purpose | Deliverable |
|---|---:|---:|---|---|
| Correctness model | 10M–25M | 20M–100M | Find pipeline and loss bugs | Passing tests and deterministic tiny run |
| Data probes | 25M–40M | 250M–500M each | Compare corpus, filtering, tokenizer, packing | Ranked data recipes with confidence intervals |
| Architecture probes | 60M–100M | 1B–3B each | Depth, GQA, QK norm, schedule | Frozen architecture baseline |
| Scaling anchor | ~150M | 5B–15B | Calibrate loss and downstream scaling | Predicted 350M and 1B curves |
| Proof release | ~350M | 30B initially; extend toward 70B–100B | Produce a useful public artifact | Base model, model card, evals, post-trained variant |
| Main run | ~1B | 50B checkpoint; extend toward 100B–200B | Deliver the 1B-class model | Base model, post-trained model, full report |

The 350M and 1B runs are deliberately checkpointed in phases. Continue only
while held-out loss, capability metrics, and the marginal value of more tokens
justify the spend.

## Architecture program

Start with a conservative modern baseline:

- decoder-only Transformer;
- RMSNorm, RoPE, SwiGLU, and tied input/output embeddings;
- grouped-query attention;
- QK normalization if the small ablation confirms stability;
- 4k native context for pretraining, with long-context extension deferred;
- BF16 training with Flash Attention/SDPA where supported;
- a deep-and-thin shape chosen from measured throughput and loss, not parameter
  count alone;
- a warmup-stable-decay (WSD) schedule as the default, pending Gate 2
  confirmation, because it enables trunk-and-branch training (below).

### Trunk-and-branch training

The 350M and 1B runs should be structured as one long stable-LR **trunk** with
short decay **branches** annealed off checkpoints at the planned decision
points (50B, 100B, 150B+ tokens). Each branch is a complete, releasable model;
the trunk keeps its options open. This converts the phased stop/continue
decisions from "commit to a token count up front" into "branch when the
evidence justifies it," at the cost of one short decay phase per decision
point. The annealing branches are also where the quality-focused mixture
(more code, math, reference, educational text) is injected, following the
OLMo 2 / SmolLM3 playbook. Rehearse the branch mechanics at 150M before
relying on them for the release runs.

The 1B candidate should include a MobileLLM-like deep option around 30 layers and
hidden size around 1280, but the project must not blindly copy that shape. Compare
it with a shallower/wider compute-matched model at 60M–150M scale.

Hold these ideas behind experiments rather than stacking them into the baseline:

- z-loss;
- NoPE or hybrid positional layers;
- per-head gates and value residuals;
- checkpoint averaging or EMA;
- knowledge distillation and specialist-model merging;
- 4-bit quantization-aware training.

[IMU-1](https://arxiv.org/abs/2602.02522) is a particularly interesting recent
430M recipe using several of these techniques and 72B tokens. Because it is new
and changes many variables at once, reproduce individual ingredients on small
models before adopting them.

## Data v4

### Candidate source mixture

The first bakeoff should compare at least three compute-matched mixtures rather
than bless one mixture by intuition:

1. **Current control:** the existing C4/Wikipedia/FineWeb mixture.
2. **Education-first:** mostly FineWeb-Edu or a reproducible equivalent, with
   reference text, code, and math.
3. **Diverse quality:** a DCLM/Dolma-style web core with education, code, math,
   and reference subsets scored separately.

A starting candidate for experiments—not the final answer—is:

| Component | Initial share |
|---|---:|
| Filtered educational/general web | 65% |
| Curated code | 12% |
| Math and science | 8% |
| Wikipedia/reference | 10% |
| Licensed synthetic educational text | 5% |

The synthetic slice deserves a real probe arm rather than permanent deferral:
for a budget this size, distillation-flavored synthetic educational text is
plausibly the highest-leverage data lever available, and whether it earns a
larger share is exactly the kind of question the 25M–40M probes exist to
answer.

The last 10%–20% of a long run should test a quality-focused annealing mixture
with more code, math, reference, and educational material. Base pretraining data
and instruction/chat data must remain explicitly labeled; instruction tuning is
a later stage.

All sources require a license and redistribution review before publishing a
corpus or model trained on them.

### Required pipeline changes

- Store sharded tokens plus a document index, source ID, quality score, and
  content hash.
- Prevent cross-document attention, or construct sequences with explicit
  document-aware packing.
- Perform exact and approximate deduplication both within and across sources.
- Create a benchmark decontamination report before training.
- Track rejection counts and distributions by source and filter.
- Export random accepted/rejected samples for manual blind review.
- Evaluate validation loss separately for web, reference, code, math, and
  synthetic data.
- Make corpus recipes immutable manifests with hashes.

### Tokenizer bakeoff

Retrain the tokenizer on a representative sample of the selected final mixture.
One additional candidate must be evaluated before the choice is frozen:
**adopting the vocabulary of a popular open-model family** (e.g. Llama or
Qwen) instead of a custom one. A shared vocabulary makes the released models
directly usable as speculative-decoding draft models for that family — one of
the most honest real-world uses of a small model — at the cost of a larger
embedding table at every model size. This trade-off must be decided now
because it cannot be revisited after the corpus is tokenized.

Compare at least 24k and 32k vocabularies (plus the compatibility candidate)
on:

- bytes or characters per token by source;
- code and math fragmentation;
- rare Unicode and malformed-text behavior;
- embedding parameter cost at every model size;
- round-trip correctness and special-token handling.

Tokenizer choice must be made before generating the expensive corpus. A changed
tokenizer invalidates all pre-tokenized data and model checkpoints.

There is a circular dependency here: the tokenizer should be trained on the
final mixture, but the data probes that pick the mixture need a tokenizer.
Resolution: all Gate 1 mixture probes run on the current 32k tokenizer as a
held-constant control variable. Once the mixture winner is chosen, retrain the
tokenizer candidates on that mixture, run the bakeoff, and accept exactly one
full re-tokenization pass before generating the main corpus.

## Evaluation

Evaluation must be useful for choosing a run, not just decorating a model card.

### During training

- held-out cross-entropy and perplexity by source;
- token throughput, model FLOPs utilization, memory, and dollars per billion
  tokens;
- gradient norm, activation/weight statistics, skipped updates, and loss spikes;
- fixed generative probes, versioned and never used as training examples.

### Model selection

Use a version-pinned `lm-evaluation-harness` rather than custom scoring code —
it provides the same-harness-as-public-baselines property for free and is more
credible externally. Use full test sets where licenses permit. At minimum
include language-modeling and common-sense suites such as
LAMBADA, HellaSwag, PIQA, ARC, OpenBookQA, WinoGrande, and BoolQ. Add GSM8K and
code evaluation only when the data mixture makes those capabilities plausible.

Compare against public models at matching scale—such as GPT-2, Pythia, and
SmolLM-family checkpoints—under the exact same prompt templates and scoring
code. Report uncertainty and contamination status. A 200-example quick run is a
smoke test, never the headline result.

For the post-trained model, freeze the product-specific test set before tuning
and compare base, SFT, and preference-tuned variants.

Conversational quality is now the first post-training product target. Its
evaluation has three layers: deterministic instruction/format checks, frozen
single- and multi-turn transcripts judged pairwise, and reproducibly blinded
human review. `evals/conversation/v1.jsonl` is the initial project-specific
suite and is permanently evaluation-only. Fireworks judgments must be
position-swapped, calibrated against human labels, and named as internal judge
variants rather than presented as official AlpacaEval or MT-Bench scores.

The initial recipe is full-parameter assistant-only SFT followed, only if SFT
earns it, by a small DPO or APO comparison. Subjective conversational RFT/GRPO
is deferred until its evaluator resists verbosity, sycophancy, position, and
format-only reward hacking. See `docs/conversation-posttraining.md`.

## Stage gates

### Gate 0 — correctness

No paid training until all of these pass:

- one batch can be intentionally overfit;
- labels are shifted exactly once;
- padding and document masks affect neither forbidden nor valid tokens;
- gradient accumulation matches the equivalent large batch;
- save/resume produces the same next updates within numerical tolerance;
- exported HF logits match the training model;
- tokenizer and data fingerprints fail fast on mismatch;
- a tiny run is deterministic enough to detect regressions.

### Gate 1 — data and tokenizer

Run repeated 25M–40M probes with identical model, optimizer, token count, and
validation sets. Requirements for a valid comparison:

- at least 3 seeds per arm; report mean and spread, and advance only when the
  winning margin exceeds the seed spread;
- held-out loss by source is the primary decision metric at this scale —
  standard downstream suites are near-chance for sub-100M models on sub-1B
  tokens and must not be allowed to decide a winner through noise;
- mixture rankings can flip with scale, so before freezing the mixture,
  confirm the ranking holds in at least one 60M–100M replication.

Advance only if a candidate beats the current control in held-out loss across
sources, without a major regression in any intended source domain.

### Gate 2 — architecture and training recipe

At 60M–100M, isolate one variable per comparison:

- current shape versus deep/thin (the ablation expected to move loss);
- full attention versus GQA (expected null on loss; confirms GQA is safe to
  adopt for its inference benefits);
- with/without QK norm;
- tokenizer winner;
- cosine versus a warmup-stable-decay-style schedule (WSD is the presumptive
  winner because it enables trunk-and-branch training; cosine must beat it on
  loss to displace it);
- AdamW versus Muon/NorMuon — promoted from the deferred list because
  small-scale speedrun results and larger-scale reports repeatedly show
  materially fewer tokens to a target loss; if it replicates here it is
  equivalent to a large credit increase, so it earns a dedicated arm;
- packed document masks versus the current sequence construction.

Advance with the smallest recipe that repeatedly wins. Novel components must
earn their complexity.

### Gate 3 — scaling forecast

Train at least three sizes on the frozen recipe and fit validation loss against
parameters, tokens, and measured compute. Publish before-the-fact predictions
for the 350M and 1B checkpoints, including error bands and estimated cost.

### Gate 4 — 350M proof release

The 350M model must:

- complete the planned first phase without unresolved instability;
- beat the existing 100M model and public size-matched controls on the
  predeclared aggregate;
- produce a complete artifact set and independently loadable HF checkpoint;
- meet the chosen “workable model” target after post-training;
- leave enough credits for a failed 1B attempt and restart.

If it fails, diagnose and repeat at 150M/350M. Do not compensate by immediately
scaling to 1B.

### Gate 5 — 1B authorization

Start the 1B run only when:

- the scaling fit predicts a worthwhile improvement;
- a full-throughput benchmark has converted tokens into dollars and wall time;
- cloud GPU quota/capacity is confirmed;
- data shards and artifact uploads have passed an end-to-end rehearsal;
- the budget includes checkpoints, evaluation, post-training, and at least 25%
  recovery reserve;
- the exact stop/continue criteria at 50B, 100B, and later tokens are written
  down.

## Compute policy

The roughly $20k of AWS and Azure credits should be treated as two constrained
provider balances, not as cash and not as one interchangeable pool. Eligibility,
expiry, GPU quota, regional capacity, and which purchasing modes credits cover
must be confirmed before planning around either provider. Do not redeem anything
until the user is ready.

Allocate the eventually usable compute approximately as follows:

| Budget envelope | Share |
|---|---:|
| Correctness, throughput, and cloud rehearsals | 5% |
| Data/tokenizer experiments | 10% |
| Architecture and schedule experiments | 15% |
| 350M base and post-training release | 25% |
| 1B phased run | 35% |
| Failure recovery and final evaluation | 10% |

The 35% line is a **reserve envelope**, not an estimate that a 1B model has a
fixed $7,000 price. At a $20,000 total credit pool it reserves $7,000 for the
main run, with another $2,000 held separately for recovery and final evaluation.

For a rough H100 cross-check, a 1B model requires approximately `6 × 1B ×
tokens` training FLOPs. Assuming the run actually sustains 200–300 TFLOP/s per
H100 and using the current
[AWS Capacity Block price](https://aws.amazon.com/ec2/capacityblocks/pricing/)
of $5.191 per H100-hour in US regions:

| Cumulative tokens | Raw H100 cost | Planning cap with 25% overhead |
|---:|---:|---:|
| 50B | $1,440–$2,160 | $1,800–$2,700 |
| 100B | $2,880–$4,330 | $3,600–$5,400 |
| 200B | $5,770–$8,650 | $7,200–$10,800 |

These are estimates, not quotations. A 1B model may underutilize an H100, while
multi-GPU communication, checkpointing, evaluation, and idle reservation time
raise the bill. Conversely, the currently listed AWS A100 Capacity Block rate
can be cheaper per useful training FLOP if the implementation runs efficiently.
Credits may also have purchasing-mode restrictions. Measure both an H100 and an
A100 run before choosing.

Therefore the default authorization should be:

- approve up to $2,700 to reach 50B tokens;
- approve up to $5,400 cumulative to reach 100B;
- extend beyond 100B only if validation improvement justifies using recovery
  reserve or if measured efficiency makes 200B fit within the $7,000 envelope.

### Spot-first purchasing

Default to spot/preemptible instances on both providers for every stage,
including the release runs. Spot pricing for A100/H100 class hardware
typically runs 50%–70% below on-demand, which is the single largest available
multiplier on the credit pool. The costs of spot — preemption-tolerant
training, frequent checkpointing, automatic resume, and instance-hunting —
are implementation work, which this project treats as free. Requirements:

- checkpoint cadence sized so a preemption loses minutes, not hours;
- automatic resume verified end-to-end (Gate 0 already requires exact resume);
- fall back to on-demand or capacity blocks only when measured preemption
  rates make spot slower or costlier in practice, or when credits exclude
  spot purchasing — confirm coverage during the step-0 credit review.

Rules:

- Spend no more than 10% of total credits before Gate 2.
- Use local hardware or a single modest GPU for correctness work.
- Use a single fast GPU for probes when it is cheaper than distributed
  overhead.
- Reserve multi-GPU nodes for a benchmarked 350M/1B recipe.
- Do not estimate the final run from vendor TFLOPS. Benchmark the exact model,
  context length, precision, attention implementation, and checkpoint cadence.
- Record effective tokens/sec and total dollars per billion tokens.
- Include data storage, egress, idle setup time, failed jobs, and evaluation in
  the budget.
- Never launch an expensive instance before verifying quota, capacity,
  checkpoint destination, automatic shutdown, and a tiny end-to-end job.

The basic compute cross-check is approximately `6 × parameters × training
tokens` floating-point operations, but the purchasing decision must use measured
end-to-end throughput.

## Repo workstreams

### 1. Correctness and reproducibility

- Implemented: tests for label shifting, batching, accumulation, deterministic
  tiny training, one-batch overfit, HF save/reload, and exact interrupted
  resume.
- Implemented: packed-document isolation test, boundary-loss masking, indexed
  corpus fingerprints in run artifacts, and an indexed trainer CLI rehearsal.
- Implemented: checkpoint compatibility fingerprints reject resume when the
  model, training recipe, tokenizer, corpora, or mixture weights differ.
- Implemented: end-to-end indexed preparation, interruption, and resume tests.
- Save the git revision and environment lock in every artifact manifest.
- Make historical broken-label runs impossible to select as baselines.

### 2. Configurable model families

- Separate model implementation from `scripts/train.py`.
- Implemented: named 25M, 60M, 150M, 350M, and 1B configs plus a meta-device
  parameter-count and shape validator.
- Log exact non-embedding and total parameter counts, FLOPs estimates, and
  attention configuration.

### 3. Data format and lineage

- Implemented foundation: immutable, integrity-checked token shards with a
  document index, source IDs, content hashes, quality scores, and metadata.
- Implemented: deterministic source-mixture sampling, padding-free
  document-aware packing, trainer/evaluator integration, and exact-resume-safe
  RNG handling.
- Implemented: data preparation emits transactional indexed train/validation
  corpora, preserves source/chunk provenance, and uses disjoint content-hash
  split assignment.
- Implemented: resumable indexed staging reuses committed shards byte-for-byte
  and verifies the replayed document/token prefix before append.
- Implemented: reviewed Tier-A writing enters only through a hash-verified
  canonical manifest with explicit human keep decisions.
- Remaining: build and sign off the first real probe corpus manifest.

### 4. Evaluation harness

- Implemented: held-out indexed loss and perplexity are reported by source.
- Version tasks, prompts, few-shot settings, and dependencies.
- Evaluate public baselines locally through the same path.
- Produce machine-readable results and a generated model-card table.

### 5. Experiment registry and scaling analysis

- Implemented: immutable multi-seed experiment groups record exact
  configurations, hashes, code state, commands, and append-only events.
- Implemented: the paid-probe preflight validates full corpus hashes, split
  disjointness, tokenizer/recipe identity, source evaluation, matched
  throughput, estimated cost under an explicit cap, checkpoint upload, and the
  experiment plan without starting training.
- Add a script or notebook that fits and plots scaling curves and predicts the
  next stage.
- Predeclare each ablation's decision metric.

### 6. Provider-neutral cloud runner

- Keep secrets outside the repo.
- Support resume from remote object storage.
- Implemented locally: atomic run state/heartbeat, append-only metrics and
  lifecycle events, stale-run inspection, checkpoint visibility, runtime budget
  alarms, a safe per-seed cost stop, and command replay protection.
- Remaining at provisioning time: provider-native billing alarms, an external
  stale-heartbeat watchdog, and instance shutdown on success or failure.
- Make the same command work on AWS or Azure after provider-specific
  provisioning.

## Immediate next steps

Implementation work is not the scarce resource — compute, credits, and
training wall-clock are. These steps are ordered by dependency, not by a
schedule; do them as fast as they can be built and verified.

0. Confirm AWS and Azure credit expiry, eligibility, GPU quota, and which
   purchasing modes credits cover. This is external, blocking information that
   reshapes every downstream decision.
1. Finish manual inspection in the generated writing review pack; blank
   decisions remain excluded.
2. Build the canonical writing manifest, then prepare the resumable indexed
   train/validation probe corpus.
3. Run the tiny correctness rehearsal and a short throughput benchmark on the
   exact intended GPU and exact 25M workload.
4. Fill the quoted hourly rate and explicit dollar cap, enable a verified
   checkpoint destination, and create the immutable three-seed group.
5. Run `uv run probe-preflight`; launch only when it reports `READY=true`.
6. Run the first repeated data probes (at least 3 seeds per arm, current 32k
   tokenizer held constant) locally or on the cheapest suitable GPU.
7. Add pinned `lm-evaluation-harness` evaluations for the existing model and
   public 100M–400M baselines.
8. Implemented foundation: versioned chat formatting, assistant-only indexed
   supervision, fingerprinted base-checkpoint initialization, a frozen
   multi-turn suite, deterministic checks, position-swapped Fireworks judging,
   and blinded human-review packs. Before post-training, build and inspect the
   first real chat corpus and run the public/synthetic/mixed SFT bakeoff.
9. After the mixture winner is chosen: train 24k and 32k tokenizer candidates
   on it and publish the fertility analysis.
10. Write the first experiment report, including losses, downstream scores,
   throughput, estimated cost, failures, and the next predeclared decision.

The only real gate on pace is training time and credit spend. Nothing here
spends meaningful compute on a 1B model.

## First release sequence

1. `gpt-probe` report: data and tokenizer bakeoff with reproducible manifests.
2. `gpt-150m-base`: scaling anchor and complete model card.
3. `gpt-350m-base`: strong small-model proof.
4. `gpt-350m-<specialty>`: measurable workable-model demonstration.
5. `gpt-1b-base`: phased main run with a published prediction-versus-result
   analysis.
6. `gpt-1b-<specialty>`: useful post-trained release.

Names are placeholders. Publishing the method, failure analysis, and scaling
forecast matters as much as the final checkpoint.

## Explicit non-goals for the first cycle

- Competing with trillion-token frontier 1B generalists on every benchmark.
- Training a 1B model merely because credits are available.
- 128k context during base pretraining.
- MoE, 1-bit/ternary weights, speculative decoding, or custom kernels before the
  dense baseline works.
- Mixing several novel architectural ideas into one uninterpretable run.
- Treating synthetic data, benchmark scores, or a single quality classifier as
  ground truth.
- Calling a base checkpoint a chat model.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-07-27 | Use a staged 25M → 60M → 150M → 350M → 1B ladder | Smaller controlled runs de-risk data, architecture, scaling, and cost |
| 2026-07-27 | Make 350M a real release target | It provides independent proof and remains useful if 1B is delayed |
| 2026-07-27 | Defer the 1B launch until five gates pass | Current data, tokenizer, tests, evaluation, and cost calibration are insufficient |
| 2026-07-27 | Target focused usefulness rather than frontier generality | Available credits are far below the compute used by current leading 1B models |
| 2026-07-27 | Treat implementation time as free; plan around compute and credits only | Agentic tooling makes dev work cheap; schedules gate on training time and spend, not build time |
| 2026-07-27 | Use prefiltered corpora (FineWeb-Edu, DCLM-baseline, Dolma) instead of a bespoke filtering stack | Filtering quality is already solved upstream; this project differentiates on mixture, packing, and decontamination |
| 2026-07-27 | Probes: ≥3 seeds per arm, loss-by-source primary, ranking confirmed at 60M–100M before freezing | Downstream suites are near-chance at probe scale and mixture rankings can flip with scale |
| 2026-07-27 | Probe mixtures on the current 32k tokenizer; retrain tokenizer once on the winning mixture | Resolves the tokenizer/mixture circular dependency with exactly one re-tokenization pass |
| 2026-07-27 | Default to spot/preemptible instances for all stages | 50%–70% savings; preemption-tolerance is implementation work, which is treated as free |
| 2026-07-27 | Structure 350M and 1B runs as WSD trunk-and-branch | Branch annealing turns token-count commitments into evidence-driven decisions and gives a releasable model per branch |
| 2026-07-27 | Promote Muon/NorMuon from deferred list to a dedicated Gate 2 arm | Replicated token-efficiency gains would be equivalent to a large credit increase |
| 2026-07-27 | Evaluate a draft-model-compatible vocabulary (Llama/Qwen family) in the tokenizer bakeoff | Shared vocab enables speculative-decoding use; irreversible once the corpus is tokenized |
| 2026-07-30 | First verified GPU probe complete: 25M × 250M tokens, loss 3.73, wiki ppl 40.9, ~$2.50 (AWS L4); Gate 0 green on AWS and Kaggle from the same commit | Registry anchor point #1; bootstrap-verified cross-cloud reproducibility |
| 2026-07-30 | Gate 1 concluded design: B0.1 baseline (FinePDFs 20 / FineWeb-Edu 28 / DCLM 15 / diversified conversation 12 / FineMath 8 / stackv2-edu code 9 / narrative 3 / reference 5 with Wikipedia de-monopolized) × 3 seeds + six bounded perturbations × 1 seed + 2 winner-confirm seeds (~11 runs, ~$22); C4 retired; conversation is a product requirement priced by P2, not a hypothesis | Five design corrections from the user: strong reference, breadth over replication, requirement≠hypothesis, conservative perturbations, modern sources; literature check (SmolLM2/OLMo 2) added the STEM trim and narrative slice, and flagged staged mixtures for the 350M/1B (probe mix must not be copied verbatim to big runs) |
| 2026-07-31 | Compare Gate 1 arms on equal-token per-source loss (`checks.source_eval`), never on training or mixture-weighted validation loss | `validation_source_weights: null` makes train.py inherit `source_weights`, so each arm was validated on its own sampling distribution — an arm trained on more predictable text scores lower without being a better model. On the corrected metric sigma falls from 0.047 to 0.008 (2.3M eval tokens vs a narrow weighted sample) and P1 moves from "within noise" to 1.86σ worse. Arm configs now set validation weights explicitly and uniformly |
| 2026-07-31 | Gate 1 result: conversation is free to double; narrowing the distribution is not free. P2 (conversation 12→24%) is within noise overall (−0.0043, 0.54σ) while improving every dialogue source (Hansard −0.106, IRC −0.088, StackExchange −0.073); P1 (edu filter 28→43, DCLM 15→5) is 1.86σ worse and P3 (STEM 17→27) is 2.24σ worse | Both losing arms concentrated the mixture and paid for it outside the domain they bought; P3 improved `code` by 0.108 and worsened everything else. The conversation dose is now priced rather than assumed, and the cost lands on math and textbook prose — registers deprioritised for a conversational model. Likely composite: B0.1 with conversation at 24%, paid proportionally |
| 2026-07-30 | Post-training v2: terminate assistant turns with real EOS (id 2) instead of the `<\|end\|>` string; move the primary lever to mid-training (anneal-stage instruction/conversation data) ahead of SFT; replace Dolly-6k with a SmolTalk2/TuluTalk-style mixture including everyday-conversations; DPO only against a predeclared bar | The tokenizer has no reserved slots, so `<\|end\|>` is five ordinary sub-pieces whose merges shift with context — termination was a spelling task the model kept failing, and the regex stop only hid it. The "irrelevant reply" failure was register mismatch (Dolly is short-form closed QA, tested conversationally), not a training bug; and at 25M no SFT data could have fixed it, since capability is installed during pretraining |
| 2026-07-30 | Artifact durability is a precondition for any retrain: training refuses to start on an output_dir holding a completed run, every model is mirrored to a private HF repo, and cloud boxes ship to S3 every 10 min over presigned PUT with a spot-interruption watcher | Nothing protected a finished model — final/ was overwritten and old checkpoints pruned on any re-run — and every artifact produced so far existed on exactly one laptop. Presigned PUT gives boxes durability without a credential ever being written into user-data |
| 2026-07-30 | Run every Gate 1 arm off one union corpus, sized to each source's peak demand across arms +10%, with arms expressed only as `data.indexed.source_weights` | Per-arm corpora would confound mixture effects with corpus-construction variance — different documents, different dedup outcomes, different filter draws. One corpus makes the sampling mixture the only variable, cuts prep from ~1.5B tokens to 460M, and makes every future arm (P6, composites, winner-confirm) cost zero prep |
| 2026-07-30 | Execute waves in parallel: all arms of a wave launch simultaneously on their own boxes; boxes never self-terminate on training completion | Cloud capacity is elastic, so serial execution buys nothing but wall-clock; and a box that dies when training ends takes its results with it, which is how the first bench run was lost |
| 2026-07-30 | PENDING (decide after Gate 3 fit): reinterpret the "1B-class" main run as the largest compute-optimal N within budget, capped at laptop-runnable (~6B 4-bit) — likely 2-3B at ~20 tok/param instead of 1B heavily over-trained | User is inference-cost-insensitive in the 1-6B range, so over-training's rationale does not apply to the main run; same compute buys strictly lower loss nearer Chinchilla-optimal. 350M proof release keeps its over-training logic (per-size public comparison) |
| 2026-07-29 | Make natural conversation the first post-training target | The user values conversational quality; it can be tested through deterministic, judge-based, and human pairwise evaluation |
| 2026-07-29 | Use assistant-only SFT before preference optimization | SFT is the primary lever at 350M–1B; DPO/APO must demonstrate marginal value and subjective RFT is too easy to game initially |

## Living-plan rule

Update this document after every gate. Replace planning ranges with measured
numbers, append rejected hypotheses, and preserve forecasts made before larger
runs. The plan has failed if it becomes a retrospective story that always makes
the latest result look intentional.
