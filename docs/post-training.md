# Conversational post-training plan

## Goal

Turn the strongest base checkpoint into a genuinely useful conversational
model while preserving the base model's language ability, provenance, exact
resume guarantees, and evaluation credibility.

The first proof target is the 350M model. The 1B post-training run should reuse
an already validated recipe rather than becoming the first full experiment.
Post-training is not expected to add knowledge that the base model never
learned; it should teach the model how to expose its capabilities through
helpful, coherent, context-aware conversation.

## Program shape

The initial program has three checkpoints:

```text
base checkpoint
  -> supervised fine-tuning (SFT)
  -> preference optimization (DPO or APO)
```

Every checkpoint is evaluated independently. Preference optimization advances
only if it improves blinded conversational comparisons over the SFT checkpoint
without a material regression in instruction following or base capabilities.

Do not begin with subjective-reward GRPO or a bespoke reward model. Those add
more ways to exploit an imperfect evaluator than this project currently needs.
RL with verifiable rewards may later be used for narrow tasks with objective
answers, but it is not the first conversational-quality method.

## Freeze evaluation before generating data

Training data generation begins only after the evaluation set is immutable and
content-hash registered. Evaluation prompts, reference answers, rubrics, and
judge templates must never be used as generation seeds or training examples.

Use three complementary evaluation layers.

### Deterministic checks

- IFEval for instruction and constraint following;
- requested format, valid JSON, exact phrase, and requested-length checks;
- role leakage and accidental role-marker generation;
- repetition, empty answers, premature termination, and runaway verbosity;
- the existing base-model benchmark suite to measure capability regression.

### Open conversational benchmarks

- MT-Bench for conventional two-turn assistant behavior;
- AlpacaEval 2 with length-controlled win rate for broad single-turn quality;
- MT-Bench-101 for finer-grained multi-turn behavior;
- a project-owned held-out suite of approximately 200 conversations.

The project-owned suite should cover:

- natural everyday conversation;
- ambiguous requests and appropriate clarification;
- remembering constraints introduced in earlier turns;
- accepting and applying a correction;
- disagreement without becoming combative or falsely deferential;
- matching requested tone and level of detail;
- concise factual explanation;
- uncertainty and calibrated admission of missing information;
- editing, rewriting, brainstorming, and comparison;
- instructions that change legitimately during the conversation;
- resisting irrelevant details from earlier turns.

### Pairwise judging and human calibration

Evaluate base versus SFT and SFT versus preference-tuned outputs using blinded,
position-swapped pairwise comparisons. Use two independently developed open
judge families through Fireworks rather than trusting one teacher or judge.
Record the exact model identifier, provider, prompt, decoding parameters,
response, parse result, latency, token usage, and estimated cost.

At least 100 randomly selected pairs per major comparison should also receive a
manual decision. Report judge-to-human agreement and disagreements by category.
Scores produced with substituted open judges are internal benchmark variants;
do not present them as official GPT-4-judged AlpacaEval or MT-Bench scores.

## Chat representation

Define one versioned chat template before preparing SFT data. It must specify:

- system, user, assistant, tool-call, and tool-result representation;
- beginning, end-of-turn, and end-of-conversation behavior;
- whether control markers are atomic tokenizer pieces;
- generation behavior when the prompt ends with a user turn;
- truncation rules for conversations longer than the context window.

The final tokenizer should contain atomic control tokens unless a tokenizer
bakeoff shows a compelling reason not to. The template and all control-token
IDs become immutable artifact inputs and must be recorded in checkpoints.

Training loss is applied to assistant response tokens and the assistant
end-of-turn marker. System and user turns are context only. Tool results are
also context only unless a specific experiment intentionally trains their
generation. Packing must preserve conversation boundaries and must never train
one conversation to predict the beginning of another.

## Dataset schema and provenance

The canonical source representation is JSONL with one conversation per record:

```json
{
  "id": "stable-record-id",
  "source": "source-id",
  "messages": [
    {"role": "system", "content": "Optional system instruction"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ],
  "metadata": {
    "license": "license identifier",
    "generation": null,
    "quality": {},
    "parent_ids": []
  }
}
```

Synthetic records additionally retain:

- teacher model and exact revision or provider model ID;
- provider and applicable policy snapshot;
- prompt-template name, version, and hash;
- source or seed record IDs;
- temperature, top-p, maximum tokens, and random seed where supported;
- raw candidate IDs and the reason the selected candidate won;
- all automatic filters and human decisions.

Prepared shards retain conversation IDs, source IDs, content hashes, template
hash, tokenizer hash, supervised token spans, and train/validation assignment.
Splits are assigned by content or conversation-family hash so variants of one
conversation cannot cross the split boundary.

## Data arms

Do not bless one mixture by intuition. Compare equal assistant-token budgets:

1. **Inspected public control.** Use only subsets whose source license,
   upstream generation terms, quality, and redistribution status have been
   reviewed.
2. **Open-teacher synthetic.** Generate conversations with a permissively
   licensed open model served through Fireworks.
3. **Mixed.** Combine the best reviewed public material with synthetic data
   targeted at missing conversational behaviors.

The synthetic generator should sample a declared behavior taxonomy rather than
asking for undirected “diverse conversations.” Vary user expertise, tone,
message length, number of turns, ambiguity, emotional intensity, domain, and
whether the best answer should clarify, answer directly, disagree, or admit
uncertainty.

Generate multiple candidates for a subset of prompts. Prefer rejection
sampling with independent scoring and manual inspection over accepting the
first teacher response. Avoid using the same model as generator, sole filter,
sole preference labeler, and sole evaluator.

Initial generation is a small pilot costing no more than $10–$25. Inspect its
accepted and rejected samples before authorizing a larger batch. Fireworks
credits are a data-generation and evaluation budget, not a reason to maximize
the number of synthetic tokens.

## Supervised fine-tuning

Use full-parameter SFT for release candidates. LoRA may be used for cheap recipe
probes, but a LoRA result does not establish the final full-training recipe.

At 350M, compare cumulative supervised budgets around:

- 10M assistant tokens;
- 30M assistant tokens;
- 100M assistant tokens.

These are decision checkpoints, not mandatory consumption targets. Stop when
held-out conversational quality saturates or base capability begins to fall.
Record both total context tokens and supervised assistant tokens; reporting
only total packed tokens hides meaningful differences between mixtures.

SFT correctness gates:

- user and system labels are always ignored;
- assistant labels are shifted exactly once;
- packed conversations cannot attend across boundaries;
- truncation preserves at least one supervised target and the context needed
  to interpret it;
- accumulation equivalence and exact resume pass with masked labels;
- inference uses the identical chat template used for training;
- one tiny conversation set can be deliberately overfit.

## Preference stage

Construct preference pairs only after selecting the strongest SFT checkpoint.
Candidate responses should include generations from the current SFT policy so
the training distribution resembles inference. Teacher answers can be included
as strong candidates, but “larger teacher output always wins” must not be the
entire preference rule.

Each pair records the prompt, complete prior conversation, chosen response,
rejected response, candidate model IDs, judge decisions, position swaps,
automatic checks, and any human decision.

Start with DPO as the reproducible control. Test APO only as a matched
alternative if the implementation and reference behavior are validated.
Predeclare learning rate, beta, token budget, primary pairwise metric, and
regression thresholds. Preference tuning is rejected if improvements are
mostly explained by longer responses, canned headings, excessive agreement, or
judge-specific style.

## Advancement gates

### SFT advances when

- it clearly beats the base model on the frozen project conversation suite;
- IFEval and deterministic conversational checks improve;
- manual reviewers prefer it for the intended behavior;
- base benchmark regressions remain within the predeclared tolerance;
- results reproduce across at least two seeds at 350M.

### Preference tuning advances when

- it clearly beats the selected SFT checkpoint in position-swapped pairwise
  evaluation;
- both open judges agree on the direction of improvement;
- blinded human review confirms the improvement;
- length-controlled and raw scores tell a consistent story;
- safety, instruction following, factuality, diversity, and base capabilities
  do not materially regress.

### The 1B post-training run is authorized when

- the complete 350M recipe has passed both gates;
- the chosen dataset manifests and evaluation set are immutable;
- the exact training workload has been benchmarked;
- full checkpoint recovery has been rehearsed;
- training and Fireworks generation/judging each have explicit budget caps.

## Required repository work

Before the first meaningful conversational run, implement:

- a versioned chat formatter shared by preparation, training, inference, and
  evaluation;
- assistant-only supervised spans in indexed shards;
- immutable chat dataset preparation and split manifests;
- initialization of SFT from a verified base checkpoint while starting a fresh
  optimizer schedule;
- masked-label correctness, packing, truncation, and resume tests;
- chat-mode local inference with multi-turn history;
- frozen deterministic conversation cases and machine-readable results;
- MT-Bench, AlpacaEval 2, MT-Bench-101, and IFEval runners with pinned versions;
- Fireworks generation and pairwise-judge adapters that default to dry-run and
  require explicit budgets;
- blinded human-review exports and judge-agreement reports;
- base/SFT/preference comparison tables for the model card.

None of these commands should generate paid data, call a paid judge, or launch
training as a side effect of setup or validation.

## Failure modes to watch

- teaching fluency while reducing factual competence;
- overtraining generic assistant phrases and unnecessary headings;
- reward or judge hacking through verbosity;
- loss of conversational diversity after preference tuning;
- training on benchmark prompts or near-duplicates;
- silent mixing of incompatible chat templates;
- licensing a dataset wrapper while ignoring the terms of its upstream
  generated outputs;
- reporting one judge's preferences as objective quality;
- calling a base checkpoint a chat model.

## Current decision

The default post-training path is:

1. freeze evaluation;
2. build and manually inspect a small synthetic-data pilot;
3. compare public, synthetic, and mixed SFT arms at 350M;
4. select the SFT checkpoint before creating preference data;
5. compare DPO, and optionally APO, against SFT;
6. promote the proven recipe to 1B without changing the dataset, template,
   evaluation, or optimization method simultaneously.
