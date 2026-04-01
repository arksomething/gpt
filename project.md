# Final Project Proposal

## Title
**Which Modern Language-Model Techniques Still Help at Small Scale?**  
**A controlled comparison of GPT-2-style and LLaMA-style decoder models at roughly 100M parameters**

> Replace the bracketed placeholders below before submitting.

## Team Members and Proposed Roles

| Team member | Proposed role | Planned responsibilities |
|---|---|---|
| [Name 1] | Writing / related work lead | Final write-up, paper summaries, bibliography, integration of experimental results into the report |
| [Name 2] | Engineering / model training lead | Implement GPT-2-style baseline, manage training runs, checkpointing, and reproducibility |
| [Name 3] | Evaluation / analysis lead | Run benchmarks, maintain development and test protocol, perform error analysis, and prepare tables/figures |

One person may fill multiple roles if needed, but we plan to divide the work so that data preparation, model implementation, and evaluation can proceed in parallel.

## Problem Statement

In modern large-language-model research, stronger data curation, modern decoder designs, and more careful evaluation have become standard practice. However, it is not always clear which of these techniques remain useful when the model is much smaller and the compute budget is limited. Many real-world use cases depend on models that are inexpensive to train or run locally, so understanding small-scale behavior is practically important.

Our project asks the following research question:

**When model size and training budget are constrained to roughly 100M parameters, which modern language-model design choices still provide measurable gains over an older GPT-2-style decoder baseline?**

Instead of treating this as a general "train a small language model" project, we propose a controlled comparison. We will hold the dataset, tokenizer, approximate parameter count, and evaluation protocol as constant as possible, then compare a modern LLaMA-style decoder against an older GPT-2-style decoder trained from scratch on the same data. This turns the project into an empirical NLP system study rather than only an engineering exercise.

## Project Type

This is a **System Project**. The main output will be a working experimental pipeline and a comparison of trained models, not just a survey or a resource alone.

## Current Progress and Evidence of Feasibility

We have already begun the project and built much of the infrastructure needed for the final system:

- a tokenizer-training pipeline using SentencePiece
- a data-preparation pipeline that filters and deduplicates web text
- a training pipeline for a roughly 100M-parameter LLaMA-style decoder model
- intrinsic evaluation through validation loss and perplexity
- a benchmark wrapper for downstream language-model evaluation on multiple tasks

This existing code base makes the proposal plausible because the remaining work is focused on controlled comparison rather than starting from zero. The main new engineering task is to implement a matched GPT-2-style baseline and run the comparison fairly.

## Proposed System and Strategy for Solving the Problem

### Core system

The current repository already supports a LLaMA-style decoder-only model trained from scratch. For the final project, we will compare it to a GPT-2-style decoder-only baseline trained under as similar conditions as possible.

Our controlled setup will aim to match:

- training corpus
- tokenizer
- approximate parameter count
- training-token budget
- optimizer budget and evaluation schedule

The comparison is important because otherwise differences in performance could come from data or tokenization rather than architecture.

### Simple version we are confident we can complete

The minimum successful version of the project is:

1. Train one LLaMA-style model at about 100M parameters.
2. Train one GPT-2-style model at about the same scale on the same data.
3. Compare them using intrinsic metrics and a fixed set of downstream benchmark tasks.

This version already answers the main research question at a useful level.

### More elaborate version if time allows

If the simple comparison is successful and we have time left, we will add one or two controlled ablations:

- turn off deduplication or some filtering step to measure the effect of data quality at small scale
- add or remove a single architectural feature, such as RoPE or RMSNorm, to isolate which "modern" component matters most
- compare against a small number of public reference models such as GPT-2, OPT-125M, or SmolLM2, while clearly noting that these are not fully controlled baselines because they were trained on different corpora

### Baselines

Our main baseline will be the **GPT-2-style model trained from scratch under matched conditions**. This is the most scientifically useful baseline because it keeps the training setup constant and changes mainly the model family.

To keep the comparison controlled, we will keep the tokenizer fixed across the main architecture comparison instead of mixing architectural differences with GPT-2's original tokenizer choices.

We will also report external reference points using existing small public models where useful, but these will be secondary baselines because they were not trained on the same corpus, tokenizer, or budget.

## Evaluation Plan

### Output of the system

The system outputs are:

- trained language-model checkpoints
- intrinsic test scores such as loss and perplexity
- downstream benchmark scores on a fixed evaluation suite
- qualitative generations and categorized model errors

### Metrics

We will use multiple scoring metrics rather than one number alone.

**Intrinsic metrics**

- validation loss
- test loss
- perplexity

These metrics measure next-token prediction quality on held-out in-domain text.

**Downstream metrics**

- task accuracy on benchmark tasks run through `lm-eval-harness`
- an unweighted average across tasks to summarize overall performance

Our current benchmark shortlist is:

- HellaSwag
- ARC-Easy
- ARC-Challenge
- PIQA
- WinoGrande
- LAMBADA

We may adjust the exact set slightly for runtime reasons, but we will keep the final set fixed before the final comparison.

### Train / development / test split

To ensure valid results, we will separate data into:

- **training set** for learning model parameters
- **development set** for model selection, early analysis, and any tuning
- **test set** for final intrinsic reporting only after decisions are fixed

At the moment, the repository already has train and validation splits. Before final experiments, we will reserve a separate held-out test split for intrinsic evaluation so that the development split is not also serving as the final test.

For external benchmark tasks, we will also follow a fixed protocol:

- choose the evaluation tasks in advance
- avoid changing the benchmark suite after seeing final results
- avoid tuning architecture or hyperparameters on benchmark test outcomes

### Error analysis

The final project will include explicit error analysis, as required by the assignment.

After the main runs, we will inspect examples where the models fail on development-time analysis and categorize the errors. For example, we will look for:

- commonsense reasoning failures
- distractor sensitivity in multiple-choice tasks
- failures on longer-context dependencies
- repetitive or degenerate generation
- cases where lower perplexity does not lead to better downstream accuracy

This analysis is important because benchmark averages alone will not explain why one small model does better than another.

### Validity and reproducibility

We will try to make the comparison scientifically valid by:

- keeping the corpus and tokenizer fixed across the main architecture comparison
- matching parameter count and training budget as closely as possible
- using the same evaluation scripts and metrics for all runs
- saving checkpoints and logs so results can be reproduced
- reporting any important implementation differences that could affect the interpretation

## Collaboration Plan

We will organize the work so that team members do not block one another.

- The engineering lead will implement and train the GPT-2-style baseline.
- The data/training pipeline can already run independently, so one member can maintain data processing and run management while another works on modeling.
- The evaluation lead can prepare benchmark scripts, scoring tables, and analysis templates before all training runs are finished.
- The writing lead can begin the proposal, related work, and methods sections in parallel with experiments.

This division reduces dependency between members. For example, evaluation code can be prepared against a stable checkpoint interface and sample outputs, while model-training work continues independently.

## Related Work

We discuss six peer-reviewed NLP papers that directly shape our proposed project.

### 1. Kudo and Richardson (2018)

**Taku Kudo and John Richardson. 2018. _SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing_. In Proceedings of EMNLP 2018: System Demonstrations, pages 66-71.**

This paper introduces SentencePiece, a tokenizer that can be trained directly from raw text rather than relying on pre-tokenized word sequences. It is relevant to our project because our repository already uses a SentencePiece tokenizer trained on our own corpus. For the proposed comparison, this paper supports the decision to keep tokenization fixed across model families. If we changed both architecture and tokenizer at the same time, we would no longer know which factor caused a performance difference.

### 2. Lee et al. (2022)

**Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. _Deduplicating Training Data Makes Language Models Better_. In Proceedings of ACL 2022, pages 8424-8445.**

Lee et al. show that language-model training corpora often contain near-duplicate documents and repeated substrings, and that deduplication can reduce memorization while also improving accuracy and reducing train-test overlap. This is directly related to our project because our current data pipeline already includes deduplication. Their results support one of our hypotheses: at small scale, data quality may matter as much as or more than raw data quantity. Their work also motivates a possible ablation in which we compare filtered/deduplicated data against a less curated condition.

### 3. Chang et al. (2024)

**Ernie Chang, Matteo Paltenghi, Yang Li, Pin-Jie Lin, Changsheng Zhao, Patrick Huber, Zechun Liu, Rastislav Rabatin, Yangyang Shi, and Vikas Chandra. 2024. _Scaling Parameter-Constrained Language Models with Quality Data_. In Proceedings of EMNLP 2024: Industry Track, pages 80-97.**

This paper is one of the closest matches to our project question. The authors study models from 25M to 1.5B parameters and argue that "effective training tokens," shaped by text diversity and syntheticity, are especially important for parameter-constrained models. The connection to our project is immediate: our target scale is around 100M parameters, exactly the regime where data quality decisions may dominate performance. We use this paper to justify why our proposal studies matched-size models and treats curation, filtering, and deduplication as core experimental variables rather than implementation details.

### 4. Chen et al. (2025)

**Zhengyu Chen, Siqi Wang, Teng Xiao, Yudong Wang, Shiqi Chen, Xunliang Cai, Junxian He, and Jingang Wang. 2025. _Revisiting Scaling Laws for Language Models: The Role of Data Quality and Training Strategies_. In Proceedings of ACL 2025, pages 23881-23899.**

Chen et al. revisit standard scaling-law assumptions and argue that data density, redundancy, and resource allocation help explain why performance improvements may slow down in practice. This paper relates to our project in two ways. First, it provides theoretical support for studying small-scale regimes instead of assuming the same trends as very large models. Second, it reinforces our decision to control the training budget carefully. If gains plateau or disappear, that may reflect scaling behavior rather than a simple success or failure of a single architecture.

### 5. Pfister, Wunderle, and Hotho (2025)

**Jan Pfister, Julia Wunderle, and Andreas Hotho. 2025. _LLäMmlein: Transparent, Compact and Competitive German-Only Language Models from Scratch_. In Proceedings of ACL 2025, pages 2227-2246.**

LLäMmlein is particularly relevant because it presents compact decoder-only models trained from scratch, with transparent reporting on preprocessing, tokenizer creation, checkpoints, and benchmark evaluation. This is close to the style of experiment we want to run. The paper shows that compact models can still be competitive when the pipeline is well designed and carefully evaluated. It also gives us a concrete example of how to report learning progress over checkpoints instead of discussing only a single final score.

### 6. Lepagnol et al. (2024)

**Pierre Lepagnol, Thomas Gerald, Sahar Ghannay, Christophe Servan, and Sophie Rosset. 2024. _Small Language Models Are Good Too: An Empirical Study of Zero-Shot Classification_. In Proceedings of LREC-COLING 2024, pages 14923-14936.**

Lepagnol et al. compare models from 77M to 40B parameters and find that small models can match or outperform larger models on some classification settings. This paper matters for our project because it supports the general premise that small models are worth studying seriously rather than only as reduced versions of larger systems. Our project differs in that we focus on controlled pretraining and architecture comparison, but their results strengthen the motivation for examining what small models can still do well.

### Optional extension paper: Li et al. (2024)

**Chenglin Li, Qianglong Chen, Liangyue Li, Caiyu Wang, Feng Tao, Yicheng Li, Zulong Chen, and Yin Zhang. 2024. _Mixed Distillation Helps Smaller Language Models Reason Better_. In Findings of EMNLP 2024, pages 1673-1690.**

This paper is not part of our minimum project scope, but it is useful as an extension direction. If time permits, we may explore whether a distilled small model or teacher-assisted reasoning data improves the final benchmark results. We include this paper mainly to show that there are modern techniques beyond architecture alone that may help small models.

## Why This Project Is Plausible

The proposal is feasible because the infrastructure already exists and because the core comparison is narrow enough to complete. We are not trying to reproduce every modern language model. Instead, we are asking a focused question:

**At roughly 100M parameters, does a modern LLaMA-style design outperform a GPT-2-style design when both are trained under the same conditions?**

That question is concrete, measurable, and aligned with the code base we already have.

## Expected Outcome

We expect one of three useful outcomes:

1. The LLaMA-style model clearly outperforms the GPT-2-style baseline, which would support the claim that modern techniques transfer well to small-scale training.
2. The difference is small, which would suggest that some "modern" techniques matter less at this scale than expected.
3. Data quality or evaluation design matters more than architecture, which would itself be a meaningful result for small-model research.

Any of these outcomes would answer the research question and produce a defensible final project.

## References

Chang, Ernie, Matteo Paltenghi, Yang Li, Pin-Jie Lin, Changsheng Zhao, Patrick Huber, Zechun Liu, Rastislav Rabatin, Yangyang Shi, and Vikas Chandra. 2024. *Scaling Parameter-Constrained Language Models with Quality Data*. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 80-97.

Chen, Zhengyu, Siqi Wang, Teng Xiao, Yudong Wang, Shiqi Chen, Xunliang Cai, Junxian He, and Jingang Wang. 2025. *Revisiting Scaling Laws for Language Models: The Role of Data Quality and Training Strategies*. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23881-23899.

Kudo, Taku, and John Richardson. 2018. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66-71.

Lee, Katherine, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. *Deduplicating Training Data Makes Language Models Better*. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8424-8445.

Lepagnol, Pierre, Thomas Gerald, Sahar Ghannay, Christophe Servan, and Sophie Rosset. 2024. *Small Language Models Are Good Too: An Empirical Study of Zero-Shot Classification*. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 14923-14936.

Li, Chenglin, Qianglong Chen, Liangyue Li, Caiyu Wang, Feng Tao, Yicheng Li, Zulong Chen, and Yin Zhang. 2024. *Mixed Distillation Helps Smaller Language Models Reason Better*. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1673-1690.

Pfister, Jan, Julia Wunderle, and Andreas Hotho. 2025. *LLäMmlein: Transparent, Compact and Competitive German-Only Language Models from Scratch*. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2227-2246.
