# Deep Learning Research: Functional vs Non-Functional Requirements

This directory documents neural-network and transformer experiments for **binary software-requirements classification**: Functional Requirements (**FR**, label `0`) and Non-Functional Requirements (**NFR**, label `1`). It follows the [dataset exploration](../dataset_exploration/) and [classical machine-learning baselines](../research_modeling/). The research uses a fixed dataset and evaluation protocol rather than combining all original source records indiscriminately.

## Dataset and experimental protocol

The starting collection is a **combined multi-source software-requirements dataset** containing 8,854 records from source identifiers D1–D5. Following source, label, duplication, and near-duplicate audits, the research used the validated **D2 v1.1 subset** (4,833 records), not the entire combined collection.

| Frozen D2 v1.1 partition | FR | NFR | Total | Role |
|---|---:|---:|---:|---|
| Train | 2,224 | 1,154 | 3,378 | Parameter fitting |
| Validation | 489 | 239 | 728 | Model comparison, checkpoint selection, diagnostics |
| Held-out test | 488 | 239 | 727 | Final evaluation only, after selection was frozen |
| **Total** | **3,201** | **1,632** | **4,833** | |

The partition assignment and similarity groups are recorded in [`../dataset_exploration/outputs/validated/54_d2_v11_frozen_split_manifest.csv`](../dataset_exploration/outputs/validated/54_d2_v11_frozen_split_manifest.csv), with metadata in [`54_d2_v11_split_metadata.json`](../dataset_exploration/outputs/validated/54_d2_v11_split_metadata.json). Related requirements were grouped to reduce train/validation/test leakage; **independence at the original-project level was not established**. Model selection and the experiments documented **here** relied on validation results; the held-out test was assessed separately in [`../final_model/`](../final_model/).

## Experimental methodology

| Scripts | Stage | What was evaluated |
|---|---|---|
| `00`–`02` | Preflight | Environment, frozen-data integrity, tokenization and length checks |
| `03`–`04` | Neural baselines | TextCNN and BiLSTM, each with standard and class-weighted variants |
| `05`–`07` | Transformer baselines | Fine-tuned DistilBERT, BERT-base, and RoBERTa-base |
| `08` | Paired comparison | Same-validation-example predictions, paired bootstrap and McNemar analyses |
| `09`–`10` | Multi-seed robustness | Transformer runs with seeds **42, 17, and 29**; F1 variability and prediction stability |
| `11`–`16` | Error and annotation review | Persistent errors, blind review preparation, AI-assisted suggestions, agreement and disagreement audits |
| `17`–`18` | Consolidation and freeze | Experiment inventory, reproducibility records, selection rationale, and frozen final checkpoint |

All reported scores below are from the **validation partition**, not the final held-out test. The positive class of primary interest is NFR, so NFR precision, recall, and F1 are reported alongside macro-F1. Transformer comparisons were performed across the same frozen validation examples.

## Main validation results

| Architecture | Seed-42 NFR F1 | Three-seed mean NFR F1 | Three-seed SD |
|---|---:|---:|---:|
| **DistilBERT** | **0.8595** | **0.8634** | **0.0036** |
| BERT-base | 0.8577 | 0.8590 | 0.0096 |
| RoBERTa-base | 0.8534 | 0.8557 | 0.0050 |

Additional seed-42 neural baselines: standard **TextCNN NFR F1 = 0.8058** and standard **BiLSTM NFR F1 = 0.7889**. Weighted variants and other metrics are recorded in the corresponding experiment reports.

DistilBERT had the strongest **mean validation NFR F1** of the three transformer families, but the observed differences were small and did **not** establish statistically significant overall superiority. The **original seed-42 DistilBERT checkpoint**, a **0.50 classification threshold**, and a **maximum sequence length of 96** were frozen for final use; the best-performing individual seed was not selected after inspecting all runs. See [`18_final_model_selection_report.txt`](outputs/reports/18_final_model_selection_report.txt) and [`18_final_model_freeze.json`](outputs/reports/18_final_model_freeze.json).

## Error analysis and annotation cautions

The study inspected cases repeatedly misclassified across runs and prepared a 43-item priority review. AI-assisted annotations were compared with original labels to flag potential ambiguity and inconsistent labeling. **These suggestions are not independently verified ground truth**; disagreement does not demonstrate that the source label is incorrect. Independent human adjudication remains a limitation. See [`11_persistent_error_report.txt`](outputs/reports/11_persistent_error_report.txt), [`15_annotation_comparison_report.txt`](outputs/reports/15_annotation_comparison_report.txt), and [`16_disagreement_audit_report.txt`](outputs/reports/16_disagreement_audit_report.txt).

## Directory guide

```text
deep_learning_research/
├── scripts/        # Scripts 00–18: training, comparisons, reviews, freeze
├── outputs/
│   ├── tables/     # Epoch histories, metrics, comparisons, audit tables
│   ├── reports/    # Experiment reports, metadata, frozen selection record
│   └── predictions/ # Per-example validation predictions (often kept local)
├── checkpoints/    # Trained weights and tokenizer/config snapshots (large; local)
├── models/         # Vocabulary JSONs for TextCNN and BiLSTM
├── logs/           # Per-run training logs (local)
├── Docs/           # Comprehensive research report
└── README.md
```

**Useful starting points:** [Full deep learning report](Docs/DEEP_LEARNING_RESEARCH_FULL_REPORT.txt), [multi-seed report](outputs/reports/09_transformer_multiseed_report.txt), [research consolidation](outputs/reports/17_research_consolidation.txt), and [transformer comparison table](outputs/tables/17_transformer_comparison.csv).

## Reproducibility and GitHub scope

The scripts rely on the frozen D2 v1.1 dataset manifest in `../dataset_exploration/`; original datasets may be unavailable in a public clone. This repository can retain scripts, configuration/metadata, summary tables, and reports while excluding **multi-gigabyte checkpoints**, training logs, and per-example prediction files with `.gitignore`. Because some tables contain original requirement text, review dataset redistribution permission before making those files public.

**Checkpoint files are not included in a normal Git clone when ignored.** As a result, reproducing training requires the underlying dataset and software environment; reproducing inference from the frozen model additionally requires its checkpoint. The separately packaged model and single held-out test results are described in [`../final_model/`](../final_model/).

No figures are linked here because the provided directory structure does not contain a `figures/` directory; results are available as versioned tables and reports.
