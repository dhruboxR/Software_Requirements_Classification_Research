# Classical Machine Learning Research

This directory contains the **classical machine learning experiments** for binary **Functional Requirement (FR)** versus **Non-Functional Requirement (NFR)** classification. It is the second modeling stage of the thesis, following [`dataset_exploration/`](../dataset_exploration/). The work establishes interpretable text-classification baselines and investigates class imbalance, feature representations, threshold selection, and validation stability before the deep learning experiments.

## Dataset and evaluation protocol

The original source is a **combined multi-source software requirements dataset** with 8,854 records across five source identifiers (`D1`–`D5`). After source, label, duplication, and similarity audits, the research selected the **D2 source subset** and prepared the frozen **D2 v1.1** binary FR/NFR dataset. **These models were not trained on all 8,854 combined records.**

| Frozen D2 v1.1 split | FR | NFR | Total | Purpose |
|---|---:|---:|---:|---|
| Train | 2,224 | 1,154 | 3,378 | Fit models and run grouped cross-validation |
| Validation | 489 | 239 | 728 | Compare models and examine errors |
| Test | 488 | 239 | 727 | Reserved for the separately selected final model |
| **Total** | **3,201** | **1,632** | **4,833** | |

The frozen split is defined by [`../dataset_exploration/outputs/validated/54_d2_v11_frozen_split_manifest.csv`](../dataset_exploration/outputs/validated/54_d2_v11_frozen_split_manifest.csv) and its corresponding metadata. Requirements were separated using **similarity groups** to reduce near-duplicate leakage across partitions. The classical ML work uses **train-only five-fold stratified group cross-validation** alongside the fixed validation split. The **test set was not used for classical model selection or threshold tuning**. Project-level independence was not established.

## Experimental methodology

The scripts are sequential research records, numbered **57–69** to follow dataset-exploration Scripts 00–56.

| Scripts | Experiment focus | Main questions |
|---|---|---|
| `57–59` | Logistic regression baselines, error analysis, threshold sweeps | How do class weighting and NFR decision thresholds affect recall and F1? |
| `60–61` | Stratified group cross-validation and out-of-fold (OOF) threshold checks | Are training-only results stable, and do proposed thresholds carry over to validation? |
| `62–63` | Linear SVM and paired validation comparisons | How does a linear margin classifier compare on the same requirements? |
| `64–65` | Additional classical baselines and RBF SVM | Do alternative estimators or nonlinear decision boundaries help? |
| `66–67` | Word and character TF-IDF features, feature comparisons | Does representing character patterns alongside words improve classification? |
| `68–69` | Tree/ensemble baselines and consolidated summary | What do the combined baseline results show? |

Experiments use lexical text features (including **TF-IDF**) and evaluate the minority **NFR** class explicitly. Saved OOF predictions are used for train-only analysis; validation predictions, paired comparisons, and error-overlap tables support interpretation. Any threshold explored on validation is treated as exploratory, not as a test-set result.

### Selected validation findings

| Finding | Recorded validation result |
|---|---:|
| Balanced linear SVM, NFR F1 | **0.8421** |
| Balanced linear SVM, NFR recall | **0.8703** |
| Balanced linear SVM, macro-F1 | **0.8805** |
| Balanced linear SVM, balanced accuracy | **0.8871** |
| Balanced linear SVM, errors | 47 false positives; 31 false negatives |

The balanced linear SVM was a strong **validation** baseline; these figures are **not classical-model test scores**. Other configurations and fold-level results are preserved in [`outputs/tables/69_d2_classical_ml_leaderboard.csv`](outputs/tables/69_d2_classical_ml_leaderboard.csv) and [`outputs/tables/69_d2_classical_ml_cv_comparison.csv`](outputs/tables/69_d2_classical_ml_cv_comparison.csv). See also [`outputs/reports/69_d2_classical_ml_summary.txt`](outputs/reports/69_d2_classical_ml_summary.txt) for the consolidated interpretation.

## Directory guide

```text
research_modeling/
├── scripts/                 # Scripts 57–69
├── outputs/
│   ├── models/              # Saved classical-model artifacts, if retained locally
│   ├── tables/              # CV, OOF, validation, and comparison CSVs
│   └── reports/             # Experiment TXT reports and JSON metadata
├── Docs/                    # Detailed research record
├── code_outputs_research_modeling.txt
└── README.md
```

**Useful starting points:**

- [`Docs/THESIS_DATASET_2_CLASSICAL_ML_RESEARCH_MODELING_FULL_REPORT.txt`](Docs/THESIS_DATASET_2_CLASSICAL_ML_RESEARCH_MODELING_FULL_REPORT.txt): detailed experiment narrative and interpretation.
- [`outputs/reports/60_d2_grouped_cv_report.txt`](outputs/reports/60_d2_grouped_cv_report.txt): training-only grouped cross-validation.
- [`outputs/reports/63_d2_paired_comparison.txt`](outputs/reports/63_d2_paired_comparison.txt): paired validation comparisons.
- [`outputs/reports/69_d2_classical_ml_summary.txt`](outputs/reports/69_d2_classical_ml_summary.txt): consolidated classical baseline summary.

## Relationship to subsequent research

The next stage, [`../deep_learning_research/`](../deep_learning_research/), evaluates TextCNN, BiLSTM, and transformer models on the **same frozen D2 v1.1 split**. The selected DistilBERT model and its **one-time held-out test evaluation** are documented separately in [`../final_model/`](../final_model/). That final test evaluation is **not** a test-set comparison of all the classical models.

**Limitations:** results come from one selected dataset source and a reused validation split, source-label consistency and project-level generalization remain concerns. Original datasets and some prediction/model artifacts may be excluded from the public Git repository. This folder documents the experimental evidence without claiming that the validation leaderboard establishes performance on unseen projects.
