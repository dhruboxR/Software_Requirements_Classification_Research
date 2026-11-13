# Dataset Exploration and Preparation

This directory documents the examination, quality assessment, provenance checks, and preparation of a multi-source software requirements dataset for **Functional Requirement (FR) versus Non-Functional Requirement (NFR) classification**. The objective was to select a defensible research subset and establish a reproducible, leakage-aware evaluation split before model development.

## Workflow

The investigation progressed through four groups of scripts. The numbering reflects the order of analysis rather than four independent datasets.

| Stage | Scripts | Main activities |
|---|---|---|
| Initial exploration | `scripts_1/` (00–07) | File inventory, schema comparison, missing values, class distribution, project/source coverage, text lengths, and overlap candidates |
| Source and label diagnostics | `scripts_2/` (08–18) | Source provenance matching, label taxonomy, conflicting annotations, repeated texts, D4/D5 audits, and PROMISE overlap |
| Dataset suitability | `scripts_3/` (19–24) | Independent class support, duplicate and conflict risk, project coverage, and candidate dataset comparison |
| Final verification and freezing | `scripts_4/` (25–56) | Label and source audits, near-duplicate checks, grouping/split feasibility, candidate manifests, D2 protocol revisions, frozen D2 v1.1 split, and readiness audit |

The initial merged dataset contained **8,854 requirements** from five source identifiers (`D1`–`D5`), with heterogeneous labels, duplicate texts, and annotation conflicts. Those issues were assessed before selecting **D2** for the binary FR/NFR research task. The original source datasets are stored separately at the project root and may be excluded from Git.

## Frozen research dataset: D2 v1.1

The final validated research split is recorded in `outputs/validated/54_d2_v11_frozen_split_manifest.csv`, with its configuration and integrity hashes in `outputs/validated/54_d2_v11_split_metadata.json`.

| Partition | Requirements | FR | NFR | Similarity groups |
|---|---:|---:|---:|---:|
| Train | 3,378 | 2,224 | 1,154 | 3,237 |
| Validation | 728 | 489 | 239 | 694 |
| Test (held out) | 727 | 488 | 239 | 694 |
| **Total** | **4,833** | **3,201** | **1,632** | **4,625** |

**Split methodology:** The D2 candidates underwent duplicate/near-duplicate grouping, label consistency checks, partition feasibility assessment, and cross-partition similarity auditing. The resulting manifest keeps the audited similarity groups within individual partitions. Dataset readiness was verified in Scripts **54–56**. The held-out test split was reserved for evaluation after model selection.

This split reduces leakage caused by closely related texts, but it **does not establish project-level independence**. The source-label taxonomy, data provenance, and independent human annotation verification also retain documented limitations; the readiness decision should not be interpreted as unconditional dataset-quality approval.

## Exploratory figures

**Original merged-dataset class distribution**

![Class distribution of the multi-source dataset](outputs/figures/03_class_distribution.png)

**Requirement word-count distribution**

![Word-count histogram](outputs/figures/05_word_count_histogram.png)

These figures describe early exploration of the merged dataset; they should **not** be interpreted as the final D2 v1.1 class distribution. Use the frozen split table above for final research counts.

## Directory guide

| Path | Contents |
|---|---|
| `scripts_1/` … `scripts_4/` | Numbered, reproducible exploration and audit scripts |
| `outputs/tables/` | Tabular summaries, matching evidence, conflicts, grouping diagnostics, and split audits |
| `outputs/reports/` | Methodology notes, limitations, readiness records, and protocol documents |
| `outputs/figures/` | Exploratory visualizations |
| `outputs/validated/` | Candidate datasets, frozen manifest, split metadata, and review records |
| `Docs/THESIS_DATASET_2_COMPLETE_EXPLORATION_RECORD.txt` | Full chronological exploration record |
| `code_outputs_scripts_1.txt` … `code_outputs_scripts_4.txt` | Saved execution logs and results for each script group |

## Reproducibility

Run scripts from the dataset exploration workspace with the project's existing Python environment. Earlier scripts depend on the original dataset files under the parent `THESIS_DATASET_2/` directory. Consult each script's imports and input paths before rerunning stages; later scripts often require artifacts from earlier stages.

Key records for checking the **final dataset protocol**:

- `outputs/reports/46_d2_research_dataset_protocol.md` and `outputs/reports/50_d2_protocol_revision_plan.txt`: protocol background and revision rationale.
- `outputs/reports/54_d2_v11_split_report.txt`: frozen split summary.
- `outputs/reports/55_d2_v11_similarity_audit.txt`: cross-partition similarity audit.
- `outputs/reports/56_d2_v11_readiness.txt` and `.json`: final readiness checks.

**Repository note:** The frozen manifest and some diagnostic CSVs may contain original requirement text. Check the source dataset's redistribution terms before committing files from `outputs/validated/` or `outputs/tables/`; ignoring only the raw dataset folder does not exclude these derived copies.

The modeling stages are maintained separately in `../research_modeling/`, `../deep_learning_research/`, and `../final_model/`.