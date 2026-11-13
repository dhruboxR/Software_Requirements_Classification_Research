# D2 Research Dataset Protocol

**Protocol version:** 1.0-candidate

**Research task:** Binary classification of software requirements.

**Current status:** Technical validation completed; scientific
approval pending.

## 1. Dataset provenance

The primary candidate dataset is the D2 subset of the Unified
Multi-Source Software Requirements Dataset.

The immediate original source used for annotation comparison is
`FR_NFR_Dataset Original.xlsx`, worksheet `datasetFR_NFR_full`.

D2 originally contains 4,993 records in the unified collection.
The audit identified 4,871 provisionally eligible unique texts
before the final source-label reliability filter.

This protocol describes a derived candidate dataset, not a
new independently annotated corpus.

## 2. Classification task

Two labels are retained:

- **FR:** Functional Requirement
- **NFR:** Non-Functional Requirement

NFR is the minority class.

### Class distribution

| Class | Count | Percentage |
| --- | --- | --- |
| FR | 3202 | 66.23 |
| NFR | 1633 | 33.77 |

## 3. Text normalization and duplicate handling

Text keys use:

1. Unicode NFKC normalization.
2. Whitespace collapse.
3. Leading and trailing whitespace removal.
4. Unicode case folding.

The candidate set contains one row per normalized text key.

The original requirement text is retained in the manifest.

Normalization supports duplicate detection. It does not
constitute semantic paraphrase detection.

## 4. Annotation eligibility

Requirements with unresolved source-label discrepancies,
blank labels, ambiguous source labels, or unmatched original
texts are withheld pending additional review.

Previously identified near-duplicate annotation disagreements
were also provisionally quarantined before this candidate
manifest was constructed.

### Final source-label audit exclusions

| Status | Count |
| --- | --- |
| AMBIGUOUS_SOURCE_LABEL | 1 |
| BLANK_SOURCE_LABEL | 3 |
| SOURCE_LABEL_DISAGREES | 29 |
| UNMATCHED_SOURCE_TEXT | 3 |

Total withheld at this stage: **36**.

Total source-consistent candidate texts:
**4835**.

These counts describe the final source-label reliability stage;
earlier conflict and similarity-review exclusions are not
additional disjoint counts and must not be added mechanically.

Source agreement means that the unified label matches the
immediate original source's label. It does not establish
independent semantic correctness.

## 5. Similarity-group construction

Character n-gram TF-IDF candidate similarities were identified
using n-grams of length 3 through 5 and a candidate cosine
similarity threshold of 0.85.

The prior candidate search used a top-10-neighbor limit.
Therefore, similarity detection is not exhaustive.

Identified similarity edges were joined through connected
components. All records within a component were assigned
to the same partition.

**Detected similarity groups:** 4629

**Largest component:** 6 requirements

**Components crossing partitions:** 0

The zero crossing count refers only to detected similarity
components. It does not prove the absence of all semantic
or source-level leakage.

## 6. Frozen evaluation design

The split was generated using grouped random partitioning
with predefined random seed 42.

### Partition distribution

| Partition | FR | NFR | Total | Percentage |
| --- | --- | --- | --- | --- |
| train | 2256 | 1136 | 3392 | 70.16 |
| validation | 472 | 251 | 723 | 14.95 |
| test | 474 | 246 | 720 | 14.89 |

The training set will be used for model fitting.

The validation set will be used for hyperparameter selection,
decision-threshold selection, and model comparisons.

The test set must remain sealed until the model-selection
protocol has been finalized.

Do not repeatedly inspect test performance to choose models,
thresholds, text-cleaning decisions, or split seeds.

## 7. Technical validation

Script 45 reported **16/16 passing technical checks**.

These covered file integrity, candidate and manifest consistency,
unique text keys, class counts, preserved grouping membership,
and disjoint identified similarity groups.

File SHA256 hashes are recorded in the companion JSON protocol.

## 8. Evaluation metrics

Primary metrics:

- NFR recall
- NFR precision
- NFR F1-score
- Macro-F1
- Balanced accuracy

Secondary evaluation:

- Per-class support and FR performance
- Confusion matrix
- Overall accuracy
- Confidence intervals for selected metrics

Any threshold optimization must use training/validation data
only, not the sealed test partition.

## 9. Scientific limitations

1. D2's original-project identities are not independently
   verified.
2. The evaluation must not be presented as verified
   independent-project generalization.
3. Matching original-source labels do not guarantee that
   the semantic annotations are correct.
4. Near-duplicate detection uses a limited candidate search.
5. Semantically equivalent paraphrases may remain in
   different partitions.
6. Preliminary annotation review requires documented
   human adjudication or explicit exclusion.
7. Large minority-class support improves evaluation
   precision but cannot guarantee strong NFR recall.

## 10. Approval conditions

The following activities remain before scientific approval:

- Record the final annotation-review decisions.
- Perform and document the final contamination assessment
  without using test performance.
- Freeze preprocessing, feature extraction, and metric
  definitions.
- Preserve the frozen test partition.

If annotation decisions require changing candidate membership,
the current manifest must be treated as superseded and a
new protocol version must be created before model training.

## 11. Current decision

**D2 is selected as the primary candidate for binary FR/NFR
classification.**

Technical integrity: **PASS**

Scientific approval: **PENDING**

Model development: **NOT STARTED**

Final test evaluation: **NOT STARTED**
