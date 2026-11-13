
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd

# ============================================================
# SCRIPT 46: D2 RESEARCH DATASET PROTOCOL
# Documentation only. No dataset modification or splitting.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
REPORTS = BASE / "reports"
VALIDATED = BASE / "validated"

REPORTS.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILE = (
    VALIDATED / "42_d2_source_consistent_candidates.csv"
)

WITHHELD_FILE = (
    VALIDATED / "42_d2_withheld_label_cases.csv"
)

MANIFEST_FILE = (
    VALIDATED / "44_d2_frozen_split_manifest.csv"
)

METADATA_FILE = (
    VALIDATED / "44_d2_split_metadata.json"
)

GROUP_FILE = (
    TABLES / "43_d2_candidate_group_membership.csv"
)

AUDIT_FILE = (
    REPORTS / "45_d2_final_approval_audit.txt"
)

REQUIRED_FILES = [
    CANDIDATE_FILE,
    WITHHELD_FILE,
    MANIFEST_FILE,
    METADATA_FILE,
    GROUP_FILE,
    AUDIT_FILE,
]

for path in REQUIRED_FILES:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def load_csv(path):
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


def markdown_table(frame):
    columns = list(frame.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = (
        "| " + " | ".join(["---"] * len(columns)) + " |"
    )

    rows = [
        "| " + " | ".join(
            str(value).replace("|", "\\|")
            for value in row
        ) + " |"
        for row in frame.itertuples(
            index=False, name=None
        )
    ]

    return "\n".join(
        [header, separator] + rows
    )


# ============================================================
# 1. Load frozen evidence
# ============================================================

candidate = load_csv(CANDIDATE_FILE)
withheld = load_csv(WITHHELD_FILE)
manifest = load_csv(MANIFEST_FILE)
membership = load_csv(GROUP_FILE)

metadata = json.loads(
    METADATA_FILE.read_text(encoding="utf-8")
)

audit_text = AUDIT_FILE.read_text(
    encoding="utf-8"
)

required_columns = {
    "Requirement ID",
    "key",
    "label",
    "similarity_group",
    "split"
}

if not required_columns.issubset(manifest.columns):
    raise ValueError(
        "Frozen manifest missing required columns"
    )

if not candidate["key"].is_unique:
    raise ValueError("Candidate text keys are not unique")

if not manifest["key"].is_unique:
    raise ValueError("Manifest text keys are not unique")

if set(candidate["key"]) != set(manifest["key"]):
    raise ValueError(
        "Candidate and manifest texts differ"
    )

if set(candidate["key"]) & set(withheld["key"]):
    raise ValueError(
        "Candidate overlaps withheld records"
    )

if len(manifest) != 4835:
    raise ValueError("Unexpected candidate count")

# Check hashes recorded during Script 44.
hash_checks = {
    "candidate": (
        sha256(CANDIDATE_FILE)
        == metadata["candidate_sha256"]
    ),
    "membership": (
        sha256(GROUP_FILE)
        == metadata["membership_sha256"]
    ),
    "manifest": (
        sha256(MANIFEST_FILE)
        == metadata["manifest_sha256"]
    )
}

if not all(hash_checks.values()):
    raise RuntimeError(
        "Frozen file hash mismatch. Stop."
    )

if "Checks passed: 16/16" not in audit_text:
    raise RuntimeError(
        "Script 45 technical audit not confirmed"
    )

# ============================================================
# 2. Compute dataset statistics
# ============================================================

split_order = ["train", "validation", "test"]

counts = (
    manifest.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(split_order, fill_value=0)
    .reindex(columns=["FR", "NFR"], fill_value=0)
)

counts["Total"] = counts.sum(axis=1)

counts["Percentage"] = (
    counts["Total"] / len(manifest) * 100
).round(2)

groups_crossing = int(
    (
        manifest.groupby(
            "similarity_group"
        )["split"].nunique() > 1
    ).sum()
)

if groups_crossing:
    raise RuntimeError(
        "Similarity group leakage detected"
    )

candidate_labels = (
    candidate["label"]
    .value_counts()
    .reindex(["FR", "NFR"], fill_value=0)
)

withheld_status = (
    withheld["source_label_status"]
    .value_counts()
    .to_dict()
)

group_count = int(
    manifest["similarity_group"].nunique()
)

largest_group = int(
    manifest.groupby(
        "similarity_group"
    ).size().max()
)

# ============================================================
# 3. Machine-readable research protocol
# ============================================================

protocol = {
    "protocol_name": (
        "D2 Binary Requirements Classification"
    ),
    "version": "1.0-candidate",
    "generated_utc": (
        datetime.now(timezone.utc).isoformat()
    ),
    "research_task": {
        "type": "binary_text_classification",
        "labels": {
            "FR": "Functional Requirement",
            "NFR": "Non-Functional Requirement"
        },
        "minority_class": "NFR"
    },
    "dataset": {
        "parent_collection": (
            "Unified Multi-Source "
            "Software Requirements Dataset"
        ),
        "source_subset": "D2",
        "immediate_source": (
            "FR_NFR_Dataset Original.xlsx"
        ),
        "original_d2_rows": 4993,
        "audited_unique_texts": 4871,
        "source_consistent_candidates": len(candidate),
        "withheld_source_label_cases": len(withheld),
        "class_counts": {
            label: int(count)
            for label, count
            in candidate_labels.items()
        }
    },
    "preprocessing": {
        "text_key": (
            "Unicode NFKC, whitespace collapse, "
            "strip, casefold"
        ),
        "deduplication": (
            "One candidate per normalized text key"
        ),
        "source_label_rule": (
            "Retain only records with one "
            "matching original-source label"
        ),
        "annotation_review": (
            "Previously flagged similarity-disagreement "
            "texts provisionally withheld"
        ),
        "semantic_label_verification": (
            "Not independently completed"
        )
    },
    "similarity_control": {
        "method": (
            "Character n-gram TF-IDF cosine "
            "similarity candidate graph"
        ),
        "ngram_range": [3, 5],
        "candidate_threshold": 0.85,
        "neighbor_search_limit": 10,
        "components": group_count,
        "largest_component": largest_group,
        "cross_partition_components": (
            groups_crossing
        ),
        "scope": (
            "Detected similarity candidates only"
        )
    },
    "partition": {
        "status": "FROZEN_PENDING_FINAL_APPROVAL",
        "seed": metadata["seed"],
        "method": metadata["protocol"],
        "requested_ratio": [0.70, 0.15, 0.15],
        "counts": {
            split: {
                label: int(counts.loc[split, label])
                for label in ["FR", "NFR"]
            }
            for split in split_order
        },
        "test_set_sealed": True
    },
    "technical_audit": {
        "passed": 16,
        "total": 16,
        "status": "PASS"
    },
    "approval": {
        "technical": "PASS",
        "scientific": "PENDING",
        "model_training": "NOT_STARTED",
        "final_test_evaluation": "NOT_STARTED"
    },
    "files_sha256": {
        path.name: sha256(path)
        for path in REQUIRED_FILES
    },
    "limitations": [
        "Original project independence is unverified.",
        "Source agreement does not prove semantic correctness.",
        "Near-duplicate candidate search is not exhaustive.",
        "Undetected semantic paraphrases may cross partitions.",
        "Preliminary human annotation review remains unresolved.",
        "Class counts do not guarantee predictive performance."
    ]
}

# ============================================================
# 4. Research methodology document
# ============================================================

class_rows = pd.DataFrame([
    {
        "Class": "FR",
        "Count": int(candidate_labels["FR"]),
        "Percentage": round(
            100 * candidate_labels["FR"] /
            len(candidate), 2
        )
    },
    {
        "Class": "NFR",
        "Count": int(candidate_labels["NFR"]),
        "Percentage": round(
            100 * candidate_labels["NFR"] /
            len(candidate), 2
        )
    }
])

split_table = counts.reset_index().rename(
    columns={"split": "Partition"}
)

withheld_table = pd.DataFrame([
    {
        "Status": k,
        "Count": int(v)
    }
    for k, v in sorted(
        withheld_status.items()
    )
])

document = f"""# D2 Research Dataset Protocol

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

{markdown_table(class_rows)}

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

{markdown_table(withheld_table)}

Total withheld at this stage: **{len(withheld)}**.

Total source-consistent candidate texts:
**{len(candidate)}**.

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

**Detected similarity groups:** {group_count}

**Largest component:** {largest_group} requirements

**Components crossing partitions:** {groups_crossing}

The zero crossing count refers only to detected similarity
components. It does not prove the absence of all semantic
or source-level leakage.

## 6. Frozen evaluation design

The split was generated using grouped random partitioning
with predefined random seed {metadata["seed"]}.

### Partition distribution

{markdown_table(split_table)}

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
"""

# ============================================================
# 5. Save protocol documents
# ============================================================

markdown_path = (
    REPORTS / "46_d2_research_dataset_protocol.md"
)

json_path = (
    REPORTS / "46_d2_research_dataset_protocol.json"
)

summary_path = (
    REPORTS / "46_d2_protocol_status.txt"
)

markdown_path.write_text(
    document, encoding="utf-8"
)

json_path.write_text(
    json.dumps(protocol, indent=2),
    encoding="utf-8"
)

summary_lines = [
    "SCRIPT 46: D2 RESEARCH PROTOCOL",
    "=" * 65,
    f"Candidate rows: {len(candidate)}",
    f"FR rows: {candidate_labels['FR']}",
    f"NFR rows: {candidate_labels['NFR']}",
    f"Withheld cases: {len(withheld)}",
    f"Similarity groups: {group_count}",
    f"Cross-partition groups: {groups_crossing}",
    "Technical audit: PASS (16/16)",
    "Scientific approval: PENDING",
    "Model training: NOT STARTED",
    "",
    f"Markdown: {markdown_path}",
    f"JSON: {json_path}"
]

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8"
)

print("=" * 72)
print("SCRIPT 46: RESEARCH DATASET PROTOCOL")
print("=" * 72)

print("\nDATASET")
print("Candidate requirements:", len(candidate))
print("FR:", int(candidate_labels["FR"]))
print("NFR:", int(candidate_labels["NFR"]))
print("Withheld cases:", len(withheld))

print("\nGROUPING AND INTEGRITY")
print("Similarity groups:", group_count)
print("Largest group:", largest_group)
print("Cross-partition groups:", groups_crossing)
print("Frozen file hashes verified:", all(hash_checks.values()))
print("Technical audit: PASS (16/16)")

print("\nRESEARCH STATUS")
print("Primary candidate: D2")
print("Scientific approval: PENDING")
print("Model training: NOT STARTED")
print("Test evaluation: NOT STARTED")

print("\nFILES GENERATED")
print(markdown_path)
print(json_path)
print(summary_path)

print("\nScript 46 completed.")
