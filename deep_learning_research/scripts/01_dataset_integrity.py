
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

# ============================================================
# SCRIPT 01: D2 v1.1 DEEP LEARNING DATASET INTEGRITY
#
# Verifies:
# - Frozen dataset SHA256
# - Dataset protocol and requirement counts
# - Unique IDs and normalized text keys
# - FR/NFR distribution
# - Similarity-group isolation
# - Agreement with final dataset readiness record
#
# Does not load requirement text into the audit dataframe.
# Does not create new dataset partitions.
# Does not train or evaluate models.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DL_ROOT = ROOT / "deep_learning_research"

DATA_ROOT = (
    ROOT / "dataset_exploration" / "outputs"
)

MANIFEST = (
    DATA_ROOT / "validated"
    / "54_d2_v11_frozen_split_manifest.csv"
)

SPLIT_META = (
    DATA_ROOT / "validated"
    / "54_d2_v11_split_metadata.json"
)

READINESS = (
    DATA_ROOT / "reports"
    / "56_d2_v11_readiness.json"
)

REPORTS = DL_ROOT / "outputs" / "reports"
TABLES = DL_ROOT / "outputs" / "tables"

REPORTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "train": {"FR": 2224, "NFR": 1154},
    "validation": {"FR": 489, "NFR": 239},
    "test": {"FR": 488, "NFR": 239},
}

EXPECTED_REQUIREMENTS = 4833
EXPECTED_GROUPS = 4625

REPORT_JSON = REPORTS / "01_dataset_integrity.json"
REPORT_TXT = REPORTS / "01_dataset_integrity.txt"
SPLIT_TABLE = TABLES / "01_partition_summary.csv"


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def verify(condition, description):
    if not condition:
        raise RuntimeError(
            f"FAILED: {description}"
        )
    print(f"PASS: {description}")


print("=" * 76)
print("SCRIPT 01: D2 v1.1 DATASET INTEGRITY VERIFICATION")
print("=" * 76)

# ------------------------------------------------------------
# 1. Verify required inputs
# ------------------------------------------------------------

for path in [MANIFEST, SPLIT_META, READINESS]:
    verify(
        path.is_file(),
        f"Required file exists: {path.name}"
    )

split_meta = json.loads(
    SPLIT_META.read_text(encoding="utf-8")
)

readiness = json.loads(
    READINESS.read_text(encoding="utf-8")
)

verify(
    split_meta.get("protocol") == "D2-v1.1",
    "Dataset protocol is D2-v1.1"
)

verify(
    readiness.get("protocol") == "D2-v1.1",
    "Readiness protocol matches"
)

manifest_hash = sha256(MANIFEST)

verify(
    manifest_hash
    == split_meta["output_sha256"][MANIFEST.name],
    "Frozen manifest SHA256 matches split metadata"
)

verify(
    manifest_hash
    == readiness["frozen_manifest_sha256"],
    "Frozen manifest SHA256 matches readiness record"
)

# ------------------------------------------------------------
# 2. Load metadata columns only
# ------------------------------------------------------------

# Requirement text is deliberately excluded.
# Test requirements are counted for integrity only.

columns = [
    "Requirement ID",
    "key",
    "label",
    "split",
    "similarity_group",
]

df = pd.read_csv(
    MANIFEST,
    usecols=columns,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

print("\nDATASET COUNTS")

verify(
    len(df) == EXPECTED_REQUIREMENTS,
    "Total requirements: 4833"
)

verify(
    df["Requirement ID"].is_unique,
    "Requirement IDs are unique"
)

verify(
    df["key"].is_unique,
    "Normalized requirement text keys are unique"
)

for column in columns:
    verify(
        df[column].str.strip().ne("").all(),
        f"No blank values: {column}"
    )

verify(
    set(df["label"]) == {"FR", "NFR"},
    "Only FR and NFR labels exist"
)

verify(
    set(df["split"]) == set(EXPECTED),
    "Only train/validation/test partitions exist"
)

verify(
    df["similarity_group"].nunique()
    == EXPECTED_GROUPS,
    "Exactly 4625 similarity groups"
)

# ------------------------------------------------------------
# 3. Verify frozen class distribution
# ------------------------------------------------------------

counts = pd.crosstab(
    df["split"],
    df["label"]
).reindex(
    index=["train", "validation", "test"],
    columns=["FR", "NFR"],
    fill_value=0
)

counts["total"] = counts.sum(axis=1)

print("\nCLASS DISTRIBUTION")
print(counts.to_string())

for partition, class_counts in EXPECTED.items():
    for label, expected_count in class_counts.items():
        verify(
            int(counts.loc[partition, label])
            == expected_count,
            f"{partition}/{label}: {expected_count}"
        )

        verify(
            split_meta["class_counts"][partition][label]
            == expected_count,
            f"Metadata agrees: {partition}/{label}"
        )

verify(
    int(counts["total"].sum())
    == EXPECTED_REQUIREMENTS,
    "All requirements belong to one partition"
)

# ------------------------------------------------------------
# 4. Verify similarity-group isolation
# ------------------------------------------------------------

group_split_count = (
    df.groupby("similarity_group")["split"]
    .nunique()
)

cross_partition_groups = int(
    (group_split_count > 1).sum()
)

group_label_count = (
    df.groupby("similarity_group")["label"]
    .nunique()
)

mixed_label_groups = int(
    (group_label_count > 1).sum()
)

verify(
    cross_partition_groups == 0,
    "No similarity groups cross partitions"
)

verify(
    mixed_label_groups == 0,
    "No mixed-label similarity groups"
)

groups_per_partition = (
    df.groupby("split")["similarity_group"]
    .nunique()
    .to_dict()
)

print("\nGROUP STATISTICS")

for partition in ["train", "validation", "test"]:
    print(
        f"{partition} groups:",
        groups_per_partition[partition]
    )

print(
    "Cross-partition groups:",
    cross_partition_groups
)

print("Mixed-label groups:", mixed_label_groups)

# ------------------------------------------------------------
# 5. Confirm historical readiness
# ------------------------------------------------------------

verify(
    readiness["research_status"]["technical_readiness"]
    == "PASS",
    "Previous dataset technical readiness passed"
)

verify(
    readiness["technical_checks"][
        "cross_partition_similarity_ge_085"
    ] == 0,
    "Previous 0.85 similarity screen passed"
)

# ------------------------------------------------------------
# 6. Save audit records
# ------------------------------------------------------------

counts.to_csv(
    SPLIT_TABLE,
    encoding="utf-8"
)

report = {
    "script": "01_dataset_integrity",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "protocol": "D2-v1.1",
    "frozen_manifest_sha256": manifest_hash,
    "requirements": len(df),
    "classes": ["FR", "NFR"],
    "similarity_groups": int(
        df["similarity_group"].nunique()
    ),
    "cross_partition_groups": cross_partition_groups,
    "mixed_label_groups": mixed_label_groups,
    "partitions": {
        partition: {
            "FR": int(counts.loc[partition, "FR"]),
            "NFR": int(counts.loc[partition, "NFR"]),
            "total": int(counts.loc[partition, "total"]),
            "groups": int(
                groups_per_partition[partition]
            )
        }
        for partition in EXPECTED
    },
    "integrity_status": "PASS",
    "historical_readiness": "PASS",
    "scientific_approval": "CONDITIONAL",
    "independent_annotation_validation": "NOT_PERFORMED",
    "project_independence": "NOT_ESTABLISHED",
    "test_text_loaded_into_audit": False,
    "test_predictions_generated": False,
    "model_training_started": False,
    "dataset_modified": False,
}

REPORT_JSON.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

lines = [
    "SCRIPT 01: D2 v1.1 DATASET INTEGRITY",
    "=" * 70,
    "",
    "DATASET",
    f"Protocol: {report['protocol']}",
    f"Requirements: {len(df)}",
    f"Similarity groups: {report['similarity_groups']}",
    f"Manifest SHA256: {manifest_hash}",
    "",
    "CLASS DISTRIBUTION",
    counts.to_string(),
    "",
    "GROUP INTEGRITY",
    f"Cross-partition groups: {cross_partition_groups}",
    f"Mixed-label groups: {mixed_label_groups}",
    "",
    "RESEARCH STATUS",
    "Dataset integrity: PASS",
    "Previous technical readiness: PASS",
    "Scientific approval: CONDITIONAL",
    "Test evaluation: NOT PERFORMED",
    "Model training: NOT STARTED",
    "Frozen dataset modified: NO"
]

REPORT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Dataset integrity: PASS")
print("Previous technical readiness: PASS")
print("Scientific approval: CONDITIONAL")
print("Test evaluation: NOT PERFORMED")
print("Model training: NOT STARTED")
print("Frozen dataset modified: NO")

print("\nFILES GENERATED")
print(REPORT_JSON)
print(REPORT_TXT)
print(SPLIT_TABLE)

print("\nScript 01 completed.")
