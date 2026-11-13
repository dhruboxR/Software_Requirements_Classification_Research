
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# SCRIPT 54: D2 v1.1 FROZEN EVALUATION SPLIT
#
# - Uses the v1.1 candidate dataset
# - Preserves updated similarity groups
# - Uses a predefined random seed
# - Creates train/validation/test partitions
# - Checks class support and group isolation
# - Saves hashes and protocol metadata
#
# Does not modify any v1.0 files.
# Does not train or evaluate a model.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

VALIDATED = BASE / "validated"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

CANDIDATES = (
    VALIDATED / "53_d2_v11_candidates.csv"
)

GROUPS = (
    TABLES / "53_d2_v11_group_membership.csv"
)

QUARANTINE = (
    VALIDATED / "53_d2_v11_quarantine_audit.csv"
)

PREPARATION_METADATA = (
    REPORTS / "53_d2_v11_preparation_metadata.json"
)

MANIFEST = (
    VALIDATED / "54_d2_v11_frozen_split_manifest.csv"
)

METADATA = (
    VALIDATED / "54_d2_v11_split_metadata.json"
)

SUMMARY_FILE = (
    TABLES / "54_d2_v11_split_summary.csv"
)

REPORT = (
    REPORTS / "54_d2_v11_split_report.txt"
)

SEED = 42
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
RATIO_TOLERANCE = 0.05
MIN_NFR_TEST = 100

for directory in (VALIDATED, TABLES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


print("=" * 76)
print("SCRIPT 54: D2 v1.1 FROZEN EVALUATION SPLIT")
print("=" * 76)

# ------------------------------------------------------------
# 1. Prevent overwriting an existing frozen version
# ------------------------------------------------------------

outputs = [
    MANIFEST,
    METADATA,
    SUMMARY_FILE,
    REPORT
]

existing = [
    str(path)
    for path in outputs
    if path.exists()
]

if existing:
    raise FileExistsError(
        "Existing v1.1 split artifacts detected.\n"
        "Files will not be overwritten:\n"
        + "\n".join(existing)
    )

# ------------------------------------------------------------
# 2. Load version 1.1 evidence
# ------------------------------------------------------------

candidate = read_csv(CANDIDATES)
membership = read_csv(GROUPS)
quarantine = read_csv(QUARANTINE)

preparation = json.loads(
    PREPARATION_METADATA.read_text(
        encoding="utf-8"
    )
)

if preparation.get("protocol") != "D2-v1.1":
    raise RuntimeError(
        "Unexpected dataset protocol version"
    )

if preparation.get("candidate_requirements") != 4833:
    raise RuntimeError(
        "Unexpected preparation metadata"
    )

# Verify the files have not changed since Script 53.
expected_hashes = preparation.get(
    "output_sha256", {}
)

for path in [CANDIDATES, GROUPS, QUARANTINE]:
    if sha256(path) != expected_hashes.get(path.name):
        raise RuntimeError(
            f"Input file integrity failure: {path}"
        )

required_candidate = {
    "Requirement ID",
    "key",
    "label"
}

required_groups = {
    "Requirement ID",
    "key",
    "label",
    "similarity_group"
}

for name, frame, required in [
    ("Candidate", candidate, required_candidate),
    ("Membership", membership, required_groups)
]:
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )

if "split" in candidate.columns:
    raise RuntimeError(
        "Candidate unexpectedly contains split assignments"
    )

if len(candidate) != 4833:
    raise ValueError(
        "Expected 4833 candidate requirements"
    )

if len(membership) != len(candidate):
    raise ValueError(
        "Candidate and membership counts differ"
    )

if not candidate["key"].is_unique:
    raise ValueError("Duplicate candidate text keys")

if not candidate["Requirement ID"].is_unique:
    raise ValueError("Duplicate requirement IDs")

if not membership["key"].is_unique:
    raise ValueError("Duplicate membership text keys")

if set(candidate["key"]) != set(membership["key"]):
    raise ValueError(
        "Candidate and membership text keys differ"
    )

# ------------------------------------------------------------
# 3. Merge and validate grouping assignments
# ------------------------------------------------------------

df = candidate.merge(
    membership[
        [
            "Requirement ID",
            "key",
            "label",
            "similarity_group"
        ]
    ],
    on="key",
    how="left",
    suffixes=("", "_membership"),
    validate="one_to_one",
    indicator=True
)

if not (df["_merge"] == "both").all():
    raise RuntimeError(
        "Unmatched group membership records"
    )

if not (
    df["Requirement ID"]
    == df["Requirement ID_membership"]
).all():
    raise RuntimeError(
        "Requirement ID mismatch"
    )

if not (
    df["label"]
    == df["label_membership"]
).all():
    raise RuntimeError(
        "Candidate label mismatch"
    )

df = df.drop(
    columns=[
        "_merge",
        "Requirement ID_membership",
        "label_membership"
    ]
)

if df["similarity_group"].eq("").any():
    raise RuntimeError(
        "Missing similarity group assignments"
    )

if set(df["label"]) != {"FR", "NFR"}:
    raise RuntimeError(
        "Unexpected classification labels"
    )

quarantine_ids = set(
    quarantine["Requirement ID"]
)

if quarantine_ids & set(df["Requirement ID"]):
    raise RuntimeError(
        "Quarantined requirements are present"
    )

# Deterministic input ordering.
df = df.sort_values(
    "key"
).reset_index(drop=True)

# ------------------------------------------------------------
# 4. Generate grouped partitions
# ------------------------------------------------------------

indices = np.arange(len(df))

groups = df[
    "similarity_group"
].to_numpy()

outer = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_FRACTION,
    random_state=SEED
)

trainval_idx, test_idx = next(
    outer.split(
        indices,
        groups=groups
    )
)

inner = GroupShuffleSplit(
    n_splits=1,
    test_size=(
        VALIDATION_FRACTION
        / (1 - TEST_FRACTION)
    ),
    random_state=SEED + 10000
)

train_local, validation_local = next(
    inner.split(
        trainval_idx,
        groups=groups[trainval_idx]
    )
)

train_idx = trainval_idx[train_local]

validation_idx = trainval_idx[
    validation_local
]

df["split"] = ""

df.loc[train_idx, "split"] = "train"

df.loc[
    validation_idx, "split"
] = "validation"

df.loc[test_idx, "split"] = "test"

# ------------------------------------------------------------
# 5. Validate split integrity
# ------------------------------------------------------------

if df["split"].eq("").any():
    raise RuntimeError(
        "Unassigned candidate requirements"
    )

group_partition_counts = (
    df.groupby("similarity_group")[
        "split"
    ].nunique()
)

crossing_groups = int(
    (group_partition_counts > 1).sum()
)

if crossing_groups != 0:
    raise RuntimeError(
        "Similarity group leakage detected"
    )

counts = (
    df.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(
        ["train", "validation", "test"],
        fill_value=0
    )
    .reindex(
        columns=["FR", "NFR"],
        fill_value=0
    )
)

counts["total"] = counts.sum(axis=1)

counts["fraction"] = (
    counts["total"] / len(df)
)

expected_fractions = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

for split, expected in expected_fractions.items():
    actual = float(
        counts.loc[split, "fraction"]
    )

    if abs(actual - expected) > RATIO_TOLERANCE:
        raise RuntimeError(
            f"{split} proportion failed: {actual:.4f}"
        )

    for label in ["FR", "NFR"]:
        if counts.loc[split, label] == 0:
            raise RuntimeError(
                f"Missing {label} in {split}"
            )

if counts.loc["test", "NFR"] < MIN_NFR_TEST:
    raise RuntimeError(
        "Insufficient NFR test support"
    )

if int(counts["total"].sum()) != len(df):
    raise RuntimeError(
        "Split counts do not total candidate count"
    )

# ------------------------------------------------------------
# 6. Verify all new connections
# ------------------------------------------------------------

required_pairs = [
    ("RQP0482022", "RQP0481194"),
    ("RQP0495693", "RQP0495076")
]

id_lookup = df.set_index("Requirement ID")

for a, b in required_pairs:
    group_a = id_lookup.loc[
        a, "similarity_group"
    ]
    group_b = id_lookup.loc[
        b, "similarity_group"
    ]

    split_a = id_lookup.loc[a, "split"]
    split_b = id_lookup.loc[b, "split"]

    if group_a != group_b:
        raise RuntimeError(
            f"Additional group connection failed: {a}, {b}"
        )

    if split_a != split_b:
        raise RuntimeError(
            f"New connection crosses partitions: {a}, {b}"
        )

# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("\nCLASS DISTRIBUTION")
print(counts.to_string())

print("\nGROUP INTEGRITY")
print("Total requirements:", len(df))
print(
    "Unique texts:",
    df["key"].nunique()
)
print(
    "Similarity groups:",
    df["similarity_group"].nunique()
)
print(
    "Groups crossing partitions:",
    crossing_groups
)
print(
    "Additional connections verified:",
    len(required_pairs)
)
print(
    "Quarantined records retained:",
    len(
        quarantine_ids
        & set(df["Requirement ID"])
    )
)

print("\nSPLIT QUALITY")
print(
    "Train fraction:",
    round(
        float(counts.loc["train", "fraction"]),
        4
    )
)
print(
    "Validation fraction:",
    round(
        float(counts.loc["validation", "fraction"]),
        4
    )
)
print(
    "Test fraction:",
    round(
        float(counts.loc["test", "fraction"]),
        4
    )
)
print(
    "NFR test examples:",
    int(counts.loc["test", "NFR"])
)
print("Preliminary quality screen: PASS")

# ------------------------------------------------------------
# 8. Save frozen v1.1 artifacts
# ------------------------------------------------------------

df.to_csv(
    MANIFEST,
    index=False,
    encoding="utf-8-sig"
)

counts.reset_index().to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

metadata = {
    "protocol": "D2-v1.1",
    "status": "FROZEN_PENDING_FINAL_AUDIT",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "seed": SEED,
    "split_method": "GroupShuffleSplit",
    "grouping": (
        "v1.1 connected similarity groups "
        "from Script 53"
    ),
    "requested_fractions": expected_fractions,
    "ratio_tolerance": RATIO_TOLERANCE,
    "candidate_requirements": len(df),
    "similarity_groups": int(
        df["similarity_group"].nunique()
    ),
    "cross_partition_groups": crossing_groups,
    "additional_connections": len(required_pairs),
    "quarantine_count": len(quarantine),
    "class_counts": {
        split: {
            "FR": int(counts.loc[split, "FR"]),
            "NFR": int(counts.loc[split, "NFR"]),
            "total": int(
                counts.loc[split, "total"]
            )
        }
        for split in expected_fractions
    },
    "input_sha256": {
        path.name: sha256(path)
        for path in [
            CANDIDATES,
            GROUPS,
            QUARANTINE,
            PREPARATION_METADATA
        ]
    },
    "output_sha256": {
        MANIFEST.name: sha256(MANIFEST),
        SUMMARY_FILE.name: sha256(SUMMARY_FILE)
    },
    "review_method": "AI_ASSISTED_RULE_BASED",
    "independent_human_review": False,
    "scientific_approval": "PENDING",
    "test_set_policy": (
        "No model-based test evaluation before "
        "final model selection"
    ),
    "limitations": [
        "AI-assisted annotation decisions remain provisional.",
        "No independent human annotation review.",
        "Original-project independence is not established.",
        "Detected similarity-group isolation is not "
        "proof of semantic independence.",
        "Further cross-partition similarity auditing is required."
    ]
}

METADATA.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)

report_lines = [
    "D2 PROTOCOL v1.1 FROZEN SPLIT",
    "=" * 65,
    "",
    "CLASS DISTRIBUTION",
    counts.to_string(),
    "",
    f"Total requirements: {len(df)}",
    f"Similarity groups: {df['similarity_group'].nunique()}",
    f"Cross-partition groups: {crossing_groups}",
    f"Additional connections verified: {len(required_pairs)}",
    "",
    "Split quality screen: PASS",
    "Scientific approval: PENDING",
    "Exhaustive similarity audit: NOT YET PERFORMED",
    "Model training: NOT STARTED",
    "Version 1.0 preserved: YES"
]

REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Protocol: D2-v1.1")
print("Frozen split: CREATED")
print("Group integrity: PASS")
print("Scientific approval: PENDING")
print("Cross-partition similarity audit: PENDING")
print("Version 1.0 modified: NO")
print("Model training: NOT STARTED")

print("\nFILES GENERATED")
print(MANIFEST)
print(METADATA)
print(SUMMARY_FILE)
print(REPORT)

print("\nScript 54 completed.")
