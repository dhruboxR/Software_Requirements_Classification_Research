
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
VALIDATED = BASE / "validated"
REPORTS = BASE / "reports"

CANDIDATES = VALIDATED / "42_d2_source_consistent_candidates.csv"
GROUPS = TABLES / "43_d2_candidate_group_membership.csv"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

for directory in (TABLES, VALIDATED, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


candidates = pd.read_csv(
    CANDIDATES, dtype=str, keep_default_na=False
)

membership = pd.read_csv(
    GROUPS, dtype=str, keep_default_na=False
)

required_candidate = {"key", "label", "Requirement ID"}
required_membership = {"key", "similarity_group"}

if not required_candidate.issubset(candidates.columns):
    raise ValueError("Candidate columns missing")

if not required_membership.issubset(membership.columns):
    raise ValueError("Group membership columns missing")

if candidates["key"].duplicated().any():
    raise ValueError("Duplicate candidate keys")

if membership["key"].duplicated().any():
    raise ValueError("Duplicate membership keys")

df = candidates.merge(
    membership[["key", "similarity_group"]],
    on="key",
    how="left",
    validate="one_to_one",
    indicator=True
)

if not (df["_merge"] == "both").all():
    raise ValueError("Some candidates have no similarity group")

df = df.drop(columns=["_merge"])

if len(df) != 4835:
    raise ValueError("Unexpected candidate count")

if set(df["label"]) != {"FR", "NFR"}:
    raise ValueError("Unexpected class labels")

df = df.sort_values("key").reset_index(drop=True)

indices = np.arange(len(df))
groups = df["similarity_group"].to_numpy()

outer = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=SEED
)

trainval_idx, test_idx = next(
    outer.split(indices, groups=groups)
)

inner = GroupShuffleSplit(
    n_splits=1,
    test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=SEED + 10000
)

train_local, val_local = next(
    inner.split(
        trainval_idx,
        groups=groups[trainval_idx]
    )
)

train_idx = trainval_idx[train_local]
val_idx = trainval_idx[val_local]

df["split"] = ""

df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "validation"
df.loc[test_idx, "split"] = "test"

# Verify group isolation.
group_splits = df.groupby("similarity_group")["split"].nunique()

if not (group_splits == 1).all():
    raise RuntimeError("Similarity group crossed partitions")

if (df["split"] == "").any():
    raise RuntimeError("Unassigned records")

summary = (
    df.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(["train", "validation", "test"])
)

summary["total"] = summary.sum(axis=1)
summary["fraction"] = summary["total"] / len(df)

if (
    (summary["FR"] == 0).any()
    or (summary["NFR"] == 0).any()
    or summary.loc["test", "NFR"] < 100
):
    raise RuntimeError("Split feasibility requirement failed")

for split_name, expected in (
    ("train", 0.70),
    ("validation", 0.15),
    ("test", 0.15)
):
    actual = float(summary.loc[split_name, "fraction"])
    if abs(actual - expected) > 0.05:
        raise RuntimeError(
            f"Unexpected {split_name} fraction: {actual}"
        )

print("=" * 75)
print("SCRIPT 44: FROZEN D2 EVALUATION PARTITIONS")
print("=" * 75)

print("\nCLASS DISTRIBUTION")
print(summary.to_string())

print("\nLEAKAGE CHECKS")
print("Total records:", len(df))
print("Unique texts:", df["key"].nunique())
print("Similarity groups:", df["similarity_group"].nunique())
print("Groups crossing partitions:", int((group_splits > 1).sum()))

manifest = VALIDATED / "44_d2_frozen_split_manifest.csv"
df.to_csv(manifest, index=False, encoding="utf-8-sig")

metadata = {
    "seed": SEED,
    "protocol": "GroupShuffleSplit",
    "grouping": "D2 character-ngram similarity components from Script 43",
    "intended_proportions": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15
    },
    "candidate_count": len(df),
    "candidate_sha256": file_sha256(CANDIDATES),
    "membership_sha256": file_sha256(GROUPS),
    "manifest_sha256": file_sha256(manifest),
    "class_counts": {
        split: {
            "FR": int(summary.loc[split, "FR"]),
            "NFR": int(summary.loc[split, "NFR"]),
            "total": int(summary.loc[split, "total"])
        }
        for split in summary.index
    },
    "status": "FROZEN_PENDING_FINAL_APPROVAL",
    "limitations": [
        "Original project independence not established",
        "Source label agreement does not establish semantic correctness",
        "Near-duplicate discovery may be incomplete",
        "Test set must not guide model selection"
    ]
}

metadata_path = VALIDATED / "44_d2_split_metadata.json"
metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nSaved:", manifest)
print("Saved:", metadata_path)
print("Status: FROZEN_PENDING_FINAL_APPROVAL")
