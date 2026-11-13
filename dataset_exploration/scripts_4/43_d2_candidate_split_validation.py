
from pathlib import Path
from collections import defaultdict
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration" / "outputs"

TABLES = BASE / "tables"
REPORTS = BASE / "reports"
VALIDATED = BASE / "validated"

CANDIDATES = (
    VALIDATED / "42_d2_source_consistent_candidates.csv"
)
PAIRS = TABLES / "30_near_duplicate_candidates.csv"

TRIALS = 300
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
TOLERANCE = 0.05

for folder in (TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def normalize(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


print("=" * 78)
print("SCRIPT 43: D2 CANDIDATE SPLIT VALIDATION")
print("=" * 78)

df = pd.read_csv(
    CANDIDATES,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

pairs = pd.read_csv(
    PAIRS,
    dtype=str,
    keep_default_na=False
)

required = {"key", "label", "Requirement ID"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Missing candidate columns: {required - set(df.columns)}"
    )

df["key"] = df["key"].map(normalize)
df["label"] = df["label"].str.strip().str.upper()

if df["key"].duplicated().any():
    raise ValueError("Candidate text keys are not unique")

if not df["label"].isin(["FR", "NFR"]).all():
    raise ValueError("Unexpected classification label")

if len(df) != 4835:
    raise ValueError(
        f"Expected 4835 candidates, found {len(df)}"
    )

df = df.sort_values("key").reset_index(drop=True)

# --------------------------------------------------
# 1. Build similarity components
# --------------------------------------------------

uf = UnionFind(len(df))
key_to_index = {
    key: i for i, key in enumerate(df["key"])
}

edges_used = 0

for _, pair in pairs[
    pairs["source"].str.strip() == "D2"
].iterrows():

    a = normalize(pair["text_a"])
    b = normalize(pair["text_b"])

    if a in key_to_index and b in key_to_index:
        uf.union(key_to_index[a], key_to_index[b])
        edges_used += 1

roots = [uf.find(i) for i in range(len(df))]
unique_roots = sorted(set(roots))
root_to_group = {
    root: index for index, root in enumerate(unique_roots)
}

df["similarity_group"] = [
    root_to_group[root] for root in roots
]

components = (
    df.groupby("similarity_group")
    .agg(
        rows=("label", "size"),
        distinct_labels=("label", "nunique"),
        fr=("label", lambda s: int((s == "FR").sum())),
        nfr=("label", lambda s: int((s == "NFR").sum())),
    )
    .reset_index()
)

# A group can contain both labels because the
# similarity detector is not a semantic adjudicator.
mixed_groups = components[
    components["distinct_labels"] > 1
].copy()

print("\nCANDIDATE GROUP STRUCTURE")
print("Candidate requirements:", len(df))
print("FR:", int((df["label"] == "FR").sum()))
print("NFR:", int((df["label"] == "NFR").sum()))
print("Similarity edges retained:", edges_used)
print("Connected groups:", len(components))
print("Largest group:", int(components["rows"].max()))
print(
    "Groups containing both labels:",
    len(mixed_groups)
)

if len(mixed_groups):
    mixed_ids = set(mixed_groups["similarity_group"])
    print(
        "Rows in mixed-label groups:",
        int(df["similarity_group"].isin(mixed_ids).sum())
    )

# --------------------------------------------------
# 2. Simulate 300 grouped splits
# --------------------------------------------------

indices = np.arange(len(df))
groups = df["similarity_group"].to_numpy()

records = []

for seed in range(TRIALS):
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=seed
    )

    trainval_idx, test_idx = next(
        outer.split(indices, groups=groups)
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=seed + 10000
    )

    train_local, val_local = next(
        inner.split(
            trainval_idx,
            groups=groups[trainval_idx]
        )
    )

    train_idx = trainval_idx[train_local]
    val_idx = trainval_idx[val_local]

    partitions = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    group_sets = {
        name: set(groups[idx])
        for name, idx in partitions.items()
    }

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ]:
        if group_sets[a] & group_sets[b]:
            raise RuntimeError(
                f"Group overlap: {a} and {b}"
            )

    record = {"trial": seed}

    for name, idx in partitions.items():
        labels = df.iloc[idx]["label"]

        record[f"{name}_rows"] = len(idx)
        record[f"{name}_FR"] = int(
            (labels == "FR").sum()
        )
        record[f"{name}_NFR"] = int(
            (labels == "NFR").sum()
        )
        record[f"{name}_fraction"] = (
            len(idx) / len(df)
        )

    record["ratio_ok"] = all(
        abs(record[f"{name}_fraction"] - target)
        <= TOLERANCE
        for name, target in [
            ("train", 0.70),
            ("val", 0.15),
            ("test", 0.15)
        ]
    )

    record["classes_present"] = all(
        record[f"{name}_{label}"] > 0
        for name in ("train", "val", "test")
        for label in ("FR", "NFR")
    )

    record["nfr_test_100"] = (
        record["test_NFR"] >= 100
    )

    record["pass"] = (
        record["ratio_ok"]
        and record["classes_present"]
        and record["nfr_test_100"]
    )

    records.append(record)

trials = pd.DataFrame(records)

# --------------------------------------------------
# 3. Save reproducible evidence
# --------------------------------------------------

df[
    [
        "Requirement ID",
        "key",
        "label",
        "similarity_group"
    ]
].to_csv(
    TABLES / "43_d2_candidate_group_membership.csv",
    index=False,
    encoding="utf-8-sig"
)

components.to_csv(
    TABLES / "43_d2_component_summary.csv",
    index=False
)

trials.to_csv(
    TABLES / "43_d2_split_simulations.csv",
    index=False
)

summary = {
    "candidate_rows": len(df),
    "similarity_groups": len(components),
    "largest_group": int(components["rows"].max()),
    "mixed_label_groups": len(mixed_groups),
    "simulations": len(trials),
    "ratio_pass": int(trials["ratio_ok"].sum()),
    "all_classes_present": int(
        trials["classes_present"].sum()
    ),
    "nfr_test_at_least_100": int(
        trials["nfr_test_100"].sum()
    ),
    "overall_pass": int(trials["pass"].sum()),
    "train_fraction_median": round(
        trials["train_fraction"].median(), 4
    ),
    "val_fraction_median": round(
        trials["val_fraction"].median(), 4
    ),
    "test_fraction_median": round(
        trials["test_fraction"].median(), 4
    ),
    "test_nfr_min": int(trials["test_NFR"].min()),
    "test_nfr_median": float(
        trials["test_NFR"].median()
    ),
    "test_nfr_max": int(trials["test_NFR"].max()),
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    TABLES / "43_d2_candidate_split_summary.csv",
    index=False
)

print("\nSPLIT FEASIBILITY SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

report = [
    "PHASE 1E: D2 FINAL CANDIDATE FEASIBILITY",
    "=" * 65,
    "",
    *[f"{key}: {value}" for key, value in summary.items()],
    "",
    "Limitations:",
    "- This is a feasibility simulation, not an approved split.",
    "- Groups rely on detected text similarity only.",
    "- Semantically similar requirements may remain undetected.",
    "- Project-independent generalization is not established.",
    "- Source agreement does not guarantee correct labels.",
    "- Mixed-label similarity groups require further review.",
    "- No model was trained.",
]

report_path = (
    REPORTS / "43_d2_candidate_split_validation.txt"
)

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", report_path)
print("Script 43 completed.")
