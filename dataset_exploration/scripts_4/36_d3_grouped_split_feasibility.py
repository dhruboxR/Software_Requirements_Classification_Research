
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "A Unified Multi-Source Dataset of Software Require"
BASE = ROOT / "dataset_exploration/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

MAIN = DATA / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)
MAPPING = TABLES / "21_d3_original_project_recovery.csv"
PAIRS = TABLES / "30_near_duplicate_candidates.csv"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

TRIALS = 300
TEST_FRACTION = 0.15
VAL_FRACTION_OF_REMAINING = 0.15 / 0.85


def norm(x):
    s = unicodedata.normalize("NFKC", str(x))
    return re.sub(r"\s+", " ", s).strip().casefold()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a):
        while a != self.parent[a]:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


df = pd.read_csv(MAIN, dtype=str, keep_default_na=False)
mapping = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
pairs = pd.read_csv(PAIRS, dtype=str, keep_default_na=False)

d3 = df[df["Datasets"].str.strip() == "D3"].copy()

if mapping["Requirement ID"].duplicated().any():
    raise ValueError("Duplicate requirement IDs in recovery mapping")

d3 = d3.merge(
    mapping[[
        "Requirement ID",
        "Original Project IDs",
        "Mapping Status"
    ]],
    on="Requirement ID",
    how="left",
    validate="one_to_one"
)

d3["key"] = d3[
    "Requirement Text (Original Requirement)"
].map(norm)
d3["label"] = d3["Class"].str.strip()

conflicts = set(
    d3.groupby("key")["label"]
    .nunique()
    .loc[lambda x: x > 1]
    .index
)

eligible = d3[
    d3["Mapping Status"].eq("unique_original_project")
    & d3["Original Project IDs"].notna()
    & d3["Original Project IDs"].ne("")
    & ~d3["Original Project IDs"].str.contains(
        "|", regex=False, na=False
    )
    & ~d3["key"].isin(conflicts)
    & d3["key"].ne("")
].copy().reset_index(drop=True)

n = len(eligible)
uf = UnionFind(n)

# 1. Connect all rows belonging to the same
# recovered original project.
project_first = {}

for i, project in enumerate(eligible["Original Project IDs"]):
    if project in project_first:
        uf.union(i, project_first[project])
    else:
        project_first[project] = i

# 2. Connect identical normalized texts.
text_first = {}
for i, key in enumerate(eligible["key"]):
    if key in text_first:
        uf.union(i, text_first[key])
    else:
        text_first[key] = i

# 3. Connect near-duplicate candidate pairs,
# including pairs with matching labels.
text_indices = {}
for i, key in enumerate(eligible["key"]):
    text_indices.setdefault(key, []).append(i)

edges = 0
for _, row in pairs[pairs["source"] == "D3"].iterrows():
    a = norm(row["text_a"])
    b = norm(row["text_b"])

    if a in text_indices and b in text_indices:
        uf.union(text_indices[a][0], text_indices[b][0])
        edges += 1

eligible["group"] = [uf.find(i) for i in range(n)]
labels = sorted(eligible["label"].unique())

components = (
    eligible.groupby("group")
    .agg(
        rows=("label", "size"),
        projects=("Original Project IDs", "nunique"),
        classes=("label", "nunique")
    )
    .reset_index()
)

print("=" * 78)
print("SCRIPT 36: D3 GROUPED SPLIT FEASIBILITY")
print("=" * 78)

print("\nGROUP STRUCTURE")
print("Eligible rows:", n)
print("Recovered projects:", eligible["Original Project IDs"].nunique())
print("Near-duplicate edges connected:", edges)
print("Connected groups:", len(components))
print("Largest group size:", int(components["rows"].max()))
print(
    "Largest group share:",
    round(100 * components["rows"].max() / n, 2),
    "%"
)

component_path = TABLES / "36_d3_component_summary.csv"
components.to_csv(component_path, index=False)

# Simulate candidate splits only. This is not
# a final or approved train/test partition.
results = []
arrays = np.arange(n)

for seed in range(TRIALS):
    try:
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_FRACTION,
            random_state=seed
        )

        trainval_idx, test_idx = next(
            outer.split(
                arrays,
                groups=eligible["group"]
            )
        )

        trainval = eligible.iloc[trainval_idx]

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=VAL_FRACTION_OF_REMAINING,
            random_state=seed + 10000
        )

        inner_train, inner_val = next(
            inner.split(
                trainval,
                groups=trainval["group"]
            )
        )

        train_idx = trainval_idx[inner_train]
        val_idx = trainval_idx[inner_val]

    except ValueError:
        continue

    partitions = {
        "train": eligible.iloc[train_idx],
        "val": eligible.iloc[val_idx],
        "test": eligible.iloc[test_idx]
    }

    groups_by_split = {
        name: set(frame["group"])
        for name, frame in partitions.items()
    }

    if any(
        groups_by_split[a] & groups_by_split[b]
        for a, b in [
            ("train", "val"),
            ("train", "test"),
            ("val", "test")
        ]
    ):
        raise RuntimeError("Group leakage detected")

    for label in labels:
        results.append({
            "trial": seed,
            "label": label,
            "train": int(
                (partitions["train"]["label"] == label).sum()
            ),
            "val": int(
                (partitions["val"]["label"] == label).sum()
            ),
            "test": int(
                (partitions["test"]["label"] == label).sum()
            )
        })

trials = pd.DataFrame(results)

if trials.empty:
    print("\nNO VALID GROUPED SPLITS GENERATED")
    raise SystemExit(1)

trial_path = TABLES / "36_d3_split_trial_class_counts.csv"
trials.to_csv(trial_path, index=False)

summary = (
    trials.groupby("label")
    .agg(
        train_min=("train", "min"),
        train_median=("train", "median"),
        val_min=("val", "min"),
        val_median=("val", "median"),
        test_min=("test", "min"),
        test_median=("test", "median"),
        test_max=("test", "max"),
        trials_test_below_10=(
            "test", lambda s: int((s < 10).sum())
        ),
        trials_test_below_20=(
            "test", lambda s: int((s < 20).sum())
        )
    )
    .reset_index()
)

summary_path = TABLES / "36_d3_split_feasibility_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSIMULATION RESULTS")
print("Successful simulations:", trials["trial"].nunique())
print("Requested approximate ratio: 70/15/15")
print("Actual proportions may vary with group sizes.")

print("\nPER-CLASS TEST SUPPORT")
print(summary.to_string(index=False))

complete = trials.groupby("trial").agg(
    all_classes_in_test=("test", lambda s: bool((s > 0).all())),
    all_classes_at_least_10=("test", lambda s: bool((s >= 10).all())),
    all_classes_at_least_20=("test", lambda s: bool((s >= 20).all()))
)

print("\nWHOLE-SPLIT FEASIBILITY")
print(
    "Trials with all classes represented in test:",
    int(complete["all_classes_in_test"].sum())
)
print(
    "Trials with >=10 test examples in every class:",
    int(complete["all_classes_at_least_10"].sum())
)
print(
    "Trials with >=20 test examples in every class:",
    int(complete["all_classes_at_least_20"].sum())
)

report = [
    "PHASE 1E: D3 LEAKAGE-CONTROLLED SPLIT FEASIBILITY",
    "=" * 65,
    f"Eligible records: {n}",
    f"Connected groups: {len(components)}",
    f"Largest group: {int(components['rows'].max())}",
    f"Successful simulations: {trials['trial'].nunique()}",
    "",
    summary.to_string(index=False),
    "",
    "Whole-split feasibility:",
    complete.sum().to_string(),
    "",
    "Limitations:",
    "- These are random feasibility simulations, not selected splits.",
    "- GroupShuffleSplit does not stratify class labels.",
    "- Text similarity edges are conservative candidate constraints.",
    "- Original project recovery remains provisional.",
    "- Near-duplicate discovery may be incomplete.",
    "- Manual annotation approval remains pending.",
    "- No model was trained or evaluated.",
]

report_path = REPORTS / "36_d3_grouped_split_feasibility.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", report_path)
print("Script 36 completed.")
