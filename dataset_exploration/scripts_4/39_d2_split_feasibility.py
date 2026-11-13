
from pathlib import Path
from collections import defaultdict
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

DATA = (
    ROOT / "A Unified Multi-Source Dataset of Software Require"
    / "Software Requirement Dataset (SR vs NSR) - After the Preprocessing.csv"
)
PAIRS = TABLES / "30_near_duplicate_candidates.csv"
REVIEW = BASE / "validated/phase1d_69_pair_preliminary_review.csv"

TRIALS = 300
CLASSES = ["FR", "NFR"]


def norm(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def make_groups(data, similarity_pairs, include_projects):
    uf = UnionFind(len(data))

    key_indices = defaultdict(list)
    project_first = {}

    for i, row in data.iterrows():
        key_indices[row["key"]].append(i)

        if include_projects:
            project = row["project_name"]
            if project:
                if project in project_first:
                    uf.union(i, project_first[project])
                else:
                    project_first[project] = i

    for indices in key_indices.values():
        for i in indices[1:]:
            uf.union(indices[0], i)

    used_edges = 0
    for _, pair in similarity_pairs.iterrows():
        a = key_indices.get(norm(pair["text_a"]), [])
        b = key_indices.get(norm(pair["text_b"]), [])

        if a and b:
            uf.union(a[0], b[0])
            used_edges += 1

    groups = np.array(
        [uf.find(i) for i in range(len(data))]
    )
    return groups, used_edges


df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
pairs = pd.read_csv(PAIRS, keep_default_na=False)
review = pd.read_csv(REVIEW, keep_default_na=False)

d2 = df[df["Datasets"].str.strip() == "D2"].copy()
d2["key"] = d2[
    "Requirement Text (Original Requirement)"
].map(norm)
d2["label"] = d2["Class"].str.strip().str.upper()
d2["project_name"] = d2["Project Name"].map(norm)

# Exclude texts with multiple original unified labels.
label_counts = d2.groupby("key")["label"].nunique()
conflicting_keys = set(
    label_counts[label_counts > 1].index
)

# Quarantine all texts in the preliminary
# disagreement review until adjudication.
flagged = set()
for _, row in review[review["source"] == "D2"].iterrows():
    flagged.add(norm(row["text_a"]))
    flagged.add(norm(row["text_b"]))

eligible = d2[
    d2["label"].isin(CLASSES)
    & d2["key"].ne("")
    & ~d2["key"].isin(conflicting_keys)
    & ~d2["key"].isin(flagged)
].copy()

# Keep one row per normalized text.
# Identical copies cannot count as independent examples.
eligible = (
    eligible.drop_duplicates("key")
    .reset_index(drop=True)
)

d2_pairs = pairs[pairs["source"] == "D2"]

print("=" * 78)
print("SCRIPT 39: D2 SPLIT FEASIBILITY")
print("=" * 78)

print("\nELIGIBLE DATA")
print("Original D2 rows:", len(d2))
print("Eligible unique texts:", len(eligible))
print("FR:", int((eligible["label"] == "FR").sum()))
print("NFR:", int((eligible["label"] == "NFR").sum()))
print("Conflicting-text groups:", len(conflicting_keys))
print("Review-flagged text keys:", len(flagged))

all_results = []
scenario_summaries = []

for scenario, include_projects in [
    ("TEXT_SIMILARITY", False),
    ("PROJECT_NAME_PLUS_SIMILARITY", True)
]:
    groups, edges = make_groups(
        eligible, d2_pairs, include_projects
    )

    component_sizes = pd.Series(groups).value_counts()

    print(f"\nSCENARIO: {scenario}")
    print("Connected groups:", len(component_sizes))
    print("Similarity edges used:", edges)
    print("Largest group:", int(component_sizes.max()))
    print(
        "Largest group percentage:",
        round(
            100 * component_sizes.max() / len(eligible),
            2
        )
    )

    scenario_summaries.append({
        "scenario": scenario,
        "eligible_texts": len(eligible),
        "connected_groups": len(component_sizes),
        "largest_group_rows": int(component_sizes.max()),
        "largest_group_pct": round(
            100 * component_sizes.max() / len(eligible), 2
        ),
        "similarity_edges_used": edges,
    })

    indices = np.arange(len(eligible))
    successful = 0

    for seed in range(TRIALS):
        try:
            outer = GroupShuffleSplit(
                n_splits=1,
                test_size=0.15,
                random_state=seed
            )

            trainval_idx, test_idx = next(
                outer.split(indices, groups=groups)
            )

            inner = GroupShuffleSplit(
                n_splits=1,
                test_size=0.15 / 0.85,
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

        except ValueError:
            continue

        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        split_groups = {
            name: set(groups[idx])
            for name, idx in split_indices.items()
        }

        if (
            split_groups["train"] & split_groups["val"]
            or split_groups["train"] & split_groups["test"]
            or split_groups["val"] & split_groups["test"]
        ):
            raise RuntimeError("Group leakage detected")

        successful += 1

        record = {
            "scenario": scenario,
            "trial": seed,
        }

        for name, idx in split_indices.items():
            labels = eligible.iloc[idx]["label"]
            record[f"{name}_size"] = len(idx)

            for label in CLASSES:
                record[f"{name}_{label}"] = int(
                    (labels == label).sum()
                )

        all_results.append(record)

    print("Successful trials:", successful)

results = pd.DataFrame(all_results)
scenarios = pd.DataFrame(scenario_summaries)

if results.empty:
    raise RuntimeError("No successful splits generated")

summary_rows = []

for scenario, part in results.groupby("scenario"):
    for label in CLASSES:
        test = part[f"test_{label}"]
        val = part[f"val_{label}"]
        train = part[f"train_{label}"]

        summary_rows.append({
            "scenario": scenario,
            "label": label,
            "successful_trials": len(part),
            "train_median": train.median(),
            "val_median": val.median(),
            "test_min": test.min(),
            "test_median": test.median(),
            "test_max": test.max(),
            "trials_test_below_20": int((test < 20).sum()),
            "trials_test_below_50": int((test < 50).sum()),
            "trials_train_class_missing": int(
                (train == 0).sum()
            ),
        })

summary = pd.DataFrame(summary_rows)

results.to_csv(
    TABLES / "39_d2_split_trial_counts.csv",
    index=False
)
scenarios.to_csv(
    TABLES / "39_d2_group_structure.csv",
    index=False
)
summary.to_csv(
    TABLES / "39_d2_split_feasibility_summary.csv",
    index=False
)

print("\n" + "=" * 78)
print("GROUP STRUCTURE")
print(scenarios.to_string(index=False))

print("\nCLASS SUPPORT BY SCENARIO")
print(summary.to_string(index=False))

lines = [
    "PHASE 1E: D2 SPLIT FEASIBILITY",
    "=" * 65,
    "",
    "GROUP STRUCTURE",
    scenarios.to_string(index=False),
    "",
    "CLASS SUPPORT",
    summary.to_string(index=False),
    "",
    "Limitations:",
    "- These are simulation trials, not approved splits.",
    "- Project names are not verified project identities.",
    "- GroupShuffleSplit does not stratify labels.",
    "- Large connected components can distort split ratios.",
    "- Candidate similarity grouping can miss paraphrases.",
    "- Flagged annotations remain unresolved.",
    "- No model training was performed.",
]

report = REPORTS / "39_d2_split_feasibility.txt"
report.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("\nSaved:", report)
print("Script 39 completed.")
