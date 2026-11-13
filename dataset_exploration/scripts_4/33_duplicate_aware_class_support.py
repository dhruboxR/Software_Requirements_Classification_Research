
from pathlib import Path
from collections import defaultdict
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"
VALIDATED = ROOT / "dataset_exploration/outputs/validated"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MAIN = DATA_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

PAIRS_FILE = TABLES / "30_near_duplicate_candidates.csv"
REVIEW_FILE = VALIDATED / "phase1d_69_pair_preliminary_review.csv"

TARGET_LABELS = {
    "D2": {"FR", "NFR"},
    "D3": {
        "NSR", "THE", "CON", "THO", "MAI",
        "IMM", "IND", "AVA", "AUD", "SUR", "INT"
    },
    "D4": {"FR", "NFR"},
}


def normalize(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


if not MAIN.exists() or not PAIRS_FILE.exists():
    raise FileNotFoundError(
        "Run the earlier dataset and near-duplicate audits first."
    )

if not REVIEW_FILE.exists():
    raise FileNotFoundError(
        f"Missing preliminary review file:\n{REVIEW_FILE}"
    )

df = pd.read_csv(MAIN, dtype=str, keep_default_na=False)
pairs = pd.read_csv(PAIRS_FILE, keep_default_na=False)
review = pd.read_csv(REVIEW_FILE, keep_default_na=False)

required_review = {
    "source", "text_a", "text_b"
}
if not required_review.issubset(review.columns):
    raise ValueError("Review CSV is missing required columns.")

df["source"] = df["Datasets"].str.strip()
df["label"] = df["Class"].str.strip().str.upper()
df["key"] = df[
    "Requirement Text (Original Requirement)"
].map(normalize)

print("=" * 78)
print("SCRIPT 33: DUPLICATE-AWARE CLASS SUPPORT")
print("=" * 78)

all_class_rows = []
all_group_rows = []
source_rows = []

for source, allowed in TARGET_LABELS.items():

    # Keep all labels during grouping. Non-target
    # labels may identify ambiguous target texts.
    source_df = df[
        (df["source"] == source) & (df["key"] != "")
    ].copy()

    grouped = (
        source_df.groupby("key")
        .agg(
            labels=("label", lambda s: set(s)),
            raw_rows=("label", "size"),
        )
    )

    keys = sorted(grouped.index.tolist())
    key_to_index = {key: i for i, key in enumerate(keys)}
    uf = UnionFind(len(keys))

    source_pairs = pairs[pairs["source"] == source]

    edges_used = 0

    # Group every detected high-similarity pair,
    # including same-label pairs.
    for _, row in source_pairs.iterrows():
        a = normalize(row["text_a"])
        b = normalize(row["text_b"])

        if a in key_to_index and b in key_to_index:
            uf.union(key_to_index[a], key_to_index[b])
            edges_used += 1

    # Flag all endpoints involved in the preliminary
    # disagreement review. They remain pending.
    flagged_keys = set()

    for _, row in review[review["source"] == source].iterrows():
        flagged_keys.add(normalize(row["text_a"]))
        flagged_keys.add(normalize(row["text_b"]))

    components = defaultdict(list)

    for key in keys:
        component = uf.find(key_to_index[key])
        components[component].append(key)

    component_details = []

    for component_id, component_keys in components.items():
        label_union = set()
        rows = 0

        for key in component_keys:
            label_union.update(grouped.at[key, "labels"])
            rows += int(grouped.at[key, "raw_rows"])

        target_labels = label_union & allowed
        other_labels = label_union - allowed

        has_review_flag = any(
            key in flagged_keys for key in component_keys
        )

        # This is conservative: multiple target labels,
        # unexpected labels, or pending review flags
        # prevent the group being counted as clean.
        clean = (
            len(target_labels) == 1
            and len(other_labels) == 0
            and not has_review_flag
        )

        detail = {
            "source": source,
            "component_id": f"{source}_G{component_id:05d}",
            "unique_texts": len(component_keys),
            "raw_rows": rows,
            "target_labels": "|".join(sorted(target_labels)),
            "other_labels": "|".join(sorted(other_labels)),
            "pending_review": has_review_flag,
            "provisionally_clean": clean,
        }

        component_details.append(detail)
        all_group_rows.append(detail)

    group_table = pd.DataFrame(component_details)

    source_rows.append({
        "source": source,
        "raw_rows": len(source_df),
        "unique_texts": len(keys),
        "similarity_edges_used": edges_used,
        "candidate_components": len(group_table),
        "pending_review_components": int(
            group_table["pending_review"].sum()
        ),
        "provisionally_clean_components": int(
            group_table["provisionally_clean"].sum()
        ),
    })

    for label in sorted(allowed):
        raw_rows = int(
            (source_df["label"] == label).sum()
        )

        class_keys = {
            key for key in keys
            if label in grouped.at[key, "labels"]
        }

        class_components = group_table[
            group_table["target_labels"].map(
                lambda x: label in str(x).split("|")
            )
        ]

        clean_components = class_components[
            class_components["provisionally_clean"]
        ]

        all_class_rows.append({
            "source": source,
            "label": label,
            "raw_rows": raw_rows,
            "unique_texts_with_label": len(class_keys),
            "candidate_components_with_label": len(
                class_components
            ),
            "pending_review_components": int(
                class_components["pending_review"].sum()
            ),
            "provisionally_clean_components": len(
                clean_components
            ),
            "provisionally_clean_unique_texts": int(
                clean_components["unique_texts"].sum()
            ),
            "theoretical_20pct_clean_component_count": (
                round(len(clean_components) * 0.20, 1)
            ),
        })

    print(
        f"\n{source}: {len(source_df)} rows, "
        f"{len(keys)} unique texts, "
        f"{len(group_table)} candidate components"
    )

class_summary = pd.DataFrame(all_class_rows)
source_summary = pd.DataFrame(source_rows)
group_summary = pd.DataFrame(all_group_rows)

class_summary.to_csv(
    TABLES / "33_duplicate_aware_class_support.csv",
    index=False,
    encoding="utf-8-sig"
)

source_summary.to_csv(
    TABLES / "33_duplicate_aware_source_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

group_summary.to_csv(
    TABLES / "33_candidate_group_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 78)
print("CLASS SUPPORT")
print("=" * 78)

print(class_summary.to_string(index=False))

print("\nSOURCE SUMMARY")
print(source_summary.to_string(index=False))

report_lines = [
    "PHASE 1E: DUPLICATE-AWARE CLASS SUPPORT",
    "=" * 70,
    "",
    "SOURCE SUMMARY",
    source_summary.to_string(index=False),
    "",
    "CLASS SUMMARY",
    class_summary.to_string(index=False),
    "",
    "LIMITATIONS",
    "- Groups are candidate text-similarity components.",
    "- TF-IDF similarity does not prove semantic identity.",
    "- Similarity search may have missed other pairs.",
    "- Connected similarity components may combine distinct meanings.",
    "- A preliminary review flag is not a confirmed label error.",
    "- Provisional clean counts are not verified sample counts.",
    "- The 20% figure is arithmetic, not a simulated test split.",
    "- Project-level independence is not established.",
    "- Original data and review decisions remain unchanged.",
]

report_path = REPORTS / "33_duplicate_aware_support.txt"
report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nSaved Script 33 tables and report.")
print("No training or testing split was created.")
