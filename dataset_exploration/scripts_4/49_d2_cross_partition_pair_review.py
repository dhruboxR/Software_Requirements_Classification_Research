
from pathlib import Path
import pandas as pd

# =====================================================
# SCRIPT 49: CROSS-PARTITION PAIR REVIEW
# Read-only: no changes to the frozen dataset.
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
VALIDATED = BASE / "validated"
REPORTS = BASE / "reports"

MANIFEST = VALIDATED / "44_d2_frozen_split_manifest.csv"
PAIRS = TABLES / "48_d2_cross_partition_review_candidates.csv"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    MANIFEST, dtype=str, keep_default_na=False,
    encoding="utf-8-sig"
)

pairs = pd.read_csv(
    PAIRS, dtype=str, keep_default_na=False,
    encoding="utf-8-sig"
)

required_manifest = {
    "Requirement ID",
    "Requirement Text (Original Requirement)",
    "label",
    "split",
    "similarity_group",
}

required_pairs = {
    "requirement_id_a",
    "requirement_id_b",
    "similarity",
    "comparison",
}

if not required_manifest.issubset(df.columns):
    raise ValueError(
        f"Missing manifest columns: "
        f"{required_manifest - set(df.columns)}"
    )

if not required_pairs.issubset(pairs.columns):
    raise ValueError(
        f"Missing pair columns: "
        f"{required_pairs - set(pairs.columns)}"
    )

if df["Requirement ID"].duplicated().any():
    raise ValueError("Duplicate requirement IDs in manifest")

lookup = df.set_index("Requirement ID")

records = []

for i, row in pairs.iterrows():
    id_a = row["requirement_id_a"]
    id_b = row["requirement_id_b"]

    if id_a not in lookup.index:
        raise ValueError(f"Missing requirement: {id_a}")

    if id_b not in lookup.index:
        raise ValueError(f"Missing requirement: {id_b}")

    a = lookup.loc[id_a]
    b = lookup.loc[id_b]

    if a["split"] == b["split"]:
        raise RuntimeError(
            f"Pair {i+1} is not cross-partition"
        )

    records.append({
        "pair_id": f"CP_{i+1:03d}",
        "comparison": row["comparison"],
        "similarity": float(row["similarity"]),
        "requirement_id_a": id_a,
        "requirement_id_b": id_b,
        "split_a": a["split"],
        "split_b": b["split"],
        "label_a": a["label"],
        "label_b": b["label"],
        "group_a": a["similarity_group"],
        "group_b": b["similarity_group"],
        "text_a": a[
            "Requirement Text (Original Requirement)"
        ],
        "text_b": b[
            "Requirement Text (Original Requirement)"
        ],
        "same_label": a["label"] == b["label"],
        "review_decision": "PENDING",
        "reviewer": "",
        "review_notes": "",
    })

review = pd.DataFrame(records)

review = review.sort_values(
    "similarity", ascending=False
).reset_index(drop=True)

output_csv = (
    TABLES / "49_d2_cross_partition_semantic_review.csv"
)

review.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 76)
print("SCRIPT 49: CROSS-PARTITION SEMANTIC REVIEW")
print("=" * 76)

print("\nREVIEW SUMMARY")
print("Pairs requiring review:", len(review))
print(
    "Same-label pairs:",
    int(review["same_label"].sum())
)
print(
    "Different-label pairs:",
    int((~review["same_label"]).sum())
)
print(
    "Pairs involving test:",
    int(
        (
            (review["split_a"] == "test")
            | (review["split_b"] == "test")
        ).sum()
    )
)

for _, row in review.iterrows():
    print("\n" + "=" * 76)
    print("PAIR:", row["pair_id"])
    print("Similarity:", row["similarity"])
    print("Comparison:", row["comparison"])

    print("\nREQUIREMENT A")
    print("ID:", row["requirement_id_a"])
    print("Partition:", row["split_a"])
    print("Label:", row["label_a"])
    print("Text:", row["text_a"])

    print("\nREQUIREMENT B")
    print("ID:", row["requirement_id_b"])
    print("Partition:", row["split_b"])
    print("Label:", row["label_b"])
    print("Text:", row["text_b"])

    print("\nReview decision: PENDING")

report_path = (
    REPORTS / "49_d2_cross_partition_review_status.txt"
)

report_path.write_text(
    "D2 CROSS-PARTITION REVIEW\n"
    + "=" * 65 + "\n"
    + f"Pairs: {len(review)}\n"
    + f"Same-label pairs: {int(review['same_label'].sum())}\n"
    + f"Different-label pairs: "
      f"{int((~review['same_label']).sum())}\n"
    + "Semantic decisions: PENDING\n"
    + "Frozen split modified: NO\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(output_csv)
print(report_path)

print("\nScript 49 completed.")
