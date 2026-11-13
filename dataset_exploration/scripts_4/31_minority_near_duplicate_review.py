
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

INPUT = TABLES / "30_near_duplicate_candidates.csv"
OUTPUT = TABLES / "31_priority_near_duplicate_review.csv"
SUMMARY = TABLES / "31_priority_review_summary.csv"

df = pd.read_csv(INPUT, keep_default_na=False)

required = {
    "source", "similarity",
    "text_a", "text_b",
    "labels_a", "labels_b",
    "disjoint_labels"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Classes requiring particular attention because
# of their limited independent support.
MINORITY = {
    "D2": {"NFR"},
    "D3": {
        "THE", "CON", "THO", "MAI", "IMM",
        "IND", "AVA", "AUD", "SUR", "INT"
    },
    "D4": {"NFR"},
}

def label_set(value):
    return set(str(value).split("|")) - {""}

def is_true(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }

df["minority_involved"] = df.apply(
    lambda row: bool(
        (label_set(row["labels_a"])
         | label_set(row["labels_b"]))
        & MINORITY.get(row["source"], set())
    ),
    axis=1
)

df["disjoint"] = df["disjoint_labels"].map(is_true)
df["similarity"] = pd.to_numeric(
    df["similarity"], errors="coerce"
)

# Prioritize pairs that contain minority-class labels,
# have disjoint label sets, and are highly similar.
df["priority_score"] = (
    df["minority_involved"].astype(int) * 100
    + df["disjoint"].astype(int) * 50
    + df["similarity"].fillna(0) * 10
)

# The ranking is only for manual review.
# It is not evidence that either label is wrong.
priority = df[
    df["minority_involved"] & df["disjoint"]
].copy()

priority = priority.sort_values(
    ["source", "priority_score"],
    ascending=[True, False]
)

priority["review_decision"] = "PENDING"
priority["review_notes"] = ""

priority.to_csv(
    OUTPUT, index=False, encoding="utf-8-sig"
)

summary_rows = []

for source in ["D2", "D3", "D4"]:
    subset = df[df["source"] == source]
    selected = priority[priority["source"] == source]

    summary_rows.append({
        "source": source,
        "all_candidate_pairs": len(subset),
        "disjoint_label_pairs": int(
            subset["disjoint"].sum()
        ),
        "minority_disjoint_pairs": len(selected),
        "highest_similarity": (
            round(selected["similarity"].max(), 5)
            if len(selected) else None
        ),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    SUMMARY, index=False, encoding="utf-8-sig"
)

print("=" * 78)
print("SCRIPT 31: MINORITY NEAR-DUPLICATE REVIEW")
print("=" * 78)

print("\nSUMMARY")
print(summary.to_string(index=False))

for source in ["D2", "D3", "D4"]:
    subset = priority[priority["source"] == source]

    print(f"\n{source}: TOP 5 PRIORITY PAIRS")

    for _, row in subset.head(5).iterrows():
        print("-" * 65)
        print("Similarity:", row["similarity"])
        print("A label:", row["labels_a"])
        print("A:", row["text_a"][:250])
        print("B label:", row["labels_b"])
        print("B:", row["text_b"][:250])

report = [
    "MINORITY NEAR-DUPLICATE REVIEW",
    "=" * 65,
    "",
    summary.to_string(index=False),
    "",
    "Important limitations:",
    "- Review priority is a heuristic, not a quality score.",
    "- Similar texts may express different requirements.",
    "- Negation and numeric differences may change meaning.",
    "- No automated relabeling or removal was performed.",
    "- All review decisions are pending.",
]

(REPORTS / "31_minority_review_summary.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", OUTPUT)
print("Saved:", SUMMARY)
print("Script 31 completed.")
