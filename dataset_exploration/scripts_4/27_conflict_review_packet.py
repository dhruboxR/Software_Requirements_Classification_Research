
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "dataset_exploration/outputs/tables"

INPUT = TABLES / "26_conflict_records_for_manual_review.csv"
OUTPUT = TABLES / "27_conflict_review_packet.csv"

TEXT_COL = "Requirement Text (Original Requirement)"

df = pd.read_csv(
    INPUT,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

required = {
    "source",
    "normalized_text",
    "label",
    "reference_labels",
    "reference_match",
    TEXT_COL,
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

records = []

for (source, key), group in df.groupby(
    ["source", "normalized_text"], sort=True
):
    labels = "|".join(sorted(set(group["label"])))

    originals = sorted(
        x for x in set(group["reference_labels"]) if x
    )

    references = "|".join(originals)

    records.append({
        "source": source,
        "requirement_text": group[TEXT_COL].iloc[0],
        "combined_labels": labels,
        "original_reference_labels": references,
        "reference_status": (
            group["reference_match"].iloc[0]
        ),
        "affected_rows": len(group),
        "review_priority": (
            "HIGH"
            if source in ("D2", "D3", "D4")
            else "NORMAL"
        ),
        "manual_decision": "PENDING",
        "reviewer_notes": "",
    })

review = pd.DataFrame(records)

review = review.sort_values(
    ["source", "affected_rows"],
    ascending=[True, False],
)

review.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("SCRIPT 27: CONFLICT REVIEW PACKET")
print("=" * 80)

print("\nTotal groups:", len(review))

print("\nGROUPS BY SOURCE")
print(review["source"].value_counts().sort_index())

print("\nD3 CONFLICTING REQUIREMENTS")
print("-" * 80)

for _, row in review[review["source"] == "D3"].iterrows():
    print("\nRequirement:")
    print(row["requirement_text"])
    print("Unified labels:", row["combined_labels"])
    print("Original labels:", row["original_reference_labels"])

print("\nD4 CONFLICT SUMMARY")
print("-" * 80)

d4 = review[review["source"] == "D4"]

print(
    d4[
        ["combined_labels", "affected_rows", "reference_status"]
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT)
print("All review decisions remain PENDING.")
