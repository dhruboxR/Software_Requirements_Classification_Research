
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd

# ============================================================
# SCRIPT 12: MANUAL ERROR REVIEW PREPARATION
#
# Uses Script 11 results.
# Prioritizes consensus classification errors.
#
# No training.
# No label modification.
# No test access.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

SOURCE = TABLES / "11_persistent_error_review.csv"

OUTPUT = TABLES / "12_priority_error_review.csv"
SUMMARY = REPORTS / "12_error_review_summary.txt"
METADATA = REPORTS / "12_error_review_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"

df = pd.read_csv(
    SOURCE,
    dtype={"requirement_id": str},
    keep_default_na=False,
)

if df["requirement_id"].duplicated().any():
    raise RuntimeError("Duplicate requirement IDs")

if "error_category" not in df:
    raise RuntimeError("Missing error categories")

if TEXT_COL not in df:
    raise RuntimeError("Missing requirement text")

df["architectures_always_wrong"] = pd.to_numeric(
    df["architectures_always_wrong"],
    errors="raise",
)

df["architectures_with_seed_variation"] = pd.to_numeric(
    df["architectures_with_seed_variation"],
    errors="raise",
)

# The first stage focuses on requirements
# that at least two architectures always miss.

priority = df.loc[
    df["architectures_always_wrong"] >= 2
].copy()

priority = priority.sort_values(
    [
        "architectures_always_wrong",
        "architectures_with_seed_variation",
        "requirement_id",
    ],
    ascending=[False, False, True],
).reset_index(drop=True)

priority.insert(
    0, "review_number", range(1, len(priority) + 1)
)

priority["reviewer_assessment"] = ""
priority["possible_issue"] = ""
priority["reviewer_notes"] = ""
priority["review_status"] = "NOT_REVIEWED"

priority["suggested_review_priority"] = (
    priority["architectures_always_wrong"].map({
        3: "P1_ALL_NINE_WRONG",
        2: "P2_TWO_ARCHITECTURES_WRONG",
    })
)

priority.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

counts = (
    priority.groupby(
        ["suggested_review_priority", "true_label"]
    )
    .size()
    .reset_index(name="count")
)

all_nine = int(
    (priority["architectures_always_wrong"] == 3).sum()
)

two_architectures = int(
    (priority["architectures_always_wrong"] == 2).sum()
)

print("=" * 76)
print("SCRIPT 12: PRIORITY ERROR REVIEW PREPARATION")
print("=" * 76)

print("\nREVIEW COUNTS")
print("All nine runs wrong:", all_nine)
print("Two architectures consistently wrong:", two_architectures)
print("Total priority requirements:", len(priority))

print("\nPRIORITY BY CLASS")
print(counts.to_string(index=False))

print("\nFIRST FIVE REVIEW ITEMS")
for _, row in priority.head(5).iterrows():
    print(
        f"\nReview #{row['review_number']} | "
        f"ID: {row['requirement_id']}"
    )
    print("Original label:", row["true_label"])
    print("Requirement:", row[TEXT_COL])
    print(
        "Architectures always wrong:",
        row["architectures_always_wrong"]
    )

instructions = [
    "SCRIPT 12: MANUAL ERROR REVIEW",
    "=" * 76,
    "",
    "Review these requirements independently.",
    "",
    "Suggested review order:",
    "1. P1: All nine transformer runs wrong.",
    "2. P2: Two architectures consistently wrong.",
    "",
    "Reviewer assessment options:",
    "- LABEL_SUPPORTED",
    "- POSSIBLE_LABEL_AMBIGUITY",
    "- POSSIBLE_LABEL_ERROR",
    "- INSUFFICIENT_CONTEXT",
    "",
    "Possible issue categories:",
    "- IMPLICIT_NFR",
    "- MIXED_FR_NFR",
    "- DOMAIN_TERMINOLOGY",
    "- AMBIGUOUS_WORDING",
    "- INSUFFICIENT_CONTEXT",
    "- OTHER",
    "",
    "The categories are proposed review codes,",
    "not automatically assigned explanations.",
    "",
    "Review the text and original label first.",
    "Inspect model predictions afterward when possible.",
    "Record supporting reasoning in reviewer_notes.",
    "",
    "Do not modify the frozen manifest.",
    "Do not use model disagreement as proof",
    "that an original label is incorrect.",
    "",
    "Review conclusions are exploratory.",
    "Do not tune models using these reviewed cases.",
    "The test partition remains sealed.",
    "",
    "COUNTS",
    f"All nine wrong: {all_nine}",
    f"Two architectures wrong: {two_architectures}",
    f"Priority review total: {len(priority)}",
    "",
    counts.to_string(index=False),
]

SUMMARY.write_text(
    "\n".join(instructions) + "\n",
    encoding="utf-8",
)

METADATA.write_text(
    json.dumps({
        "experiment": "DL-12",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": SOURCE.name,
        "priority_requirements": len(priority),
        "all_nine_wrong": all_nine,
        "two_architectures_wrong": two_architectures,
        "manual_review_complete": False,
        "labels_modified": False,
        "training_performed": False,
        "test_accessed": False,
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Priority review worksheet: CREATED")
print("Manual review: PENDING")
print("Original labels modified: NO")
print("Model training: NO")
print("Test set: SEALED")

print("\nFILES GENERATED")
print(OUTPUT)
print(SUMMARY)
print(METADATA)

print("\nScript 12 completed.")
