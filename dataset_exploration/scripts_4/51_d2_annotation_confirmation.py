
from pathlib import Path
import pandas as pd

# ============================================================
# SCRIPT 51: D2 ANNOTATION CONFIRMATION WORKSHEET
# Creates a review worksheet. Does not modify the dataset.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
REPORTS = BASE / "reports"

PAIR_FILE = (
    TABLES / "49_d2_cross_partition_semantic_review.csv"
)

PLAN_FILE = (
    TABLES / "50_d2_provisional_review_decisions.csv"
)

OUTPUT = (
    TABLES / "51_d2_human_annotation_review.csv"
)

REPORT = (
    REPORTS / "51_d2_annotation_review_status.txt"
)

for path in [PAIR_FILE, PLAN_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

pairs = pd.read_csv(
    PAIR_FILE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

plan = pd.read_csv(
    PLAN_FILE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

if pairs["pair_id"].duplicated().any():
    raise ValueError("Duplicate pair IDs in Script 49")

if plan["pair_id"].duplicated().any():
    raise ValueError("Duplicate pair IDs in Script 50")

if set(pairs["pair_id"]) != set(plan["pair_id"]):
    raise ValueError("Pair IDs do not match")

review = pairs.merge(
    plan[
        [
            "pair_id",
            "assessment",
            "label_issue",
            "proposed_action",
            "reason"
        ]
    ],
    on="pair_id",
    how="inner",
    validate="one_to_one"
)

# Reviewer must explicitly complete these fields.
review["reviewer_name"] = ""
review["review_date"] = ""
review["semantic_decision"] = "PENDING"
review["label_decision"] = "PENDING"
review["approved_action"] = "PENDING"
review["reviewer_rationale"] = ""
review["review_status"] = "PENDING"

# Preserve an existing worksheet rather than overwriting
# completed reviewer decisions.
if OUTPUT.exists():
    raise FileExistsError(
        "Review worksheet already exists. "
        "Existing reviewer entries were preserved: "
        f"{OUTPUT}"
    )

review.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 76)
print("SCRIPT 51: D2 HUMAN ANNOTATION REVIEW")
print("=" * 76)

print("\nREVIEW WORKSHEET")
print("Pairs:", len(review))
print(
    "Requirements involved:",
    len(
        set(review["requirement_id_a"])
        | set(review["requirement_id_b"])
    )
)

print("\nPROPOSED DECISIONS")
print(
    review[
        [
            "pair_id",
            "assessment",
            "label_issue",
            "proposed_action",
            "review_status"
        ]
    ].to_string(index=False)
)

print("\nREQUIRED REVIEW FIELDS")
print("reviewer_name")
print("review_date")
print("semantic_decision")
print("label_decision")
print("approved_action")
print("reviewer_rationale")
print("review_status")

print("\nALLOWED SEMANTIC DECISIONS")
print("EQUIVALENT")
print("RELATED_BUT_DISTINCT")
print("DISTINCT")
print("UNCERTAIN")

print("\nALLOWED LABEL DECISIONS")
print("BOTH_FR")
print("BOTH_NFR")
print("ORIGINAL_LABELS_ACCEPTABLE")
print("LABEL_UNCERTAIN")

print("\nALLOWED ACTIONS")
print("QUARANTINE_BOTH")
print("CONNECT_SIMILARITY_GROUP")
print("KEEP_SEPARATE")
print("ESCALATE_REVIEW")

print("\nALLOWED REVIEW STATUS")
print("CONFIRMED")
print("NEEDS_REVIEW")

print("\nIMPORTANT")
print(
    "The proposed decisions are recommendations, "
    "not verified human annotations."
)
print("Frozen version 1.0 remains unchanged.")
print("Version 1.1 has not been generated.")

report_lines = [
    "SCRIPT 51: D2 HUMAN ANNOTATION REVIEW",
    "=" * 65,
    f"Review pairs: {len(review)}",
    "Reviewer confirmations: 0",
    "Review status: PENDING",
    "Protocol v1.0: PRESERVED",
    "Protocol v1.1: NOT GENERATED",
    "Dataset modified: NO",
    "",
    f"Worksheet: {OUTPUT}"
]

REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(OUTPUT)
print(REPORT)

print("\nScript 51 completed.")
