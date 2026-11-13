
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd

# ============================================================
# SCRIPT 14: ANNOTATION REVIEW VALIDATOR
#
# - Validates the blind review worksheet
# - Checks reviewer labels, confidence and reasoning
# - Summarizes manual review progress
# - Writes annotation guidelines and progress reports
#
# Does not read original labels or model predictions.
# Does not modify the worksheet.
# Does not access the test partition.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"

SOURCE = TABLES / "13_blind_requirement_review.csv"

PROTOCOL = REPORTS / "14_annotation_protocol.txt"
REPORT = REPORTS / "14_review_progress.txt"
METADATA = REPORTS / "14_review_progress.json"

TEXT_COL = "Requirement Text (Original Requirement)"

LABELS = {"FR", "NFR", "UNCERTAIN"}
CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
STATUSES = {"NOT_REVIEWED", "REVIEWED"}

ISSUE_CODES = {
    "NONE",
    "MIXED_FR_NFR",
    "IMPLICIT_QUALITY",
    "INTERFACE_AMBIGUITY",
    "SECURITY_AMBIGUITY",
    "PERFORMANCE_AMBIGUITY",
    "DOMAIN_TERMINOLOGY",
    "AMBIGUOUS_WORDING",
    "INSUFFICIENT_CONTEXT",
    "OTHER",
}

print("=" * 76)
print("SCRIPT 14: ANNOTATION REVIEW VALIDATOR")
print("=" * 76)

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(
    SOURCE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

required = {
    "blind_review_number",
    "requirement_id",
    TEXT_COL,
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_reasoning",
    "reviewer_issue_category",
    "review_status",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing worksheet columns: {sorted(missing)}"
    )

if len(df) != 43:
    raise RuntimeError(
        f"Expected 43 requirements; found {len(df)}"
    )

if not df["requirement_id"].is_unique:
    raise RuntimeError("Duplicate requirement IDs")

if df[TEXT_COL].str.strip().eq("").any():
    raise RuntimeError("Blank requirement text")

if not df["blind_review_number"].is_unique:
    raise RuntimeError("Duplicate review numbers")

# Normalize review-field formatting for validation only.
# The original worksheet is never overwritten.

review_columns = [
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_reasoning",
    "reviewer_issue_category",
    "review_status",
]

for col in review_columns:
    df[col] = df[col].str.strip()

for col in [
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_issue_category",
    "review_status",
]:
    df[col] = df[col].str.upper()

invalid_rows = []

for _, row in df.iterrows():
    number = row["blind_review_number"]
    status = row["review_status"]

    errors = []

    if status not in STATUSES:
        errors.append("Invalid review status")

    if status == "REVIEWED":
        if row["reviewer_label"] not in LABELS:
            errors.append("Invalid/missing reviewer label")

        if row["reviewer_confidence"] not in CONFIDENCE:
            errors.append("Invalid/missing confidence")

        if not row["reviewer_reasoning"]:
            errors.append("Missing reviewer reasoning")

        if row["reviewer_issue_category"] not in ISSUE_CODES:
            errors.append("Invalid/missing issue category")

    if errors:
        invalid_rows.append({
            "review_number": number,
            "requirement_id": row["requirement_id"],
            "issues": "; ".join(errors),
        })

reviewed = df["review_status"].eq("REVIEWED")
pending = df["review_status"].eq("NOT_REVIEWED")

reviewed_count = int(reviewed.sum())
pending_count = int(pending.sum())

complete = (
    reviewed_count == 43
    and len(invalid_rows) == 0
)

print("\nWORKSHEET VALIDATION")
print("Requirements:", len(df))
print("Unique IDs: PASS")
print("Original labels accessed: NO")
print("Model predictions accessed: NO")

print("\nREVIEW PROGRESS")
print("Reviewed:", reviewed_count)
print("Pending:", pending_count)
print(
    "Completion percentage:",
    round(reviewed_count / len(df) * 100, 2)
)

print("\nANNOTATION DISTRIBUTION")

if reviewed_count:
    print(
        df.loc[reviewed, "reviewer_label"]
        .value_counts()
        .to_string()
    )
else:
    print("No annotations submitted yet.")

print("\nVALIDATION ERRORS")
print("Invalid reviewed records:", len(invalid_rows))

for error in invalid_rows[:20]:
    print(
        f"Review #{error['review_number']}: "
        f"{error['issues']}"
    )

# ------------------------------------------------------------
# Write formal annotation protocol
# ------------------------------------------------------------

protocol_lines = [
    "D2 v1.1 MANUAL ANNOTATION PROTOCOL",
    "=" * 76,
    "",
    "PURPOSE",
    "Review 43 validation requirements selected",
    "through persistent transformer error analysis.",
    "",
    "LABEL DEFINITIONS",
    "",
    "FR:",
    "The requirement primarily specifies a system",
    "behavior, operation, or capability.",
    "",
    "NFR:",
    "The requirement primarily specifies a system",
    "quality property or operational constraint.",
    "",
    "UNCERTAIN:",
    "The requirement cannot be confidently assigned",
    "FR or NFR without further context or adjudication.",
    "",
    "REVIEW PROCEDURE",
    "1. Read each requirement without consulting the key.",
    "2. Identify the principal requirement intent.",
    "3. Assign FR, NFR, or UNCERTAIN.",
    "4. Record HIGH, MEDIUM, or LOW confidence.",
    "5. Write an evidence-based justification.",
    "6. Assign one primary issue category.",
    "7. Set review_status to REVIEWED.",
    "8. Do not modify the original requirement text.",
    "9. Do not change the frozen dataset labels.",
    "",
    "ISSUE CATEGORIES",
    *sorted(ISSUE_CODES),
    "",
    "REVIEW GUIDANCE",
    "A quality-related keyword does not prove NFR.",
    "An action verb does not automatically prove FR.",
    "Functional behaviors may support quality objectives.",
    "Mixed requirements must be assessed carefully.",
    "When context is insufficient, use UNCERTAIN.",
    "",
    "INDEPENDENCE LIMITATIONS",
    "These examples were selected using model errors.",
    "Some cases have already been discussed openly.",
    "A second independent reviewer is preferable.",
    "The original dataset taxonomy should be checked.",
    "",
    "RESEARCH LIMITATIONS",
    "No test requirements are reviewed.",
    "No model training is conducted.",
    "No original labels are modified.",
    "Review judgments must not be used to tune models",
    "on the same validation partition.",
    "",
    "REFERENCE FRAMEWORKS",
    "ISO/IEC/IEEE 29148:2018",
    "ISO/IEC 25010:2023",
    "Dataset-specific annotation definitions: TO VERIFY",
]

PROTOCOL.write_text(
    "\n".join(protocol_lines) + "\n",
    encoding="utf-8",
)

progress_lines = [
    "SCRIPT 14: ANNOTATION REVIEW PROGRESS",
    "=" * 76,
    "",
    f"Total requirements: {len(df)}",
    f"Reviewed: {reviewed_count}",
    f"Pending: {pending_count}",
    f"Invalid reviewed records: {len(invalid_rows)}",
    f"Review complete: {complete}",
    "",
    "ANNOTATION DISTRIBUTION",
    (
        df.loc[reviewed, "reviewer_label"]
        .value_counts()
        .to_string()
        if reviewed_count
        else "No annotations submitted yet."
    ),
    "",
    "VALIDATION ERRORS",
    *[
        f"{e['requirement_id']}: {e['issues']}"
        for e in invalid_rows
    ],
    "",
    "Original labels accessed: NO",
    "Model predictions accessed: NO",
    "Frozen dataset modified: NO",
    "Test set: SEALED",
]

REPORT.write_text(
    "\n".join(progress_lines) + "\n",
    encoding="utf-8",
)

METADATA.write_text(
    json.dumps({
        "experiment": "DL-14",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "requirements": len(df),
        "reviewed": reviewed_count,
        "pending": pending_count,
        "invalid_records": len(invalid_rows),
        "review_complete": complete,
        "original_labels_accessed": False,
        "model_predictions_accessed": False,
        "original_labels_modified": False,
        "test_accessed": False,
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print(
    "Manual annotation review:",
    "COMPLETED" if complete else "PENDING"
)
print("Frozen dataset modified: NO")
print("Test set: SEALED")

print("\nFILES GENERATED")
print(PROTOCOL)
print(REPORT)
print(METADATA)

print("\nScript 14 completed.")
