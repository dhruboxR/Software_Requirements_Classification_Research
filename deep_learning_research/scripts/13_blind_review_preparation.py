
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

SOURCE = TABLES / "12_priority_error_review.csv"

BLIND_OUTPUT = TABLES / "13_blind_requirement_review.csv"
KEY_OUTPUT = TABLES / "13_original_labels_and_predictions.csv"
REPORT_OUTPUT = REPORTS / "13_blind_review_preparation.txt"

TEXT_COL = "Requirement Text (Original Requirement)"
SEED = 42

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(
    SOURCE,
    dtype={"requirement_id": str},
    keep_default_na=False,
)

required = {
    "requirement_id",
    TEXT_COL,
    "true_label",
    "architectures_always_wrong",
}

if not required.issubset(df.columns):
    raise RuntimeError(
        f"Missing columns: {required - set(df.columns)}"
    )

if len(df) != 43:
    raise RuntimeError("Expected 43 priority requirements")

if not df["requirement_id"].is_unique:
    raise RuntimeError("Duplicate requirement IDs")

if df[TEXT_COL].str.strip().eq("").any():
    raise RuntimeError("Blank requirement text")

# Store the original labels and predictions separately.
key_columns = [
    col for col in df.columns
    if col not in {
        "reviewer_assessment",
        "reviewer_notes",
        "review_status",
        "possible_issue",
    }
]

key = df[key_columns].copy()

# Randomize order without using model predictions
# or dataset labels to determine review order.
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(df))

blind = df.iloc[indices][
    ["requirement_id", TEXT_COL]
].reset_index(drop=True)

blind.insert(
    0, "blind_review_number",
    range(1, len(blind) + 1)
)

blind["reviewer_label"] = ""
blind["reviewer_confidence"] = ""
blind["reviewer_reasoning"] = ""
blind["reviewer_issue_category"] = ""
blind["review_status"] = "NOT_REVIEWED"

blind.to_csv(
    BLIND_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

key.to_csv(
    KEY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

report = [
    "SCRIPT 13: BLIND MANUAL REVIEW PREPARATION",
    "=" * 70,
    "",
    f"Requirements: {len(blind)}",
    "Original labels hidden in review worksheet: YES",
    "Model predictions hidden in review worksheet: YES",
    "Review order randomized: YES",
    f"Shuffle seed: {SEED}",
    "Original annotations modified: NO",
    "Model training: NO",
    "Test accessed: NO",
    "",
    "REVIEW INSTRUCTIONS",
    "1. Read the requirement without opening the key.",
    "2. Assign FR, NFR, or UNCERTAIN.",
    "3. Record confidence: HIGH, MEDIUM, or LOW.",
    "4. Explain the classification using a defined taxonomy.",
    "5. Record ambiguity or mixed intent where applicable.",
    "6. Complete the worksheet before checking the key.",
    "7. Preserve disagreements for later adjudication.",
    "",
    "LIMITATIONS",
    "- Cases were selected using validation errors.",
    "- Previously exposed cases are not fully blind.",
    "- Independent reviewers are preferable.",
    "- Review does not establish external generalization.",
    "- The dataset remains frozen.",
]

REPORT_OUTPUT.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("=" * 70)
print("SCRIPT 13: BLIND REVIEW PREPARATION")
print("=" * 70)

print("\nREVIEW WORKSHEET")
print("Requirements:", len(blind))
print("Original labels visible: NO")
print("Model predictions visible: NO")
print("Review order randomized: YES")

print("\nRESEARCH STATUS")
print("Blind review worksheet: CREATED")
print("Manual review: PENDING")
print("Original annotations modified: NO")
print("Test set: SEALED")

print("\nFILES GENERATED")
print(BLIND_OUTPUT)
print(KEY_OUTPUT)
print(REPORT_OUTPUT)

print("\nScript 13 completed.")
