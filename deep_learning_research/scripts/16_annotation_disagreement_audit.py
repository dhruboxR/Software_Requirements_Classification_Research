
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

# ============================================================
# SCRIPT 16: ANNOTATION DISAGREEMENT AUDIT
#
# Inputs:
#   Script 15 annotation comparison details
#   Script 15 annotation disagreement summary
#
# Outputs:
#   Independent review worksheet (labels hidden)
#   Restricted comparison reference
#   Audit summary and research report
#
# No training.
# No source label modification.
# No test partition access.
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

TABLES = PROJECT / "outputs" / "tables"
REPORTS = PROJECT / "outputs" / "reports"

SOURCE = TABLES / "15_annotation_comparison_details.csv"
PREVIOUS_SUMMARY = TABLES / "15_annotation_agreement_summary.csv"

AUDIT_FILE = TABLES / "16_disagreement_audit.csv"
BLIND_FILE = TABLES / "16_independent_review_worksheet.csv"
REFERENCE_FILE = TABLES / "16_adjudication_reference.csv"
SUMMARY_FILE = TABLES / "16_audit_summary.csv"

REPORT_FILE = REPORTS / "16_disagreement_audit_report.txt"
METADATA_FILE = REPORTS / "16_disagreement_audit_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"

EXPECTED_TOTAL = 43
EXPECTED_AGREE = 11
EXPECTED_DISAGREE = 25
EXPECTED_UNCERTAIN = 7
EXPECTED_NONE_DISAGREEMENTS = 13

RANDOM_SEED = 16042

# Original labels for these cases appeared in earlier
# discussion. They cannot be claimed fully blinded.
PREVIOUSLY_EXPOSED = {
    "RQP0481091",
    "RQP0481137",
    "RQP0481443",
    "RQP0481580",
    "RQP0481751",
}

VALID_LABELS = {"FR", "NFR"}
VALID_AI_LABELS = {"FR", "NFR", "UNCERTAIN"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_hash(path):
    sha = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            sha.update(chunk)

    return sha.hexdigest()


print("=" * 76)
print("SCRIPT 16: ANNOTATION DISAGREEMENT AUDIT")
print("=" * 76)

# ------------------------------------------------------------
# 1. Load and verify Script 15 outputs
# ------------------------------------------------------------

for path in [SOURCE, PREVIOUS_SUMMARY]:
    if not path.is_file():
        raise FileNotFoundError(path)

df = pd.read_csv(
    SOURCE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

previous_summary = pd.read_csv(
    PREVIOUS_SUMMARY
)

required = {
    "requirement_id",
    TEXT_COL,
    "true_label",
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_reasoning",
    "reviewer_issue_category",
    "agreement_status",
    "annotation_source",
}

require(
    required.issubset(df.columns),
    f"Missing columns: {sorted(required - set(df.columns))}",
)

require(
    len(df) == EXPECTED_TOTAL,
    "Unexpected requirement count",
)

require(
    df["requirement_id"].is_unique,
    "Duplicate requirement IDs",
)

require(
    df[TEXT_COL].str.strip().ne("").all(),
    "Blank requirement text",
)

for col in [
    "true_label",
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_issue_category",
    "agreement_status",
    "annotation_source",
]:
    df[col] = df[col].str.strip().str.upper()

require(
    df["true_label"].isin(VALID_LABELS).all(),
    "Invalid original labels",
)

require(
    df["reviewer_label"].isin(VALID_AI_LABELS).all(),
    "Invalid AI labels",
)

require(
    df["reviewer_confidence"].isin(VALID_CONFIDENCE).all(),
    "Invalid confidence values",
)

require(
    df["annotation_source"].eq(
        "AI_ASSISTED_NOT_INDEPENDENT_HUMAN"
    ).all(),
    "Incorrect annotation provenance",
)

# Recalculate comparison status independently.
expected_status = np.select(
    [
        df["reviewer_label"].eq("UNCERTAIN"),
        df["reviewer_label"].eq(df["true_label"]),
    ],
    ["UNCERTAIN", "AGREE"],
    default="DISAGREE",
)

require(
    (df["agreement_status"].to_numpy()
     == expected_status).all(),
    "Script 15 agreement status mismatch",
)

status_counts = Counter(df["agreement_status"])

agree = status_counts["AGREE"]
disagree = status_counts["DISAGREE"]
uncertain = status_counts["UNCERTAIN"]

require(
    (agree, disagree, uncertain)
    == (
        EXPECTED_AGREE,
        EXPECTED_DISAGREE,
        EXPECTED_UNCERTAIN,
    ),
    "Script 15 counts differ from recorded results",
)

require(
    len(previous_summary) == 1,
    "Unexpected previous summary format",
)

for column, expected in [
    ("total_requirements", EXPECTED_TOTAL),
    ("agreement", EXPECTED_AGREE),
    ("disagreement", EXPECTED_DISAGREE),
    ("uncertain", EXPECTED_UNCERTAIN),
]:
    require(
        int(previous_summary.iloc[0][column]) == expected,
        f"Previous summary mismatch: {column}",
    )

print("\nINPUT VALIDATION")
print("Requirements:", len(df))
print("Agreement:", agree)
print("Disagreement:", disagree)
print("Uncertain:", uncertain)
print("Annotation provenance: AI-ASSISTED")
print("Input verification: PASS")

# ------------------------------------------------------------
# 2. Select cases requiring adjudication
# ------------------------------------------------------------

audit = df.loc[
    df["agreement_status"].isin(
        ["DISAGREE", "UNCERTAIN"]
    )
].copy()

require(
    len(audit) == 32,
    "Expected 32 disputed/uncertain requirements",
)

audit["previously_exposed_case"] = (
    audit["requirement_id"].isin(
        PREVIOUSLY_EXPOSED
    )
)

audit["none_issue_disagreement"] = (
    audit["agreement_status"].eq("DISAGREE")
    & audit["reviewer_issue_category"].eq("NONE")
)

audit["low_confidence"] = (
    audit["reviewer_confidence"].eq("LOW")
)

none_count = int(
    audit["none_issue_disagreement"].sum()
)

require(
    none_count == EXPECTED_NONE_DISAGREEMENTS,
    "Unexpected NONE-category disagreement count",
)

# Priority flags can overlap. They are not separate
# groups and must not be summed as unique cases.
audit["audit_flags"] = audit.apply(
    lambda row: ";".join(
        flag
        for condition, flag in [
            (
                row["none_issue_disagreement"],
                "NONE_CATEGORY_DISAGREEMENT",
            ),
            (
                row["low_confidence"],
                "LOW_CONFIDENCE",
            ),
            (
                row["previously_exposed_case"],
                "PREVIOUSLY_EXPOSED",
            ),
            (
                row["agreement_status"] == "UNCERTAIN",
                "UNCERTAIN_CLASSIFICATION",
            ),
        ]
        if condition
    ) or "STANDARD_DISAGREEMENT",
    axis=1,
)

def assign_priority(row):
    if row["low_confidence"]:
        return "P1_LOW_CONFIDENCE"

    if row["agreement_status"] == "UNCERTAIN":
        return "P2_UNCERTAIN"

    if row["none_issue_disagreement"]:
        return "P3_UNEXPLAINED_DISAGREEMENT"

    return "P4_OTHER_DISAGREEMENT"


audit["audit_priority"] = audit.apply(
    assign_priority,
    axis=1,
)

priority_order = {
    "P1_LOW_CONFIDENCE": 1,
    "P2_UNCERTAIN": 2,
    "P3_UNEXPLAINED_DISAGREEMENT": 3,
    "P4_OTHER_DISAGREEMENT": 4,
}

audit["_priority_sort"] = (
    audit["audit_priority"].map(priority_order)
)

audit = audit.sort_values(
    ["_priority_sort", "requirement_id"]
).drop(
    columns="_priority_sort"
).reset_index(drop=True)

# ------------------------------------------------------------
# 3. Create complete internal audit table
# ------------------------------------------------------------

audit.to_csv(
    AUDIT_FILE,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 4. Create separate independent-review worksheet
#
# No original labels.
# No proposed AI labels.
# No AI reasoning.
# No model predictions.
# No AI-derived priority or disagreement flags.
# ------------------------------------------------------------

blind = audit[
    ["requirement_id", TEXT_COL]
].copy()

rng = np.random.default_rng(RANDOM_SEED)

blind = blind.iloc[
    rng.permutation(len(blind))
].reset_index(drop=True)

blind.insert(
    0,
    "review_number",
    range(1, len(blind) + 1),
)

blind["human_label"] = ""
blind["human_confidence"] = ""
blind["human_issue_category"] = ""
blind["human_reasoning"] = ""
blind["human_review_status"] = "NOT_REVIEWED"

blind.to_csv(
    BLIND_FILE,
    index=False,
    encoding="utf-8-sig",
)

for forbidden in [
    "true_label",
    "reviewer_label",
    "agreement_status",
    "audit_priority",
]:
    require(
        forbidden not in blind.columns,
        f"Blind worksheet contains {forbidden}",
    )

# ------------------------------------------------------------
# 5. Create restricted comparison reference
# ------------------------------------------------------------

reference_columns = [
    "requirement_id",
    "true_label",
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_reasoning",
    "reviewer_issue_category",
    "agreement_status",
    "audit_priority",
    "audit_flags",
    "previously_exposed_case",
]

reference = audit[reference_columns].copy()

reference.to_csv(
    REFERENCE_FILE,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 6. Generate summary statistics
# ------------------------------------------------------------

print("\nDISAGREEMENT AUDIT")

print("Total selected:", len(audit))
print("Disagreements:", disagree)
print("Uncertain classifications:", uncertain)
print("Disagreements marked NONE:", none_count)

print(
    "Low-confidence cases:",
    int(audit["low_confidence"].sum()),
)

print(
    "Previously exposed selected cases:",
    int(audit["previously_exposed_case"].sum()),
)

print("\nAUDIT PRIORITY")

priority_counts = (
    audit["audit_priority"]
    .value_counts()
    .reindex(priority_order.keys(), fill_value=0)
)

print(priority_counts.to_string())

print("\nDISAGREEMENTS BY ORIGINAL LABEL")

label_counts = (
    audit.loc[
        audit["agreement_status"].eq("DISAGREE"),
        "true_label",
    ]
    .value_counts()
    .reindex(["FR", "NFR"], fill_value=0)
)

print(label_counts.to_string())

print("\nISSUE CATEGORIES")

issue_counts = (
    audit["reviewer_issue_category"]
    .value_counts()
    .sort_index()
)

print(issue_counts.to_string())

summary = pd.DataFrame([{
    "total_requirements": EXPECTED_TOTAL,
    "agreement": agree,
    "disagreement": disagree,
    "uncertain": uncertain,
    "selected_for_adjudication": len(audit),
    "none_issue_disagreements": none_count,
    "low_confidence_cases": int(
        audit["low_confidence"].sum()
    ),
    "previously_exposed_selected": int(
        audit["previously_exposed_case"].sum()
    ),
    "independent_human_review_complete": False,
    "original_annotations_modified": False,
    "test_accessed": False,
}])

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

# ------------------------------------------------------------
# 7. Write research report
# ------------------------------------------------------------

report_lines = [
    "SCRIPT 16: ANNOTATION DISAGREEMENT AUDIT",
    "=" * 76,
    "",
    "DATASET",
    "D2-v1.1",
    "Validation-derived priority requirements: 43",
    "Review type: AI-assisted exploratory assessment",
    "",
    "SCRIPT 15 RESULTS",
    f"Agreement: {agree}",
    f"Disagreement: {disagree}",
    f"Uncertain: {uncertain}",
    "",
    "AUDIT POPULATION",
    f"Selected cases: {len(audit)}",
    f"NONE-category disagreements: {none_count}",
    "Low-confidence cases: "
    + str(int(audit["low_confidence"].sum())),
    "Previously exposed selected cases: "
    + str(int(audit["previously_exposed_case"].sum())),
    "",
    "PRIORITY COUNTS",
    priority_counts.to_string(),
    "",
    "DISAGREEMENTS BY ORIGINAL LABEL",
    label_counts.to_string(),
    "",
    "ISSUE CATEGORIES",
    issue_counts.to_string(),
    "",
    "REVIEW DESIGN",
    "The independent-review worksheet contains",
    "only requirement IDs, texts, and blank human",
    "annotation fields.",
    "",
    "Original dataset labels and AI judgments are",
    "stored in a separate comparison reference.",
    "",
    "The reviewer must not open the reference before",
    "completing the independent annotations.",
    "",
    "If a reviewer has previously seen labels or model",
    "predictions for a case, that exposure must be",
    "reported rather than claiming strict blinding.",
    "",
    "RECOMMENDED HUMAN LABELS",
    "FR",
    "NFR",
    "UNCERTAIN",
    "",
    "CONFIDENCE LEVELS",
    "HIGH",
    "MEDIUM",
    "LOW",
    "",
    "METHODOLOGICAL LIMITATIONS",
    "- Cases were selected using validation errors.",
    "- AI predictions are not independent ground truth.",
    "- Some original labels were previously exposed.",
    "- Dataset-specific label definitions remain to",
    "  be verified.",
    "- Low agreement does not prove label errors.",
    "- This audit cannot estimate full-dataset",
    "  annotation error prevalence.",
    "- No new training or model tuning was performed.",
    "- Original dataset labels remain unchanged.",
    "- Test partition remains sealed.",
    "",
    "NEXT RESEARCH STAGE",
    "Independent human review or formal documentation",
    "of unresolved cases, followed by consolidation",
    "of existing modeling results.",
]

REPORT_FILE.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

METADATA_FILE.write_text(
    json.dumps({
        "experiment": "DL-16",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "D2-v1.1",
        "review_type": "AI_ASSISTED_DISAGREEMENT_AUDIT",
        "source_sha256": file_hash(SOURCE),
        "previous_summary_sha256": file_hash(
            PREVIOUS_SUMMARY
        ),
        "selected_requirements": len(audit),
        "agreements": agree,
        "disagreements": disagree,
        "uncertain": uncertain,
        "none_issue_disagreements": none_count,
        "blind_review_shuffle_seed": RANDOM_SEED,
        "human_review_complete": False,
        "original_labels_modified": False,
        "model_training": False,
        "test_accessed": False,
        "final_model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

# ------------------------------------------------------------
# 8. Final verification
# ------------------------------------------------------------

assert len(blind) == 32
assert len(reference) == 32
assert set(blind["requirement_id"]) == set(
    reference["requirement_id"]
)

print("\nRESEARCH STATUS")
print("Annotation disagreement audit: COMPLETED")
print("Independent review worksheet: CREATED")
print("Independent human annotation: PENDING")
print("Original annotations modified: NO")
print("Model training: NO")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    AUDIT_FILE,
    BLIND_FILE,
    REFERENCE_FILE,
    SUMMARY_FILE,
    REPORT_FILE,
    METADATA_FILE,
]:
    print(path)

print("\nScript 16 completed.")
