
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

# ============================================================
# SCRIPT 15: AI-ASSISTED ANNOTATION COMPARISON
#
# Compare proposed AI labels with original dataset labels.
#
# No model training.
# No modification of original annotations.
# No test-set access.
# Not an independent human agreement study.
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

TABLES = PROJECT / "outputs" / "tables"
REPORTS = PROJECT / "outputs" / "reports"

AI_FILE = (
    TABLES /
    "13_ai_assisted_requirement_review_completed.csv"
)

ORIGINAL_FILE = (
    TABLES /
    "13_original_labels_and_predictions.csv"
)

SUMMARY_FILE = (
    TABLES /
    "15_annotation_agreement_summary.csv"
)

MATRIX_FILE = (
    TABLES /
    "15_annotation_comparison_matrix.csv"
)

DETAIL_FILE = (
    TABLES /
    "15_annotation_comparison_details.csv"
)

DISAGREEMENT_FILE = (
    TABLES /
    "15_annotation_disagreements.csv"
)

UNCERTAIN_FILE = (
    TABLES /
    "15_uncertain_requirements.csv"
)

ISSUES_FILE = (
    TABLES /
    "15_annotation_issue_summary.csv"
)

REPORT_FILE = (
    REPORTS /
    "15_annotation_comparison_report.txt"
)

META_FILE = (
    REPORTS /
    "15_annotation_comparison_metadata.json"
)

TEXT_COL = "Requirement Text (Original Requirement)"

EXPOSED_IDS = {
    "RQP0481091",
    "RQP0481137",
    "RQP0481443",
    "RQP0481580",
    "RQP0481751",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


print("=" * 76)
print("SCRIPT 15: AI-ASSISTED ANNOTATION COMPARISON")
print("=" * 76)

# ------------------------------------------------------------
# 1. Load source files
# ------------------------------------------------------------

for path in [AI_FILE, ORIGINAL_FILE]:
    if not path.is_file():
        raise FileNotFoundError(path)

ai = pd.read_csv(
    AI_FILE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

original = pd.read_csv(
    ORIGINAL_FILE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

ai_required = {
    "requirement_id",
    TEXT_COL,
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_reasoning",
    "reviewer_issue_category",
    "review_status",
    "annotation_source",
}

original_required = {
    "requirement_id",
    "true_label",
}

require(
    ai_required.issubset(ai.columns),
    f"Missing AI columns: {ai_required - set(ai.columns)}",
)

require(
    original_required.issubset(original.columns),
    "Missing original-label columns",
)

require(len(ai) == 43, "Expected 43 AI annotations")
require(
    len(original) == 43,
    "Expected 43 original-label records",
)

require(
    ai["requirement_id"].is_unique,
    "Duplicate IDs in AI annotations",
)

require(
    original["requirement_id"].is_unique,
    "Duplicate IDs in original-label key",
)

require(
    set(ai["requirement_id"])
    == set(original["requirement_id"]),
    "AI and original requirement IDs do not match",
)

# ------------------------------------------------------------
# 2. Validate annotation fields
# ------------------------------------------------------------

for column in [
    "reviewer_label",
    "reviewer_confidence",
    "review_status",
    "reviewer_issue_category",
    "annotation_source",
]:
    ai[column] = ai[column].str.strip().str.upper()

ai["reviewer_reasoning"] = (
    ai["reviewer_reasoning"].str.strip()
)

ai[TEXT_COL] = ai[TEXT_COL].str.strip()

original["true_label"] = (
    original["true_label"].str.strip().str.upper()
)

require(
    ai["reviewer_label"].isin(
        ["FR", "NFR", "UNCERTAIN"]
    ).all(),
    "Unexpected AI label",
)

require(
    original["true_label"].isin(["FR", "NFR"]).all(),
    "Unexpected original label",
)

require(
    ai["reviewer_confidence"].isin(
        ["HIGH", "MEDIUM", "LOW"]
    ).all(),
    "Invalid confidence",
)

require(
    ai["review_status"].eq("REVIEWED").all(),
    "Some AI annotations are not marked REVIEWED",
)

require(
    ai["annotation_source"].eq(
        "AI_ASSISTED_NOT_INDEPENDENT_HUMAN"
    ).all(),
    "AI annotation provenance is missing or unexpected",
)

require(
    ai["reviewer_reasoning"].ne("").all(),
    "Missing classification reasoning",
)

require(
    ai[TEXT_COL].ne("").all(),
    "Missing requirement text",
)

require(
    ai["reviewer_issue_category"].ne("").all(),
    "Missing issue category",
)

# ------------------------------------------------------------
# 3. Merge AI judgments and original labels
#
# Model predictions are intentionally excluded.
# ------------------------------------------------------------

comparison = ai.merge(
    original[["requirement_id", "true_label"]],
    on="requirement_id",
    how="inner",
    validate="one_to_one",
)

require(
    len(comparison) == 43,
    "Comparison record count mismatch",
)

comparison["agreement_status"] = np.select(
    [
        comparison["reviewer_label"].eq("UNCERTAIN"),
        comparison["reviewer_label"].eq(
            comparison["true_label"]
        ),
    ],
    ["UNCERTAIN", "AGREE"],
    default="DISAGREE",
)

comparison["previously_exposed_case"] = (
    comparison["requirement_id"].isin(EXPOSED_IDS)
)

comparison = comparison.sort_values(
    "blind_review_number",
    key=lambda s: pd.to_numeric(s, errors="raise"),
).reset_index(drop=True)

# ------------------------------------------------------------
# 4. Agreement statistics
# ------------------------------------------------------------

total = len(comparison)

agree = int(
    comparison["agreement_status"].eq("AGREE").sum()
)

disagree = int(
    comparison["agreement_status"].eq("DISAGREE").sum()
)

uncertain = int(
    comparison["agreement_status"].eq("UNCERTAIN").sum()
)

decisive = agree + disagree

require(
    agree + disagree + uncertain == total,
    "Agreement accounting error",
)

coverage = decisive / total

agreement_on_decisive = (
    agree / decisive if decisive else float("nan")
)

overall_agreement = agree / total

print("\nDATA VALIDATION")
print("Requirements:", total)
print("Unique requirement IDs: PASS")
print("AI annotation provenance: VERIFIED")
print("Original labels modified: NO")
print("Model prediction columns analyzed: NO")
print("Test partition accessed: NO")

print("\nANNOTATION COMPARISON")
print("Agreement:", agree)
print("Disagreement:", disagree)
print("Uncertain:", uncertain)
print("Decisive AI annotations:", decisive)
print(f"Decisive coverage: {coverage:.4f}")
print(
    "Agreement among decisive annotations:",
    (
        f"{agreement_on_decisive:.4f}"
        if decisive
        else "N/A"
    ),
)
print(
    f"Agreement as fraction of all 43: "
    f"{overall_agreement:.4f}"
)

# ------------------------------------------------------------
# 5. Comparison matrix
# ------------------------------------------------------------

matrix = pd.crosstab(
    comparison["true_label"],
    comparison["reviewer_label"],
).reindex(
    index=["FR", "NFR"],
    columns=["FR", "NFR", "UNCERTAIN"],
    fill_value=0,
)

print("\nORIGINAL LABEL VS AI PROPOSED LABEL")
print(matrix.to_string())

# ------------------------------------------------------------
# 6. Disagreements, uncertainty, and issue categories
# ------------------------------------------------------------

disagreements = comparison.loc[
    comparison["agreement_status"].eq("DISAGREE")
].copy()

uncertainties = comparison.loc[
    comparison["agreement_status"].eq("UNCERTAIN")
].copy()

issue_summary = (
    comparison.groupby(
        ["reviewer_issue_category", "agreement_status"],
        dropna=False,
    )
    .size()
    .reset_index(name="requirements")
    .sort_values(
        ["reviewer_issue_category", "agreement_status"]
    )
)

print("\nDISAGREEMENT BY ORIGINAL CLASS")
print(
    disagreements["true_label"]
    .value_counts()
    .reindex(["FR", "NFR"], fill_value=0)
    .to_string()
)

print("\nUNCERTAIN REQUIREMENTS")
if not uncertainties.empty:
    print(
        uncertainties[
            [
                "requirement_id",
                "true_label",
                "reviewer_confidence",
                "reviewer_issue_category",
            ]
        ].to_string(index=False)
    )
else:
    print("None")

print("\nISSUE CATEGORY SUMMARY")
print(issue_summary.to_string(index=False))

# ------------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "total_requirements": total,
    "agreement": agree,
    "disagreement": disagree,
    "uncertain": uncertain,
    "decisive_annotations": decisive,
    "decisive_coverage": coverage,
    "agreement_among_decisive": agreement_on_decisive,
    "agreement_fraction_all": overall_agreement,
    "previously_exposed_cases": int(
        comparison["previously_exposed_case"].sum()
    ),
    "review_type": "AI_ASSISTED_EXPLORATORY",
}])

summary.to_csv(SUMMARY_FILE, index=False)

matrix.rename_axis(
    index="original_label",
    columns="ai_proposed_label",
).to_csv(MATRIX_FILE)

comparison.to_csv(
    DETAIL_FILE,
    index=False,
    encoding="utf-8-sig",
)

disagreements.to_csv(
    DISAGREEMENT_FILE,
    index=False,
    encoding="utf-8-sig",
)

uncertainties.to_csv(
    UNCERTAIN_FILE,
    index=False,
    encoding="utf-8-sig",
)

issue_summary.to_csv(
    ISSUES_FILE,
    index=False,
)

# ------------------------------------------------------------
# 8. Research report
# ------------------------------------------------------------

report = [
    "SCRIPT 15: AI-ASSISTED ANNOTATION COMPARISON",
    "=" * 76,
    "",
    "REVIEW PROVENANCE",
    "AI-assisted semantic classification.",
    "NOT independent human annotation.",
    "NOT expert label adjudication.",
    "",
    "DATA",
    f"Priority requirements: {total}",
    "Dataset: D2-v1.1, validation subset",
    "Test accessed: NO",
    "",
    "AGREEMENT",
    f"Agreement: {agree}",
    f"Disagreement: {disagree}",
    f"Uncertain: {uncertain}",
    f"Decisive annotations: {decisive}",
    f"Decisive coverage: {coverage:.4f}",
    (
        f"Agreement among decisive: "
        f"{agreement_on_decisive:.4f}"
        if decisive
        else "Agreement among decisive: N/A"
    ),
    f"Agreement fraction of all 43: {overall_agreement:.4f}",
    "",
    "ORIGINAL VS AI LABEL MATRIX",
    matrix.to_string(),
    "",
    "ISSUE CATEGORY COUNTS",
    issue_summary.to_string(index=False),
    "",
    "DISAGREEMENT RECORDS",
    disagreements[
        [
            "requirement_id",
            "true_label",
            "reviewer_label",
            "reviewer_confidence",
            "reviewer_issue_category",
            "reviewer_reasoning",
        ]
    ].to_string(index=False),
    "",
    "UNCERTAIN RECORDS",
    uncertainties[
        [
            "requirement_id",
            "true_label",
            "reviewer_issue_category",
            "reviewer_reasoning",
        ]
    ].to_string(index=False),
    "",
    "LIMITATIONS",
    "- This is AI versus original-label agreement.",
    "- The sample was selected by model error.",
    "- AI classifications are not verified ground truth.",
    "- Five requirement labels were exposed previously.",
    "- The original annotation taxonomy needs verification.",
    "- No independent human reviewers participated.",
    "- Agreement is not a generalization estimate.",
    "- No training or validation-based model tuning.",
    "- The frozen dataset was not modified.",
    "- The test partition remains sealed.",
]

REPORT_FILE.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

META_FILE.write_text(
    json.dumps({
        "experiment": "DL-15",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "review_type": "AI_ASSISTED_EXPLORATORY",
        "requirements": total,
        "agreement": agree,
        "disagreement": disagree,
        "uncertain": uncertain,
        "decisive_coverage": coverage,
        "agreement_among_decisive": (
            agreement_on_decisive if decisive else None
        ),
        "previously_exposed_ids": sorted(EXPOSED_IDS),
        "original_labels_modified": False,
        "training_performed": False,
        "test_accessed": False,
        "final_model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("AI-assisted comparison: COMPLETED")
print("Independent human review: NOT PERFORMED")
print("Original labels modified: NO")
print("Model training: NO")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    SUMMARY_FILE,
    MATRIX_FILE,
    DETAIL_FILE,
    DISAGREEMENT_FILE,
    UNCERTAIN_FILE,
    ISSUES_FILE,
    REPORT_FILE,
    META_FILE,
]:
    print(path)

print("\nScript 15 completed.")
