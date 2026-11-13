
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# SCRIPT 58: D2 BASELINE ERROR ANALYSIS
# - Compares B01 and B02 validation predictions
# - Studies NFR false negatives
# - Tracks corrected and newly introduced errors
# - Does not access the test set
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "research_modeling/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

PRED_FILE = TABLES / "57_d2_validation_predictions.csv"

B01 = "B01_LogisticRegression"
B02 = "B02_BalancedLogisticRegression"

df = pd.read_csv(PRED_FILE)

required = {
    "requirement_id", "true_label",
    "predicted_label", "nfr_probability", "model"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

if set(df["model"]) != {B01, B02}:
    raise ValueError("Unexpected model names")

if df.duplicated(["model", "requirement_id"]).any():
    raise ValueError("Duplicate model predictions")

a = df[df["model"] == B01].copy()
b = df[df["model"] == B02].copy()

if len(a) != 728 or len(b) != 728:
    raise ValueError("Unexpected validation count")

merged = a[
    ["requirement_id", "true_label",
     "predicted_label", "nfr_probability"]
].merge(
    b[
        ["requirement_id", "true_label",
         "predicted_label", "nfr_probability"]
    ],
    on="requirement_id",
    suffixes=("_b01", "_b02"),
    validate="one_to_one"
)

if len(merged) != 728:
    raise ValueError("Incomplete model comparison")

if not (
    merged["true_label_b01"]
    == merged["true_label_b02"]
).all():
    raise ValueError("True-label mismatch")

merged = merged.rename(
    columns={"true_label_b01": "true_label"}
).drop(columns=["true_label_b02"])

if set(merged["true_label"]) != {"FR", "NFR"}:
    raise ValueError("Unexpected labels")

if int((merged["true_label"] == "NFR").sum()) != 239:
    raise ValueError("Unexpected NFR validation support")

for model in ["b01", "b02"]:
    if not merged[f"predicted_label_{model}"].isin(
        ["FR", "NFR"]
    ).all():
        raise ValueError("Invalid predicted label")

    if not merged[f"nfr_probability_{model}"].between(
        0, 1
    ).all():
        raise ValueError("Invalid probability")

    merged[f"correct_{model}"] = (
        merged[f"predicted_label_{model}"]
        == merged["true_label"]
    )

merged["probability_change"] = (
    merged["nfr_probability_b02"]
    - merged["nfr_probability_b01"]
)

# Mutually exclusive classification of every example.
conditions = [
    merged["correct_b01"] & merged["correct_b02"],
    ~merged["correct_b01"] & merged["correct_b02"],
    merged["correct_b01"] & ~merged["correct_b02"],
    ~merged["correct_b01"] & ~merged["correct_b02"]
]

choices = [
    "BOTH_CORRECT",
    "B02_FIXED_B01_ERROR",
    "B02_INTRODUCED_ERROR",
    "BOTH_WRONG"
]

merged["comparison_status"] = np.select(
    conditions, choices, default="UNKNOWN"
)

if (merged["comparison_status"] == "UNKNOWN").any():
    raise RuntimeError("Unclassified comparison rows")

print("=" * 76)
print("SCRIPT 58: BASELINE ERROR ANALYSIS")
print("=" * 76)

print("\nOVERALL COMPARISON")
print(
    merged["comparison_status"]
    .value_counts()
    .to_string()
)

print("\nPER-CLASS COMPARISON")
class_summary = pd.crosstab(
    merged["true_label"],
    merged["comparison_status"]
)

print(class_summary.to_string())

# ------------------------------------------------------------
# NFR-specific error analysis
# ------------------------------------------------------------

nfr = merged[
    merged["true_label"] == "NFR"
].copy()

b01_missed = nfr[
    merged.loc[nfr.index, "predicted_label_b01"] == "FR"
]

b02_missed = nfr[
    merged.loc[nfr.index, "predicted_label_b02"] == "FR"
]

both_missed = nfr[
    (nfr["predicted_label_b01"] == "FR")
    & (nfr["predicted_label_b02"] == "FR")
]

rescued = nfr[
    (nfr["predicted_label_b01"] == "FR")
    & (nfr["predicted_label_b02"] == "NFR")
]

newly_missed = nfr[
    (nfr["predicted_label_b01"] == "NFR")
    & (nfr["predicted_label_b02"] == "FR")
]

print("\nNFR ERROR ANALYSIS")
print("Total NFR validation examples:", len(nfr))
print("B01 missed NFR:", len(b01_missed))
print("B02 missed NFR:", len(b02_missed))
print("Missed by both:", len(both_missed))
print("B01 misses corrected by B02:", len(rescued))
print("NFR newly missed by B02:", len(newly_missed))

if len(b01_missed):
    reduction = 100 * (
        len(b01_missed) - len(b02_missed)
    ) / len(b01_missed)
    print(
        "Net false-negative reduction (%):",
        round(reduction, 2)
    )

# ------------------------------------------------------------
# Probability analysis
# ------------------------------------------------------------

print("\nNFR PROBABILITY SUMMARY")

probability_summary = (
    merged.groupby("true_label")
    .agg(
        b01_mean_nfr_probability=(
            "nfr_probability_b01", "mean"
        ),
        b02_mean_nfr_probability=(
            "nfr_probability_b02", "mean"
        ),
        mean_change=(
            "probability_change", "mean"
        )
    ).round(4)
)

print(probability_summary.to_string())

# ------------------------------------------------------------
# Save validation-only diagnostics
# ------------------------------------------------------------

merged.to_csv(
    TABLES / "58_d2_prediction_comparison.csv",
    index=False
)

class_summary.to_csv(
    TABLES / "58_d2_error_transition_summary.csv"
)

probability_summary.to_csv(
    TABLES / "58_d2_probability_summary.csv"
)

# Save all remaining B02 NFR errors for later
# text-based investigation. IDs only in this file.
b02_missed.sort_values(
    "nfr_probability_b02"
).to_csv(
    TABLES / "58_d2_b02_missed_nfr.csv",
    index=False
)

rescued.sort_values(
    "probability_change",
    ascending=False
).to_csv(
    TABLES / "58_d2_nfr_rescued_by_balancing.csv",
    index=False
)

report_lines = [
    "D2 v1.1 BASELINE ERROR ANALYSIS",
    "=" * 65,
    "",
    "Evaluation partition: VALIDATION ONLY",
    f"Validation examples: {len(merged)}",
    f"NFR examples: {len(nfr)}",
    "",
    "OVERALL COMPARISON",
    merged["comparison_status"].value_counts().to_string(),
    "",
    "PER-CLASS COMPARISON",
    class_summary.to_string(),
    "",
    "NFR ERROR ANALYSIS",
    f"B01 false negatives: {len(b01_missed)}",
    f"B02 false negatives: {len(b02_missed)}",
    f"Missed by both: {len(both_missed)}",
    f"Corrected by B02: {len(rescued)}",
    f"Newly missed by B02: {len(newly_missed)}",
    "",
    "PROBABILITY SUMMARY",
    probability_summary.to_string(),
    "",
    "Limitations:",
    "- Only one validation partition was evaluated.",
    "- No confidence intervals calculated yet.",
    "- Model probabilities have not been calibrated.",
    "- No test records or predictions were accessed.",
    "- Semantic correctness of labels is not guaranteed."
]

report_path = REPORTS / "58_d2_baseline_error_analysis.txt"

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Error analysis: COMPLETED")
print("Evaluation: VALIDATION ONLY")
print("Test set: SEALED")
print("Model training: NONE IN SCRIPT 58")

print("\nSaved:", report_path)
print("Script 58 completed.")
