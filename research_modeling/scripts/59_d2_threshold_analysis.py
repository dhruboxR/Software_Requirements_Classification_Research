
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# SCRIPT 59: D2 v1.1 THRESHOLD ANALYSIS
#
# Uses saved validation predictions from Script 57.
# No model training.
# No test-set access.
# Threshold selection is exploratory.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "research_modeling/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

INPUT = TABLES / "57_d2_validation_predictions.csv"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MODELS = [
    "B01_LogisticRegression",
    "B02_BalancedLogisticRegression",
]

THRESHOLDS = np.round(
    np.arange(0.20, 0.8001, 0.025), 3
)

TARGET_NFR_RECALL = 0.85
BASELINE_THRESHOLD = 0.50

df = pd.read_csv(
    INPUT,
    dtype={
        "requirement_id": str,
        "true_label": str,
        "predicted_label": str,
        "model": str,
    }
)

required = {
    "requirement_id",
    "true_label",
    "predicted_label",
    "nfr_probability",
    "model",
}

if not required.issubset(df.columns):
    raise ValueError("Missing prediction columns")

if set(df["model"]) != set(MODELS):
    raise ValueError("Unexpected model names")

if df.duplicated(
    ["model", "requirement_id"]
).any():
    raise ValueError("Duplicate predictions")

df["nfr_probability"] = pd.to_numeric(
    df["nfr_probability"],
    errors="raise"
)

if not df["nfr_probability"].between(0, 1).all():
    raise ValueError("Invalid predicted probabilities")

print("=" * 76)
print("SCRIPT 59: D2 v1.1 THRESHOLD ANALYSIS")
print("=" * 76)

print("\nEXPERIMENT CONFIGURATION")
print("Evaluation: VALIDATION ONLY")
print("Threshold minimum:", THRESHOLDS.min())
print("Threshold maximum:", THRESHOLDS.max())
print("Threshold count:", len(THRESHOLDS))
print("NFR recall target:", TARGET_NFR_RECALL)
print("Test set accessed: NO")

records = []
ranking_metrics = []

for model_name in MODELS:

    subset = (
        df[df["model"] == model_name]
        .sort_values("requirement_id")
        .reset_index(drop=True)
    )

    if len(subset) != 728:
        raise ValueError(
            f"Unexpected validation size: {model_name}"
        )

    if subset["requirement_id"].nunique() != 728:
        raise ValueError("Duplicate validation IDs")

    if set(subset["true_label"]) != {"FR", "NFR"}:
        raise ValueError("Unexpected actual labels")

    if (
        (subset["true_label"] == "NFR").sum()
        != 239
    ):
        raise ValueError("Unexpected NFR support")

    y_true = (
        subset["true_label"]
        .map({"FR": 0, "NFR": 1})
        .to_numpy(dtype=int)
    )

    probabilities = (
        subset["nfr_probability"]
        .to_numpy(dtype=float)
    )

    ranking_metrics.append({
        "model": model_name,
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
    })

    for threshold in THRESHOLDS:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        records.append({
            "model": model_name,
            "threshold": float(threshold),
            "accuracy": accuracy_score(
                y_true, y_pred
            ),
            "balanced_accuracy": (
                balanced_accuracy_score(
                    y_true, y_pred
                )
            ),
            "macro_f1": f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ),
            "nfr_precision": precision_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            ),
            "nfr_recall": recall_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            ),
            "nfr_f1": f1_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

results = pd.DataFrame(records)
rank_df = pd.DataFrame(ranking_metrics)

# ------------------------------------------------------------
# Select exploratory thresholds
# ------------------------------------------------------------

best_rows = []
target_rows = []
baseline_rows = []

for name in MODELS:

    part = results[
        results["model"] == name
    ].copy()

    # Exploratory best NFR F1 on validation.
    # Tie-break using higher precision, then threshold.
    best = part.sort_values(
        ["nfr_f1", "nfr_precision", "threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    best_rows.append(best)

    # Among configurations satisfying recall >= 0.85,
    # prefer maximum precision, then F1.
    feasible = part[
        part["nfr_recall"] >= TARGET_NFR_RECALL
    ]

    if not feasible.empty:

        selected = feasible.sort_values(
            ["nfr_precision", "nfr_f1", "threshold"],
            ascending=[False, False, False]
        ).iloc[0]

        target_rows.append(selected)

    baseline = part[
        np.isclose(
            part["threshold"],
            BASELINE_THRESHOLD
        )
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "Missing baseline threshold"
        )

    baseline_rows.append(baseline.iloc[0])

best_df = pd.DataFrame(best_rows)
target_df = pd.DataFrame(target_rows)
baseline_df = pd.DataFrame(baseline_rows)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_path = (
    TABLES / "59_d2_threshold_sweep.csv"
)

best_path = (
    TABLES / "59_d2_best_nfr_f1_thresholds.csv"
)

target_path = (
    TABLES / "59_d2_recall_target_thresholds.csv"
)

rank_path = (
    TABLES / "59_d2_ranking_metrics.csv"
)

results.to_csv(results_path, index=False)
best_df.to_csv(best_path, index=False)
target_df.to_csv(target_path, index=False)
rank_df.to_csv(rank_path, index=False)

display_columns = [
    "model",
    "threshold",
    "nfr_precision",
    "nfr_recall",
    "nfr_f1",
    "macro_f1",
    "balanced_accuracy",
    "fp",
    "fn",
]

print("\nBASELINE THRESHOLD (0.50)")
print(
    baseline_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nEXPLORATORY BEST NFR F1")
print(
    best_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nRECALL >= 0.85 WITH HIGHEST PRECISION")

if target_df.empty:
    print("No configuration reached the recall target")
else:
    print(
        target_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

print("\nTHRESHOLD-INDEPENDENT RANKING METRICS")
print(
    rank_df.round(4).to_string(index=False)
)

# ------------------------------------------------------------
# Save methodological record
# ------------------------------------------------------------

metadata = {
    "experiment": 59,
    "dataset_protocol": "D2-v1.1",
    "evaluation": "VALIDATION_ONLY",
    "models": MODELS,
    "thresholds": THRESHOLDS.tolist(),
    "baseline_threshold": BASELINE_THRESHOLD,
    "recall_target": TARGET_NFR_RECALL,
    "test_accessed": False,
    "model_training": False,
    "selection_status": "EXPLORATORY",
    "limitations": [
        "Thresholds were compared on one validation set.",
        "Selecting the best validation result introduces "
        "selection optimism.",
        "Probability calibration was not assessed.",
        "No test-set performance is reported.",
        "No final model or threshold is approved."
    ]
}

metadata_path = (
    REPORTS / "59_d2_threshold_metadata.json"
)

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

report = [
    "D2 v1.1 THRESHOLD ANALYSIS",
    "=" * 65,
    "",
    "Evaluation: VALIDATION ONLY",
    f"Thresholds evaluated: {len(THRESHOLDS)}",
    "",
    "BASELINE THRESHOLD",
    baseline_df[display_columns].round(4).to_string(index=False),
    "",
    "BEST NFR F1 (EXPLORATORY)",
    best_df[display_columns].round(4).to_string(index=False),
    "",
    "RECALL TARGET",
    (
        target_df[display_columns].round(4).to_string(index=False)
        if not target_df.empty
        else "No feasible threshold"
    ),
    "",
    "THRESHOLD-INDEPENDENT METRICS",
    rank_df.round(4).to_string(index=False),
    "",
    "IMPORTANT:",
    "These are validation-based exploratory comparisons.",
    "Do not treat the best threshold as unbiased performance.",
    "The frozen test set remains sealed.",
]

report_path = (
    REPORTS / "59_d2_threshold_analysis.txt"
)

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Threshold analysis: COMPLETED")
print("Model training: NONE")
print("Test evaluation: NONE")
print("Threshold selection: EXPLORATORY")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
print(results_path)
print(best_path)
print(target_path)
print(rank_path)
print(report_path)
print(metadata_path)

print("\nScript 59 completed.")
