
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# SCRIPT 61: TRAIN-OOF THRESHOLD VALIDATION
#
# - Uses OOF-selected thresholds from Script 60
# - Applies them to existing Script 57 validation scores
# - Does not search for new thresholds
# - Does not train models
# - Does not access the test partition
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "research_modeling/outputs"
TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

PREDICTIONS = (
    TABLES / "57_d2_validation_predictions.csv"
)

OOF_BEST = TABLES / "60_d2_oof_best_f1.csv"
OOF_RECALL = TABLES / "60_d2_oof_recall_target.csv"

SUMMARY_PATH = (
    TABLES / "61_d2_oof_threshold_validation.csv"
)

REPORT_PATH = (
    REPORTS / "61_d2_oof_threshold_validation.txt"
)

METADATA_PATH = (
    REPORTS / "61_d2_oof_threshold_validation.json"
)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


# ------------------------------------------------------------
# 1. Load existing experiment outputs
# ------------------------------------------------------------

pred = pd.read_csv(
    PREDICTIONS,
    dtype={"requirement_id": str}
)

best = pd.read_csv(OOF_BEST)
recall_target = pd.read_csv(OOF_RECALL)

models = [
    "B01_LogisticRegression",
    "B02_BalancedLogisticRegression",
]

if set(pred["model"]) != set(models):
    raise ValueError("Unexpected prediction models")

if pred.duplicated(
    ["model", "requirement_id"]
).any():
    raise ValueError("Duplicate validation predictions")

if set(best["model"]) != set(models):
    raise ValueError("Incomplete OOF F1 selection")

if set(recall_target["model"]) != set(models):
    raise ValueError("Incomplete OOF recall selection")

for frame in (best, recall_target):
    if frame["model"].duplicated().any():
        raise ValueError("Duplicate threshold selections")

# Thresholds are read from Script 60, not selected
# from Script 57 or Script 59 validation performance.
configs = []

for objective, source in [
    ("MAX_NFR_F1", best),
    ("RECALL_TARGET_085", recall_target),
]:
    for _, row in source.iterrows():
        configs.append({
            "objective": objective,
            "model": row["model"],
            "threshold": float(row["threshold"]),
        })

print("=" * 76)
print("SCRIPT 61: TRAIN-OOF THRESHOLD VALIDATION")
print("=" * 76)

print("\nCONFIGURATION")
print("Dataset protocol: D2-v1.1")
print("Threshold source: TRAIN OOF (Script 60)")
print("Probability source: VALIDATION (Script 57)")
print("Threshold optimization in Script 61: NONE")
print("Test partition accessed: NO")

# ------------------------------------------------------------
# 2. Evaluate fixed configurations
# ------------------------------------------------------------

records = []

for config in configs:

    subset = (
        pred[pred["model"] == config["model"]]
        .sort_values("requirement_id")
        .reset_index(drop=True)
    )

    if len(subset) != 728:
        raise ValueError("Unexpected validation size")

    if subset["requirement_id"].nunique() != 728:
        raise ValueError("Duplicate validation IDs")

    if set(subset["true_label"]) != {"FR", "NFR"}:
        raise ValueError("Unexpected true labels")

    y_true = (
        subset["true_label"]
        .map({"FR": 0, "NFR": 1})
        .to_numpy(dtype=int)
    )

    if (
        np.count_nonzero(y_true == 0) != 489
        or np.count_nonzero(y_true == 1) != 239
    ):
        raise ValueError("Unexpected class support")

    probabilities = pd.to_numeric(
        subset["nfr_probability"],
        errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")

    if np.any(
        (probabilities < 0)
        | (probabilities > 1)
    ):
        raise ValueError("Invalid probabilities")

    threshold = config["threshold"]

    if not 0 < threshold < 1:
        raise ValueError("Invalid threshold")

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    records.append({
        "objective": config["objective"],
        "model": config["model"],
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true, y_pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "nfr_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "nfr_recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "nfr_f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

results = pd.DataFrame(records)

if len(results) != 4:
    raise RuntimeError("Expected four configurations")

# ------------------------------------------------------------
# 3. Compare performance by research objective
# ------------------------------------------------------------

display_cols = [
    "objective",
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

print("\nFIXED OOF-SELECTED THRESHOLDS")
print(
    results[display_cols]
    .round(4)
    .to_string(index=False)
)

print("\nCONFUSION MATRICES")
print("Rows: actual FR, NFR")
print("Columns: predicted FR, NFR")

for _, row in results.iterrows():
    print(
        f"\n{row['objective']} | {row['model']} "
        f"| threshold={row['threshold']:.3f}"
    )

    print([
        [int(row["tn"]), int(row["fp"])],
        [int(row["fn"]), int(row["tp"])]
    ])

# ------------------------------------------------------------
# 4. Save research outputs
# ------------------------------------------------------------

results.to_csv(
    SUMMARY_PATH,
    index=False
)

report_lines = [
    "SCRIPT 61: TRAIN-OOF THRESHOLD VALIDATION",
    "=" * 65,
    "",
    "Dataset: D2-v1.1",
    "Threshold selection: training OOF only",
    "Evaluation: existing external validation scores",
    "Test accessed: NO",
    "",
    "FIXED OOF-SELECTED THRESHOLDS",
    results[display_cols].round(4).to_string(index=False),
    "",
    "LIMITATIONS",
    "- The external validation set was previously explored.",
    "- These are not unbiased independent estimates.",
    "- No additional threshold search was performed.",
    "- No test predictions were generated.",
    "- Original-project independence is not established.",
]

REPORT_PATH.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

metadata = {
    "experiment": 61,
    "dataset_protocol": "D2-v1.1",
    "threshold_source": "TRAIN_OOF_SCRIPT_60",
    "probability_source": "VALIDATION_SCRIPT_57",
    "validation_previously_explored": True,
    "test_accessed": False,
    "model_training": False,
    "threshold_search": False,
    "selection_status": "DEVELOPMENT_STAGE",
    "input_sha256": {
        path.name: sha256(path)
        for path in (
            PREDICTIONS,
            OOF_BEST,
            OOF_RECALL,
        )
    },
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("OOF threshold validation: COMPLETED")
print("Model training: NONE")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
print(SUMMARY_PATH)
print(REPORT_PATH)
print(METADATA_PATH)

print("\nScript 61 completed.")
