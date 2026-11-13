
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# FINAL MODEL - SCRIPT 03
# COMPREHENSIVE FINAL EVALUATION REPORT
#
# Reads saved results only.
# No model loading or inference.
# No training or threshold optimization.
# No test-based model selection.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

TABLES = ROOT / "outputs/tables"
PREDS = ROOT / "outputs/predictions"
REPORTS = ROOT / "outputs/reports"
FIGURES = ROOT / "outputs/figures"
CONFIGS = ROOT / "configs"

RESEARCH = THESIS / "deep_learning_research"

FREEZE = CONFIGS / "18_final_model_freeze.json"
INTEGRITY = REPORTS / "00_model_integrity.json"
EVAL_META = REPORTS / "01_final_test_metrics.json"
PRED_FILE = PREDS / "01_final_test_predictions.csv"
METRICS_FILE = TABLES / "01_final_test_metrics.csv"

STABILITY = (
    RESEARCH / "outputs/tables"
    / "09_transformer_seed_stability.csv"
)

OUTPUT_REPORT = REPORTS / "03_final_evaluation_full_report.txt"
OUTPUT_COMPARISON = TABLES / "03_validation_test_comparison.csv"
OUTPUT_INVENTORY = TABLES / "03_final_artifact_inventory.csv"
OUTPUT_METADATA = REPORTS / "03_final_report_metadata.json"

OUTPUTS = [
    OUTPUT_REPORT,
    OUTPUT_COMPARISON,
    OUTPUT_INVENTORY,
    OUTPUT_METADATA,
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    require(path.is_file(), f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    return f"{value:.6f}"


print("=" * 76)
print("FINAL MODEL - SCRIPT 03: RESEARCH REPORT")
print("=" * 76)

# ------------------------------------------------------------
# 1. Safeguards and provenance
# ------------------------------------------------------------

for path in OUTPUTS:
    require(
        not path.exists(),
        f"Output already exists; refusing overwrite: {path}",
    )

freeze = read_json(FREEZE)
integrity = read_json(INTEGRITY)
evaluation = read_json(EVAL_META)

require(
    freeze["selection_status"] == "FROZEN",
    "Model selection must be frozen",
)
require(
    integrity["status"] == "PASS",
    "Model integrity must pass",
)
require(
    evaluation["status"] == "COMPLETED",
    "Final evaluation must be complete",
)
require(
    evaluation["selected_architecture"] == "distilbert",
    "Unexpected model",
)
require(
    int(evaluation["selected_seed"]) == 42,
    "Unexpected seed",
)
require(
    float(evaluation["threshold"]) == 0.5,
    "Unexpected threshold",
)
require(
    evaluation["training_performed"] is False,
    "Unexpected training flag",
)
require(
    evaluation["test_evaluation_performed"] is True,
    "No final test evaluation recorded",
)

# ------------------------------------------------------------
# 2. Verify final predictions and recompute metrics
# ------------------------------------------------------------

require(PRED_FILE.is_file(), "Missing test predictions")
require(METRICS_FILE.is_file(), "Missing test metrics")

require(
    sha256(PRED_FILE) == evaluation["prediction_file_sha256"],
    "Test prediction checksum mismatch",
)

pred = pd.read_csv(
    PRED_FILE,
    dtype={"requirement_id": str},
)

require(len(pred) == 727, "Expected 727 test records")
require(
    pred["requirement_id"].is_unique,
    "Duplicate requirement IDs",
)
require(
    pred["true_label"].isin(["FR", "NFR"]).all(),
    "Invalid true labels",
)
require(
    pred["predicted_label"].isin(["FR", "NFR"]).all(),
    "Invalid predicted labels",
)

y = pred["true_label"].map(
    {"FR": 0, "NFR": 1}
).to_numpy(dtype=int)

p = pred["predicted_label"].map(
    {"FR": 0, "NFR": 1}
).to_numpy(dtype=int)

prob = pd.to_numeric(
    pred["nfr_probability"],
    errors="raise",
).to_numpy(dtype=float)

require(
    np.isfinite(prob).all()
    and ((prob >= 0) & (prob <= 1)).all(),
    "Invalid predicted probabilities",
)

require(
    np.array_equal(p, (prob >= 0.5).astype(int)),
    "Predictions differ from frozen threshold",
)

cm = confusion_matrix(y, p, labels=[0, 1])
tn, fp, fn, tp = map(int, cm.ravel())

metrics = {
    "accuracy": accuracy_score(y, p),
    "balanced_accuracy": balanced_accuracy_score(y, p),
    "macro_f1": f1_score(y, p, average="macro"),
    "weighted_f1": f1_score(y, p, average="weighted"),
    "nfr_precision": precision_score(y, p),
    "nfr_recall": recall_score(y, p),
    "nfr_f1": f1_score(y, p),
    "average_precision": average_precision_score(y, prob),
    "roc_auc": roc_auc_score(y, prob),
}

saved = pd.read_csv(METRICS_FILE)

require(len(saved) == 1, "Unexpected metrics rows")

for name, value in metrics.items():
    require(
        name in saved.columns,
        f"Missing saved metric: {name}",
    )
    require(
        abs(float(saved.iloc[0][name]) - value) < 1e-6,
        f"Metric mismatch: {name}",
    )

require(
    (tn, fp, fn, tp) == (441, 47, 31, 208),
    "Confusion matrix mismatch",
)

# ------------------------------------------------------------
# 3. Validation and test comparison
# ------------------------------------------------------------

val = freeze["validation_evidence"]

validation = {
    "accuracy": 0.9052,
    "balanced_accuracy": 0.8995,
    "macro_f1": 0.8940,
    "nfr_precision": 0.8373,
    "nfr_recall": 0.8828,
    "nfr_f1": float(val["seed42_nfr_f1"]),
    "average_precision": 0.9270,
    "roc_auc": 0.9613,
}

# Validation figures come from the previously recorded
# seed-42 DistilBERT evaluation (Script 05).
# They are rounded values, so changes are approximate.
comparison_rows = []

for name, validation_value in validation.items():
    test_value = float(metrics[name])

    comparison_rows.append({
        "metric": name,
        "validation_seed42": validation_value,
        "final_test": test_value,
        "test_minus_validation": (
            test_value - validation_value
        ),
    })

comparison = pd.DataFrame(comparison_rows)

# ------------------------------------------------------------
# 4. Three-seed architecture comparison
# ------------------------------------------------------------

require(
    STABILITY.is_file(),
    f"Missing research stability table: {STABILITY}",
)

stability = pd.read_csv(STABILITY)

require(
    set(stability["architecture"])
    == {"distilbert", "bert", "roberta"},
    "Unexpected transformer architectures",
)

require(
    (stability["seed_count"].astype(int) == 3).all(),
    "Unexpected seed counts",
)

# ------------------------------------------------------------
# 5. Inventory of final-model artifacts
# ------------------------------------------------------------

artifact_paths = [
    FREEZE, INTEGRITY, EVAL_META,
    PRED_FILE, METRICS_FILE, STABILITY,
]

artifact_paths += sorted(FIGURES.glob("02_*.png"))
artifact_paths += sorted(FIGURES.glob("02_*.pdf"))

require(
    len(list(FIGURES.glob("02_*.png"))) == 4,
    "Expected four PNG figures",
)
require(
    len(list(FIGURES.glob("02_*.pdf"))) == 4,
    "Expected four PDF figures",
)

inventory = pd.DataFrame([
    {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    for path in artifact_paths
])

# ------------------------------------------------------------
# 6. Prepare detailed TXT research report
# ------------------------------------------------------------

report = [
    "FINAL DISTILBERT MODEL: COMPLETE EVALUATION REPORT",
    "=" * 78,
    "",
    "1. RESEARCH OBJECTIVE",
    "Binary classification of software requirements",
    "into Functional Requirements (FR) and",
    "Non-Functional Requirements (NFR).",
    "",
    "2. DATASET AND EVALUATION PROTOCOL",
    "Dataset: D2-v1.1",
    "Training: 3378 requirements",
    "Validation: 728 requirements",
    "Frozen test: 727 requirements",
    "Test FR: 488",
    "Test NFR: 239",
    "Test similarity groups: 694",
    "Cross-partition similarity-group overlap: NONE",
    "Project-level independence: NOT ESTABLISHED",
    "",
    "3. FINAL MODEL CONFIGURATION",
    "Architecture: DistilBERT",
    "Pretrained model: distilbert/distilbert-base-uncased",
    "Checkpoint: original seed 42",
    "Parameters: 66,955,010",
    "Maximum sequence length: 96",
    "FR class index: 0",
    "NFR class index: 1",
    "Decision threshold: 0.5",
    "Model selected before test evaluation.",
    "No test-based threshold optimization.",
    "",
    "4. MODEL-SELECTION RATIONALE",
    "DistilBERT had the highest observed mean",
    "validation NFR F1 across three random seeds.",
    "The original seed-42 checkpoint was selected",
    "without choosing the best-performing seed.",
    "Model selection reused the same validation set",
    "over multiple experiments.",
    "",
    "5. TRANSFORMER THREE-SEED STABILITY",
    stability.round(6).to_string(index=False),
    "",
    "6. FINAL TEST METRICS",
]

for name, value in metrics.items():
    report.append(f"{name}: {fmt(value)}")

report.extend([
    "",
    "7. FINAL CONFUSION MATRIX",
    "Rows = true labels, columns = predicted labels",
    "             Pred FR   Pred NFR",
    f"True FR      {tn:7d}   {fp:8d}",
    f"True NFR     {fn:7d}   {tp:8d}",
    "",
    f"Correct classifications: {tn + tp}",
    f"Incorrect classifications: {fp + fn}",
    "",
    "8. VALIDATION VERSUS TEST",
    comparison.round(6).to_string(index=False),
    "",
    "Validation metrics above are rounded historical",
    "seed-42 figures, except NFR F1 obtained from",
    "the frozen selection record.",
    "",
    "9. CLASSIFICATION ERROR INTERPRETATION",
    f"False positives: {fp}",
    f"False negatives: {fn}",
    "False positives are FR requirements predicted NFR.",
    "False negatives are NFR requirements predicted FR.",
    "No test error cases were used for model retuning.",
    "",
    "10. PREVIOUS QUALITATIVE ANALYSIS",
    "A separate validation-based error review selected",
    "43 difficult requirements.",
    "AI-assisted assessment agreed with the original",
    "labels for 11 cases, disagreed for 25, and",
    "recorded 7 uncertain judgments.",
    "These findings do not establish annotation errors.",
    "Independent human adjudication remains incomplete.",
    "",
    "11. GENERATED FIGURES",
])

report += [
    str(path)
    for path in sorted(FIGURES.glob("02_*"))
]

report.extend([
    "",
    "12. RESEARCH LIMITATIONS",
    "- Evaluation uses one held-out test partition.",
    "- Similarity-group separation was verified;",
    "  project-level independence was not established.",
    "- The validation partition was repeatedly reused",
    "  during experimental development.",
    "- Three seeds per transformer were evaluated.",
    "- Dataset taxonomy and annotation quality",
    "  require further independent assessment.",
    "- Selected error cases are not representative",
    "  of the complete dataset.",
    "- Results should not be interpreted as evidence",
    "  of superiority on unrelated datasets.",
    "",
    "13. REPRODUCIBILITY",
    "Selected checkpoint: SHA256 verified",
    "Tokenizer: SHA256 verified",
    "Dataset manifest: SHA256 verified",
    "Final predictions: SHA256 verified",
    "No further training performed.",
    "No additional test inference performed by",
    "this report-generation script.",
    "",
    "14. FINAL RESEARCH STATUS",
    "Final model selection: LOCKED",
    "Frozen test evaluation: COMPLETED",
    "Evaluation figures: COMPLETED",
    "Final research report: COMPLETED",
    "Post-test tuning: NOT PERFORMED",
    "",
    "15. SOURCE ARTIFACT INVENTORY",
    inventory.to_string(index=False),
])

# ------------------------------------------------------------
# 7. Save report and provenance
# ------------------------------------------------------------

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False,
)

inventory.to_csv(
    OUTPUT_INVENTORY,
    index=False,
)

OUTPUT_REPORT.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

OUTPUT_METADATA.write_text(
    json.dumps({
        "script": "FINAL_MODEL_03",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "COMPLETED",
        "frozen_test_rows": len(pred),
        "test_predictions_sha256": sha256(PRED_FILE),
        "source_evaluation_metadata_sha256": sha256(
            EVAL_META
        ),
        "final_nfr_f1": float(metrics["nfr_f1"]),
        "test_inference_performed": False,
        "training_performed": False,
        "threshold_optimized": False,
        "model_selection_changed": False,
    }, indent=2),
    encoding="utf-8",
)

print("\nRESULT VERIFICATION")
print("Final test records:", len(pred))
print("Metrics recomputed: PASS")
print("Prediction SHA256: PASS")
print("Confusion matrix: PASS")
print("Figure count: 8")
print("New inference: NO")

print("\nFINAL METRICS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nRESEARCH STATUS")
print("Final research report: COMPLETED")
print("Model selection: LOCKED")
print("Training: NO")
print("Test inference: NO")
print("Threshold optimization: NO")

print("\nFILES GENERATED")
for path in OUTPUTS:
    print(path)

print("\nScript 03 completed.")
