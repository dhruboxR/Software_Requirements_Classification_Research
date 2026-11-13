
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ============================================================
# SCRIPT 18: FINAL MODEL SELECTION AND CHECKPOINT FREEZE
#
# Selected architecture: DistilBERT
# Selected run: original seed 42
# Selection metric: mean validation NFR F1 across seeds
# Deployment checkpoint: original seed-42 checkpoint
#
# NO test partition access.
# NO model training.
# NO new inference.
# NO threshold optimization.
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

CHECKPOINT = PROJECT / "checkpoints/05_distilbert_best"

PREDICTIONS = (
    PROJECT / "outputs/predictions"
    / "05_distilbert_validation_predictions.csv"
)

SEED_RESULTS = (
    PROJECT / "outputs/tables"
    / "09_transformer_seed_results.csv"
)

STABILITY = (
    PROJECT / "outputs/tables"
    / "09_transformer_seed_stability.csv"
)

REPORTS = PROJECT / "outputs/reports"

FREEZE_JSON = REPORTS / "18_final_model_freeze.json"
FREEZE_TXT = REPORTS / "18_final_model_selection_report.txt"

SELECTED_MODEL = "distilbert"
SELECTED_SEED = 42
MAX_LENGTH = 96
NFR_THRESHOLD = 0.5

EXPECTED_VALIDATION = 728


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def checkpoint_inventory(directory):
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file()
    )

    require(files, "Checkpoint directory contains no files")

    inventory = []

    for path in files:
        inventory.append({
            "relative_path": str(path.relative_to(directory)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    return inventory


print("=" * 76)
print("SCRIPT 18: FINAL MODEL SELECTION AND CHECKPOINT FREEZE")
print("=" * 76)

# ------------------------------------------------------------
# 1. Verify required research artifacts
# ------------------------------------------------------------

for path in [PREDICTIONS, SEED_RESULTS, STABILITY]:
    require(path.is_file(), f"Missing source file: {path}")

require(
    CHECKPOINT.is_dir(),
    f"Missing checkpoint directory: {CHECKPOINT}",
)

# Freeze files must not be silently overwritten.
require(
    not FREEZE_JSON.exists(),
    "Freeze JSON already exists; refusing to overwrite",
)

require(
    not FREEZE_TXT.exists(),
    "Freeze report already exists; refusing to overwrite",
)

# ------------------------------------------------------------
# 2. Verify checkpoint structure
# ------------------------------------------------------------

config_path = CHECKPOINT / "config.json"

require(
    config_path.is_file(),
    "Missing model config.json",
)

weight_candidates = [
    CHECKPOINT / "model.safetensors",
    CHECKPOINT / "pytorch_model.bin",
]

weight_files = [
    path for path in weight_candidates
    if path.is_file()
]

# Also support an explicitly indexed sharded checkpoint.
index_files = [
    CHECKPOINT / "model.safetensors.index.json",
    CHECKPOINT / "pytorch_model.bin.index.json",
]

weight_indexes = [
    path for path in index_files
    if path.is_file()
]

require(
    bool(weight_files or weight_indexes),
    "No standard model weights or weight index found",
)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

model_type = str(config.get("model_type", "")).lower()

require(
    model_type == "distilbert",
    f"Unexpected checkpoint model type: {model_type}",
)

# Check mapping if checkpoint declares it.
label2id = config.get("label2id")

if isinstance(label2id, dict):
    if "FR" in label2id or "NFR" in label2id:
        require(
            label2id.get("FR") == 0
            and label2id.get("NFR") == 1,
            "Declared FR/NFR label mapping differs",
        )

inventory = checkpoint_inventory(CHECKPOINT)

# ------------------------------------------------------------
# 3. Verify the seed-42 research record
# ------------------------------------------------------------

runs = pd.read_csv(SEED_RESULTS)
stability = pd.read_csv(STABILITY)

required_run_columns = {
    "architecture", "seed", "nfr_f1",
}

require(
    required_run_columns.issubset(runs.columns),
    "Seed results missing required columns",
)

selected_run = runs.loc[
    (runs["architecture"] == SELECTED_MODEL)
    & (runs["seed"].astype(int) == SELECTED_SEED)
]

require(
    len(selected_run) == 1,
    "Expected exactly one original seed-42 result",
)

selected_stability = stability.loc[
    stability["architecture"] == SELECTED_MODEL
]

require(
    len(selected_stability) == 1,
    "Expected one DistilBERT stability record",
)

recorded_f1 = float(
    selected_run.iloc[0]["nfr_f1"]
)

mean_f1 = float(
    selected_stability.iloc[0]["nfr_f1_mean"]
)

# ------------------------------------------------------------
# 4. Verify validation predictions and label mapping
# ------------------------------------------------------------

pred = pd.read_csv(
    PREDICTIONS,
    dtype={"requirement_id": str},
)

required_prediction_columns = {
    "requirement_id",
    "true_label",
    "predicted_label",
    "nfr_probability",
}

require(
    required_prediction_columns.issubset(pred.columns),
    "Missing prediction columns",
)

require(
    len(pred) == EXPECTED_VALIDATION,
    "Unexpected validation prediction count",
)

require(
    pred["requirement_id"].is_unique,
    "Duplicate validation requirement IDs",
)

require(
    pred["true_label"].isin(["FR", "NFR"]).all(),
    "Unexpected true labels",
)

require(
    pred["predicted_label"].isin(["FR", "NFR"]).all(),
    "Unexpected predicted labels",
)

prob = pd.to_numeric(
    pred["nfr_probability"],
    errors="raise",
).to_numpy()

require(
    np.isfinite(prob).all(),
    "Non-finite probabilities found",
)

require(
    ((prob >= 0) & (prob <= 1)).all(),
    "Probabilities outside [0, 1]",
)

threshold_predictions = np.where(
    prob >= NFR_THRESHOLD,
    "NFR",
    "FR",
)

require(
    (
        threshold_predictions
        == pred["predicted_label"].to_numpy()
    ).all(),
    "Stored predictions do not match threshold 0.5",
)

y_true = (
    pred["true_label"].eq("NFR").astype(int)
)

y_pred = (
    pred["predicted_label"].eq("NFR").astype(int)
)

recomputed_f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

require(
    abs(recomputed_f1 - recorded_f1) < 0.0002,
    "Seed-42 NFR F1 does not match saved research result",
)

# ------------------------------------------------------------
# 5. Prepare the immutable selection record
# ------------------------------------------------------------

source_hashes = {
    "validation_predictions": sha256(PREDICTIONS),
    "seed_results": sha256(SEED_RESULTS),
    "seed_stability": sha256(STABILITY),
}

freeze = {
    "experiment": "DL-18",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset_protocol": "D2-v1.1",

    "selection_status": "FROZEN",

    "selected_architecture": SELECTED_MODEL,
    "selected_seed": SELECTED_SEED,

    "checkpoint_directory": str(CHECKPOINT),
    "checkpoint_files": inventory,

    "tokenizer_checkpoint": (
        "distilbert/distilbert-base-uncased"
    ),
    "max_length": MAX_LENGTH,

    "classification_rule": {
        "positive_class": "NFR",
        "negative_class": "FR",
        "positive_class_id": 1,
        "negative_class_id": 0,
        "nfr_probability_threshold": NFR_THRESHOLD,
    },

    "model_selection_rule": (
        "Architecture chosen using highest mean "
        "validation NFR F1 over seeds 42, 17, and 29. "
        "Original seed-42 checkpoint used to avoid "
        "selecting a random seed by maximum validation "
        "score."
    ),

    "validation_evidence": {
        "seed42_nfr_f1": recomputed_f1,
        "three_seed_mean_nfr_f1": mean_f1,
        "validation_requirements": len(pred),
    },

    "no_additional_training": True,
    "no_ensemble": True,
    "no_threshold_optimization": True,
    "original_labels_modified": False,

    "selection_limitations": [
        "Architecture chosen after extensive validation reuse",
        "Only three training seeds per transformer",
        "Single validation partition",
        "Project-level independence not established",
        "Independent human annotation review incomplete",
    ],

    "test_accessed": False,
    "test_evaluation_status": "PENDING",

    "source_sha256": source_hashes,
}

# ------------------------------------------------------------
# 6. Write freeze metadata and human-readable report
# ------------------------------------------------------------

REPORTS.mkdir(parents=True, exist_ok=True)

FREEZE_JSON.write_text(
    json.dumps(freeze, indent=2) + "\n",
    encoding="utf-8",
)

report = [
    "FINAL MODEL SELECTION FREEZE",
    "=" * 76,
    "",
    "Dataset protocol: D2-v1.1",
    "Selected architecture: DistilBERT",
    "Selected checkpoint: original seed 42",
    f"Checkpoint: {CHECKPOINT}",
    "",
    f"Validation NFR F1 (seed 42): {recomputed_f1:.4f}",
    f"Mean NFR F1 (three seeds): {mean_f1:.4f}",
    "",
    "Inference configuration:",
    f"Tokenizer: {freeze['tokenizer_checkpoint']}",
    f"Maximum length: {MAX_LENGTH}",
    "Positive label: NFR",
    "Positive label ID: 1",
    f"Classification threshold: {NFR_THRESHOLD}",
    "",
    "Checkpoint files hashed: "
    + str(len(inventory)),
    "",
    "SELECTION RATIONALE",
    freeze["model_selection_rule"],
    "",
    "LIMITATIONS",
    *[
        "- " + item
        for item in freeze["selection_limitations"]
    ],
    "",
    "RESEARCH STATUS",
    "Final model selection: FROZEN",
    "New training: NO",
    "Test partition accessed: NO",
    "Test evaluation: PENDING",
    "",
    "Do not retune this model based on test results.",
]

FREEZE_TXT.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\nMODEL SELECTION")
print("Architecture:", SELECTED_MODEL)
print("Seed:", SELECTED_SEED)
print("Checkpoint:", CHECKPOINT)
print("Model type:", model_type)
print("Checkpoint files hashed:", len(inventory))

print("\nVALIDATION VERIFICATION")
print("Validation requirements:", len(pred))
print(f"Seed-42 NFR F1: {recomputed_f1:.4f}")
print(f"Three-seed mean NFR F1: {mean_f1:.4f}")
print("Stored predictions vs threshold: PASS")

print("\nRESEARCH STATUS")
print("Final model selection: FROZEN")
print("Additional training: NO")
print("Threshold optimization: NO")
print("Test set: SEALED")
print("Test evaluation: PENDING")

print("\nFILES GENERATED")
print(FREEZE_JSON)
print(FREEZE_TXT)

print("\nScript 18 completed.")
