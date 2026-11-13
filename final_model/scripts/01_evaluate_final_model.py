
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# FINAL MODEL - SCRIPT 01
# ONE-TIME FROZEN TEST EVALUATION
#
# Dataset: D2-v1.1
# Model: DistilBERT
# Seed: 42
# Max length: 96
# Positive class: NFR (1)
# Threshold: 0.5
#
# No training.
# No hyperparameter tuning.
# No threshold optimization.
# No checkpoint selection.
#
# Test partition is opened only after integrity checks.
# Existing final outputs cannot be overwritten.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

MODEL_DIR = ROOT / "model" / "distilbert_seed42"
TOKENIZER_DIR = ROOT / "model" / "tokenizer"

CONFIG_DIR = ROOT / "configs"
TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
REPORTS = ROOT / "outputs" / "reports"

FREEZE_FILE = CONFIG_DIR / "18_final_model_freeze.json"
INTEGRITY_FILE = REPORTS / "00_model_integrity.json"

DATA_DIR = (
    THESIS
    / "dataset_exploration"
    / "outputs"
    / "validated"
)

MANIFEST = DATA_DIR / "54_d2_v11_frozen_split_manifest.csv"
DATA_METADATA = DATA_DIR / "54_d2_v11_split_metadata.json"

PREDICTION_FILE = (
    PREDICTIONS / "01_final_test_predictions.csv"
)

METRICS_FILE = (
    TABLES / "01_final_test_metrics.csv"
)

CONFUSION_FILE = (
    TABLES / "01_final_test_confusion_matrix.csv"
)

CLASS_METRICS_FILE = (
    TABLES / "01_final_test_class_metrics.csv"
)

REPORT_FILE = (
    REPORTS / "01_final_test_report.txt"
)

JSON_FILE = (
    REPORTS / "01_final_test_metrics.json"
)

TEXT_COL = "Requirement Text (Original Requirement)"

EXPECTED_TEST_ROWS = 727
EXPECTED_TEST_FR = 488
EXPECTED_TEST_NFR = 239

MAX_LENGTH = 96
THRESHOLD = 0.5
BATCH_SIZE = 8

LABEL_TO_ID = {"FR": 0, "NFR": 1}
ID_TO_LABEL = {0: "FR", 1: "NFR"}

OUTPUT_FILES = [
    PREDICTION_FILE,
    METRICS_FILE,
    CONFUSION_FILE,
    CLASS_METRICS_FILE,
    REPORT_FILE,
    JSON_FILE,
]


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


def verify_inventory(directory, records):
    require(
        directory.is_dir(),
        f"Directory not found: {directory}",
    )

    expected = set()

    for record in records:
        relative = record["relative_path"]

        path = directory / relative

        require(
            path.is_file(),
            f"Missing file: {path}",
        )

        require(
            path.stat().st_size == int(
                record["size_bytes"]
            ),
            f"File size mismatch: {path}",
        )

        require(
            sha256(path) == record["sha256"],
            f"SHA256 mismatch: {path}",
        )

        expected.add(relative)

    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }

    require(
        actual == expected,
        f"File inventory mismatch: {directory}",
    )


print("=" * 76)
print("FINAL MODEL - SCRIPT 01: FROZEN TEST EVALUATION")
print("=" * 76)

# ------------------------------------------------------------
# 1. Prevent accidental repeated evaluation
# ------------------------------------------------------------

print("\nSTEP 1: ONE-TIME EVALUATION SAFEGUARDS")

for path in OUTPUT_FILES:
    require(
        not path.exists(),
        "Evaluation output already exists. "
        f"Refusing to overwrite: {path}",
    )

require(
    FREEZE_FILE.is_file(),
    "Missing frozen model selection record",
)

require(
    INTEGRITY_FILE.is_file(),
    "Run final_model Script 00 first",
)

freeze = json.loads(
    FREEZE_FILE.read_text(encoding="utf-8")
)

integrity = json.loads(
    INTEGRITY_FILE.read_text(encoding="utf-8")
)

require(
    freeze.get("selection_status") == "FROZEN",
    "Final selection is not frozen",
)

require(
    freeze.get("selected_architecture") == "distilbert",
    "Incorrect selected architecture",
)

require(
    freeze.get("selected_seed") == 42,
    "Incorrect selected seed",
)

require(
    freeze.get("dataset_protocol") == "D2-v1.1",
    "Incorrect dataset protocol",
)

require(
    freeze.get("no_additional_training") is True,
    "Training freeze was not confirmed",
)

require(
    freeze.get("no_threshold_optimization") is True,
    "Threshold freeze was not confirmed",
)

require(
    freeze.get("no_ensemble") is True,
    "Unexpected ensemble configuration",
)

require(
    int(freeze.get("max_length")) == MAX_LENGTH,
    "Token length mismatch",
)

rule = freeze["classification_rule"]

require(
    rule["positive_class"] == "NFR"
    and rule["negative_class"] == "FR",
    "Class-name mismatch",
)

require(
    int(rule["positive_class_id"]) == 1
    and int(rule["negative_class_id"]) == 0,
    "Class-index mismatch",
)

require(
    float(rule["nfr_probability_threshold"])
    == THRESHOLD,
    "Threshold mismatch",
)

require(
    integrity.get("status") == "PASS",
    "Final model integrity check has not passed",
)

require(
    integrity.get("checkpoint_sha256_verified") is True,
    "Checkpoint integrity not confirmed",
)

require(
    integrity.get("test_accessed") is False,
    "Script 00 should not access the test set",
)

print("Model selection: FROZEN")
print("Architecture: DistilBERT")
print("Seed: 42")
print("Threshold: 0.5")
print("Additional training: DISABLED")
print("Output overwrite: DISABLED")

# ------------------------------------------------------------
# 2. Verify model and tokenizer hashes
# ------------------------------------------------------------

print("\nSTEP 2: MODEL AND TOKENIZER INTEGRITY")

verify_inventory(
    MODEL_DIR,
    freeze["checkpoint_files"],
)

verify_inventory(
    TOKENIZER_DIR,
    integrity["tokenizer_files"],
)

print("Frozen model SHA256: PASS")
print("Packaged tokenizer SHA256: PASS")

# ------------------------------------------------------------
# 3. Validate dataset manifest integrity
# ------------------------------------------------------------

print("\nSTEP 3: DATASET INTEGRITY")

require(
    MANIFEST.is_file(),
    "Frozen dataset manifest not found",
)

require(
    DATA_METADATA.is_file(),
    "Frozen dataset metadata not found",
)

data_metadata = json.loads(
    DATA_METADATA.read_text(encoding="utf-8")
)

require(
    data_metadata.get("protocol") == "D2-v1.1",
    "Unexpected dataset metadata protocol",
)

expected_hash = data_metadata[
    "output_sha256"
][MANIFEST.name]

manifest_hash = sha256(MANIFEST)

require(
    manifest_hash == expected_hash,
    "Frozen dataset manifest SHA256 mismatch",
)

print("Dataset: D2-v1.1")
print("Frozen manifest SHA256: PASS")

# ------------------------------------------------------------
# 4. Load the frozen test partition
#
# The test requirements are accessed here for
# final evaluation, not model selection.
# ------------------------------------------------------------

print("\nSTEP 4: LOAD SEALED TEST PARTITION")

columns = [
    "Requirement ID",
    TEXT_COL,
    "label",
    "split",
    "similarity_group",
]

manifest = pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    usecols=columns,
)

require(
    len(manifest) == 4833,
    "Unexpected frozen dataset size",
)

require(
    manifest["Requirement ID"].is_unique,
    "Duplicate requirement IDs",
)

require(
    manifest["label"].isin(["FR", "NFR"]).all(),
    "Invalid dataset labels",
)

require(
    manifest["split"].isin(
        ["train", "validation", "test"]
    ).all(),
    "Unexpected split values",
)

test = manifest.loc[
    manifest["split"] == "test"
].copy().reset_index(drop=True)

require(
    len(test) == EXPECTED_TEST_ROWS,
    "Unexpected test requirement count",
)

require(
    test[TEXT_COL].str.strip().ne("").all(),
    "Blank test requirement text found",
)

require(
    int(test["label"].eq("FR").sum())
    == EXPECTED_TEST_FR,
    "Unexpected test FR count",
)

require(
    int(test["label"].eq("NFR").sum())
    == EXPECTED_TEST_NFR,
    "Unexpected test NFR count",
)

require(
    test["similarity_group"].str.strip().ne("").all(),
    "Missing test similarity group",
)

train_val_groups = set(
    manifest.loc[
        manifest["split"] != "test",
        "similarity_group",
    ]
)

test_groups = set(
    test["similarity_group"]
)

require(
    not (train_val_groups & test_groups),
    "Similarity-group overlap with test detected",
)

print("Test requirements:", len(test))
print("FR:", EXPECTED_TEST_FR)
print("NFR:", EXPECTED_TEST_NFR)
print("Test similarity groups:", len(test_groups))
print("Cross-partition similarity overlap: NONE")

# ------------------------------------------------------------
# 5. Load frozen model
# ------------------------------------------------------------

print("\nSTEP 5: LOAD FINAL MODEL")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    str(TOKENIZER_DIR),
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

require(
    model.config.model_type == "distilbert",
    "Loaded model is not DistilBERT",
)

require(
    model.config.num_labels == 2,
    "Expected exactly two model outputs",
)

model.to(device)
model.eval()

print("Device:", device)
print("Model loaded: PASS")
print("Training: DISABLED")

# ------------------------------------------------------------
# 6. Frozen inference
# ------------------------------------------------------------

print("\nSTEP 6: FINAL TEST INFERENCE")

texts = test[TEXT_COL].tolist()

all_probs = []

with torch.inference_mode():

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        outputs = model(**tokens)

        probabilities = torch.softmax(
            outputs.logits.float(),
            dim=-1,
        )[:, 1]

        all_probs.extend(
            probabilities.cpu().numpy().tolist()
        )

        completed = min(
            start + BATCH_SIZE,
            len(texts),
        )

        if (
            completed % 80 == 0
            or completed == len(texts)
        ):
            print(
                f"Inference completed: "
                f"{completed}/{len(texts)}"
            )

probabilities = np.asarray(
    all_probs,
    dtype=np.float64,
)

require(
    len(probabilities) == len(test),
    "Incomplete inference results",
)

require(
    np.isfinite(probabilities).all(),
    "Non-finite output probabilities",
)

require(
    (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all(),
    "Invalid output probabilities",
)

predicted_ids = (
    probabilities >= THRESHOLD
).astype(int)

true_ids = test["label"].map(
    LABEL_TO_ID
).to_numpy(dtype=int)

# ------------------------------------------------------------
# 7. Calculate metrics
# ------------------------------------------------------------

print("\nSTEP 7: FINAL TEST METRICS")

accuracy = accuracy_score(
    true_ids,
    predicted_ids,
)

balanced_acc = balanced_accuracy_score(
    true_ids,
    predicted_ids,
)

nfr_precision = precision_score(
    true_ids,
    predicted_ids,
    pos_label=1,
    zero_division=0,
)

nfr_recall = recall_score(
    true_ids,
    predicted_ids,
    pos_label=1,
    zero_division=0,
)

nfr_f1 = f1_score(
    true_ids,
    predicted_ids,
    pos_label=1,
    zero_division=0,
)

macro_f1 = f1_score(
    true_ids,
    predicted_ids,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    true_ids,
    predicted_ids,
    average="weighted",
    zero_division=0,
)

pr_auc = average_precision_score(
    true_ids,
    probabilities,
)

roc_auc = roc_auc_score(
    true_ids,
    probabilities,
)

cm = confusion_matrix(
    true_ids,
    predicted_ids,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

metrics = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "nfr_precision": float(nfr_precision),
    "nfr_recall": float(nfr_recall),
    "nfr_f1": float(nfr_f1),
    "average_precision": float(pr_auc),
    "roc_auc": float(roc_auc),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_acc:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")
print(f"NFR precision:     {nfr_precision:.4f}")
print(f"NFR recall:        {nfr_recall:.4f}")
print(f"NFR F1:            {nfr_f1:.4f}")
print(f"Average precision: {pr_auc:.4f}")
print(f"ROC AUC:           {roc_auc:.4f}")

print("\nCONFUSION MATRIX")
print("Rows = true labels; columns = predicted labels")
print("             Pred FR   Pred NFR")
print(f"True FR      {tn:7d}   {fp:8d}")
print(f"True NFR     {fn:7d}   {tp:8d}")

# ------------------------------------------------------------
# 8. Save final predictions
# ------------------------------------------------------------

print("\nSTEP 8: SAVE FINAL OUTPUTS")

for directory in [TABLES, PREDICTIONS, REPORTS]:
    directory.mkdir(parents=True, exist_ok=True)

predictions_df = pd.DataFrame({
    "requirement_id": test["Requirement ID"],
    "similarity_group": test["similarity_group"],
    "true_label": test["label"],
    "predicted_label": [
        ID_TO_LABEL[int(value)]
        for value in predicted_ids
    ],
    "nfr_probability": probabilities,
    "correct": predicted_ids == true_ids,
})

predictions_df.to_csv(
    PREDICTION_FILE,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame([metrics]).to_csv(
    METRICS_FILE,
    index=False,
)

pd.DataFrame(
    cm,
    index=["True_FR", "True_NFR"],
    columns=["Pred_FR", "Pred_NFR"],
).to_csv(CONFUSION_FILE)

class_report = classification_report(
    true_ids,
    predicted_ids,
    labels=[0, 1],
    target_names=["FR", "NFR"],
    output_dict=True,
    zero_division=0,
)

class_metrics = []

for label in ["FR", "NFR"]:
    class_metrics.append({
        "class": label,
        **class_report[label],
    })

pd.DataFrame(class_metrics).to_csv(
    CLASS_METRICS_FILE,
    index=False,
)

# ------------------------------------------------------------
# 9. Save reproducibility metadata
# ------------------------------------------------------------

evaluation_metadata = {
    "experiment": "FINAL_MODEL_01",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "status": "COMPLETED",

    "dataset_protocol": "D2-v1.1",
    "dataset_manifest_sha256": manifest_hash,

    "selected_architecture": "distilbert",
    "selected_seed": 42,

    "model_checkpoint": str(MODEL_DIR),
    "checkpoint_integrity": "VERIFIED",

    "tokenizer_directory": str(TOKENIZER_DIR),
    "tokenizer_integrity": "VERIFIED",

    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "threshold": THRESHOLD,

    "positive_label": "NFR",
    "positive_class_id": 1,

    "test_requirements": len(test),
    "test_fr": EXPECTED_TEST_FR,
    "test_nfr": EXPECTED_TEST_NFR,
    "test_similarity_groups": len(test_groups),

    "device": str(device),
    "torch_version": torch.__version__,
    "python_version": sys.version.split()[0],

    "metrics": metrics,

    "prediction_file_sha256": sha256(
        PREDICTION_FILE
    ),

    "training_performed": False,
    "threshold_tuned_on_test": False,
    "model_selected_using_test": False,
    "test_evaluation_performed": True,

    "research_limitations": [
        "Architecture selected after repeated validation reuse",
        "Only three seeds used for architecture stability",
        "Test sample may contain related project requirements",
        "Project-level independence was not established",
        "Original dataset labels were preserved",
        "Independent annotation adjudication incomplete",
    ],
}

JSON_FILE.write_text(
    json.dumps(
        evaluation_metadata,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

report_lines = [
    "FINAL MODEL TEST EVALUATION",
    "=" * 76,
    "",
    "MODEL CONFIGURATION",
    "Architecture: DistilBERT",
    "Seed: 42",
    "Max sequence length: 96",
    "NFR threshold: 0.5",
    "Checkpoint: FROZEN",
    "",
    "DATASET",
    "Protocol: D2-v1.1",
    f"Test requirements: {len(test)}",
    f"FR requirements: {EXPECTED_TEST_FR}",
    f"NFR requirements: {EXPECTED_TEST_NFR}",
    f"Similarity groups: {len(test_groups)}",
    "Cross-partition group overlap: NONE",
    "",
    "FINAL TEST RESULTS",
    *[
        f"{key}: {value:.6f}"
        if isinstance(value, float)
        else f"{key}: {value}"
        for key, value in metrics.items()
    ],
    "",
    "CLASSIFICATION REPORT",
    classification_report(
        true_ids,
        predicted_ids,
        labels=[0, 1],
        target_names=["FR", "NFR"],
        digits=4,
        zero_division=0,
    ),
    "",
    "CONFUSION MATRIX",
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    "",
    "RESEARCH INTEGRITY",
    "Original checkpoint verified: YES",
    "Dataset manifest SHA256 verified: YES",
    "Threshold optimization: NO",
    "Additional training: NO",
    "Model selection using test results: NO",
    "",
    "LIMITATIONS",
    "- Single frozen test partition.",
    "- Architecture selection reused validation data.",
    "- Project-level independence not established.",
    "- Dataset annotation concerns remain unresolved.",
    "- No independent human label adjudication.",
    "",
    "RESEARCH STATUS",
    "Frozen test evaluation: COMPLETED",
    "Model selection: LOCKED",
    "Additional tuning based on test results: PROHIBITED",
]

REPORT_FILE.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("Predictions:", PREDICTION_FILE)
print("Metrics:", METRICS_FILE)
print("Class metrics:", CLASS_METRICS_FILE)
print("Confusion matrix:", CONFUSION_FILE)
print("Report:", REPORT_FILE)
print("Metadata:", JSON_FILE)

print("\nFINAL RESEARCH STATUS")
print("Model selection: LOCKED")
print("Frozen test evaluation: COMPLETED")
print("Additional training: NO")
print("Threshold optimization: NO")
print("Final test predictions: SAVED")
print("Final test metrics: SAVED")

print("\nScript 01 completed successfully.")
