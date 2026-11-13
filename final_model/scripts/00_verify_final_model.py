
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ============================================================
# FINAL MODEL - SCRIPT 00
# INTEGRITY AND LOADING VERIFICATION
#
# Selected model : DistilBERT
# Seed           : 42
# Dataset        : D2-v1.1
# Positive label : NFR
#
# Verify:
# 1. Frozen selection record
# 2. Copied checkpoint SHA256 hashes
# 3. Model architecture and classification configuration
# 4. Local tokenizer availability
# 5. Successful model loading
# 6. Research provenance
#
# No training.
# No test access.
# No prediction generation.
# No threshold optimization.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

MODEL_DIR = ROOT / "model" / "distilbert_seed42"
TOKENIZER_DIR = ROOT / "model" / "tokenizer"

FREEZE_FILE = (
    ROOT / "configs" / "18_final_model_freeze.json"
)

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_FILE = REPORT_DIR / "00_model_integrity_report.txt"
JSON_FILE = REPORT_DIR / "00_model_integrity.json"

EXPECTED_ORIGINAL = (
    THESIS / "deep_learning_research"
    / "checkpoints" / "05_distilbert_best"
)

EXPECTED_MODEL = "distilbert"
EXPECTED_SEED = 42
EXPECTED_PROTOCOL = "D2-v1.1"
EXPECTED_MAX_LENGTH = 96
EXPECTED_THRESHOLD = 0.5
EXPECTED_TOKENIZER = (
    "distilbert/distilbert-base-uncased"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def list_files(directory):
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file()
    )


def verify_checkpoint_hashes(freeze):
    expected_files = freeze.get("checkpoint_files")

    require(
        isinstance(expected_files, list)
        and bool(expected_files),
        "Frozen checkpoint inventory is missing",
    )

    expected_names = set()

    for record in expected_files:
        name = record["relative_path"]
        expected_hash = record["sha256"]
        expected_size = int(record["size_bytes"])

        path = MODEL_DIR / name

        require(
            path.is_file(),
            f"Missing frozen checkpoint file: {name}",
        )

        require(
            path.stat().st_size == expected_size,
            f"File size mismatch: {name}",
        )

        actual_hash = sha256(path)

        require(
            actual_hash == expected_hash,
            f"SHA256 mismatch: {name}",
        )

        expected_names.add(name)

    actual_names = {
        str(p.relative_to(MODEL_DIR))
        for p in list_files(MODEL_DIR)
    }

    require(
        actual_names == expected_names,
        "Copied checkpoint file inventory differs "
        "from frozen inventory",
    )

    return expected_files


print("=" * 76)
print("FINAL MODEL - SCRIPT 00: INTEGRITY VERIFICATION")
print("=" * 76)

# ------------------------------------------------------------
# STEP 1. Verify required directories and freeze manifest
# ------------------------------------------------------------

print("\nSTEP 1: PROJECT STRUCTURE")

require(
    MODEL_DIR.is_dir(),
    f"Missing model directory: {MODEL_DIR}",
)

require(
    FREEZE_FILE.is_file(),
    f"Missing freeze manifest: {FREEZE_FILE}",
)

freeze = json.loads(
    FREEZE_FILE.read_text(encoding="utf-8")
)

require(
    freeze.get("selection_status") == "FROZEN",
    "Model selection is not frozen",
)

require(
    freeze.get("dataset_protocol") == EXPECTED_PROTOCOL,
    "Unexpected dataset protocol",
)

require(
    freeze.get("selected_architecture") == EXPECTED_MODEL,
    "Unexpected selected architecture",
)

require(
    freeze.get("selected_seed") == EXPECTED_SEED,
    "Unexpected selected seed",
)

require(
    freeze.get("test_accessed") is False,
    "Freeze record does not confirm sealed test",
)

require(
    freeze.get("checkpoint_directory")
    == str(EXPECTED_ORIGINAL),
    "Unexpected original checkpoint provenance",
)

print("Final model directory: FOUND")
print("Freeze manifest: FOUND")
print("Model selection: FROZEN")
print("Dataset protocol:", EXPECTED_PROTOCOL)

# ------------------------------------------------------------
# STEP 2. Verify all frozen checkpoint hashes
# ------------------------------------------------------------

print("\nSTEP 2: CHECKPOINT INTEGRITY")

inventory = verify_checkpoint_hashes(freeze)

print("Checkpoint files:", len(inventory))
print("File sizes: PASS")
print("SHA256 verification: PASS")

# ------------------------------------------------------------
# STEP 3. Verify inference configuration
# ------------------------------------------------------------

print("\nSTEP 3: INFERENCE CONFIGURATION")

require(
    int(freeze["max_length"]) == EXPECTED_MAX_LENGTH,
    "Unexpected maximum token length",
)

require(
    freeze["tokenizer_checkpoint"] == EXPECTED_TOKENIZER,
    "Unexpected tokenizer identifier",
)

rule = freeze.get("classification_rule", {})

require(
    rule.get("positive_class") == "NFR",
    "Unexpected positive class",
)

require(
    rule.get("negative_class") == "FR",
    "Unexpected negative class",
)

require(
    rule.get("positive_class_id") == 1,
    "Unexpected NFR class ID",
)

require(
    rule.get("negative_class_id") == 0,
    "Unexpected FR class ID",
)

require(
    float(rule["nfr_probability_threshold"])
    == EXPECTED_THRESHOLD,
    "Unexpected classification threshold",
)

config = AutoConfig.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

require(
    config.model_type == EXPECTED_MODEL,
    "Loaded configuration is not DistilBERT",
)

require(
    config.num_labels == 2,
    "Expected exactly two classification labels",
)

# Verify explicit checkpoint label declarations,
# when the configuration contains FR/NFR names.
label2id = getattr(config, "label2id", {})

if "FR" in label2id or "NFR" in label2id:
    require(
        label2id.get("FR") == 0
        and label2id.get("NFR") == 1,
        "Checkpoint label mapping mismatch",
    )

print("Architecture:", config.model_type)
print("Number of labels:", config.num_labels)
print("Tokenizer:", EXPECTED_TOKENIZER)
print("Maximum sequence length:", EXPECTED_MAX_LENGTH)
print("Positive class: NFR (ID 1)")
print("Negative class: FR (ID 0)")
print("Threshold:", EXPECTED_THRESHOLD)
print("Configuration: PASS")

# ------------------------------------------------------------
# STEP 4. Verify and package tokenizer
#
# Prefer an already packaged local tokenizer.
# If absent, use locally cached tokenizer files.
# No internet downloads are permitted.
#
# Tokenizer is stored separately from the frozen
# checkpoint to preserve its original file hashes.
# ------------------------------------------------------------

print("\nSTEP 4: TOKENIZER VERIFICATION")

if TOKENIZER_DIR.is_dir() and list_files(TOKENIZER_DIR):

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
    )

    tokenizer_source = "FINAL_MODEL_LOCAL"

else:

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        tokenizer_source = "COPIED_CHECKPOINT"

    except (OSError, ValueError):

        tokenizer = AutoTokenizer.from_pretrained(
            EXPECTED_TOKENIZER,
            local_files_only=True,
        )

        tokenizer_source = "LOCAL_HF_CACHE"

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save_pretrained(str(TOKENIZER_DIR))

# Reload independently from the final folder,
# proving the packaged tokenizer is loadable.

packaged_tokenizer = AutoTokenizer.from_pretrained(
    str(TOKENIZER_DIR),
    local_files_only=True,
)

require(
    packaged_tokenizer.is_fast,
    "Expected a fast tokenizer",
)

probe = packaged_tokenizer(
    "The system shall generate monthly reports.",
    truncation=True,
    max_length=EXPECTED_MAX_LENGTH,
    padding=False,
)

require(
    len(probe["input_ids"]) > 0,
    "Tokenizer produced an empty token sequence",
)

require(
    len(probe["input_ids"]) <= EXPECTED_MAX_LENGTH,
    "Tokenizer exceeded maximum length",
)

tokenizer_inventory = [
    {
        "relative_path": str(
            p.relative_to(TOKENIZER_DIR)
        ),
        "size_bytes": p.stat().st_size,
        "sha256": sha256(p),
    }
    for p in list_files(TOKENIZER_DIR)
]

print("Tokenizer source:", tokenizer_source)
print("Packaged tokenizer:", TOKENIZER_DIR)
print("Tokenizer files:", len(tokenizer_inventory))
print("Local tokenizer loading: PASS")

# ------------------------------------------------------------
# STEP 5. Load model without test inference
# ------------------------------------------------------------

print("\nSTEP 5: MODEL LOADING")

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

model.eval()

require(
    model.config.model_type == EXPECTED_MODEL,
    "Loaded model architecture mismatch",
)

require(
    model.config.num_labels == 2,
    "Loaded model output labels mismatch",
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

require(
    parameter_count > 0,
    "Model has no parameters",
)

require(
    all(
        not p.requires_grad or p.numel() > 0
        for p in model.parameters()
    ),
    "Invalid model parameters",
)

print("Model loading: PASS")
print("Model architecture:", model.config.model_type)
print("Number of parameters:", f"{parameter_count:,}")
print("Inference mode: EVAL")

del model

# ------------------------------------------------------------
# STEP 6. Save verification report
# ------------------------------------------------------------

print("\nSTEP 6: SAVE VERIFICATION REPORT")

result = {
    "script": "FINAL_MODEL_00",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "status": "PASS",

    "dataset_protocol": EXPECTED_PROTOCOL,

    "selected_architecture": EXPECTED_MODEL,
    "selected_seed": EXPECTED_SEED,

    "checkpoint_directory": str(MODEL_DIR),
    "checkpoint_file_count": len(inventory),
    "checkpoint_files": inventory,

    "checkpoint_sha256_verified": True,

    "tokenizer_directory": str(TOKENIZER_DIR),
    "tokenizer_source": tokenizer_source,
    "tokenizer_files": tokenizer_inventory,

    "max_length": EXPECTED_MAX_LENGTH,
    "nfr_threshold": EXPECTED_THRESHOLD,

    "positive_label": "NFR",
    "positive_label_id": 1,

    "model_parameters": parameter_count,

    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,

    "training_performed": False,
    "test_accessed": False,
    "test_evaluation_performed": False,
}

JSON_FILE.write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)

report = [
    "FINAL MODEL: INTEGRITY VERIFICATION",
    "=" * 76,
    "",
    "STATUS: PASS",
    "",
    "MODEL",
    "Architecture: DistilBERT",
    "Seed: 42",
    f"Checkpoint: {MODEL_DIR}",
    f"Parameters: {parameter_count:,}",
    "",
    "CHECKPOINT INTEGRITY",
    f"Checkpoint files verified: {len(inventory)}",
    "All frozen SHA256 hashes: MATCH",
    "",
    "TOKENIZER",
    f"Source: {tokenizer_source}",
    f"Directory: {TOKENIZER_DIR}",
    f"Packaged files: {len(tokenizer_inventory)}",
    "",
    "INFERENCE CONFIGURATION",
    "Dataset: D2-v1.1",
    "Maximum token length: 96",
    "FR: class 0",
    "NFR: class 1",
    "NFR probability threshold: 0.5",
    "",
    "MODEL LOAD",
    "Status: PASS",
    "Mode: EVAL",
    "",
    "RESEARCH STATUS",
    "Final model verification: COMPLETED",
    "Model training: NO",
    "Test accessed: NO",
    "Test evaluation: PENDING",
    "",
    "The frozen checkpoint remains unchanged.",
    "The tokenizer is packaged separately.",
]

REPORT_FILE.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("JSON report:", JSON_FILE)
print("Text report:", REPORT_FILE)

print("\nFINAL VERIFICATION STATUS")
print("Checkpoint integrity: PASS")
print("Model configuration: PASS")
print("Tokenizer packaging: PASS")
print("Model loading: PASS")
print("Training performed: NO")
print("Test set: SEALED")
print("Test evaluation: PENDING")

print("\nScript 00 completed successfully.")
