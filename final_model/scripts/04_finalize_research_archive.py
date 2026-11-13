
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

# ============================================================
# FINAL MODEL - SCRIPT 04
# RESEARCH ARCHIVE AND INTEGRITY AUDIT
#
# No model loading.
# No training or inference.
# No test-set re-evaluation.
# No modification of previous results.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "model/distilbert_seed42"
TOKENIZER = ROOT / "model/tokenizer"
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"

FREEZE = CONFIGS / "18_final_model_freeze.json"
INTEGRITY = REPORTS / "00_model_integrity.json"
EVALUATION = REPORTS / "01_final_test_metrics.json"
FINAL_REPORT = REPORTS / "03_final_evaluation_full_report.txt"

ARCHIVE = REPORTS / "04_final_research_archive.json"
ARCHIVE_REPORT = REPORTS / "04_archive_integrity_report.txt"

REQUIRED = [
    FREEZE,
    INTEGRITY,
    EVALUATION,
    FINAL_REPORT,
    OUTPUTS / "predictions/01_final_test_predictions.csv",
    OUTPUTS / "tables/01_final_test_metrics.csv",
    OUTPUTS / "tables/01_final_test_class_metrics.csv",
    OUTPUTS / "tables/01_final_test_confusion_matrix.csv",
    OUTPUTS / "tables/03_validation_test_comparison.csv",
    OUTPUTS / "tables/03_final_artifact_inventory.csv",
    REPORTS / "02_figure_generation_report.txt",
    REPORTS / "03_final_report_metadata.json",
]

FIGURE_NAMES = [
    "02_confusion_matrix",
    "02_precision_recall_curve",
    "02_roc_curve",
    "02_per_class_metrics",
]

for name in FIGURE_NAMES:
    for extension in ["png", "pdf"]:
        REQUIRED.append(
            OUTPUTS / "figures" / f"{name}.{extension}"
        )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


print("=" * 76)
print("FINAL MODEL - SCRIPT 04: RESEARCH ARCHIVE")
print("=" * 76)

# ------------------------------------------------------------
# 1. Verify required files
# ------------------------------------------------------------

require(
    not ARCHIVE.exists() and not ARCHIVE_REPORT.exists(),
    "Archive outputs already exist. Refusing overwrite.",
)

for path in REQUIRED:
    require(
        path.is_file(),
        f"Missing required artifact: {path}",
    )

require(MODEL.is_dir(), "Missing final model")
require(TOKENIZER.is_dir(), "Missing tokenizer")

freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
integrity = json.loads(
    INTEGRITY.read_text(encoding="utf-8")
)
evaluation = json.loads(
    EVALUATION.read_text(encoding="utf-8")
)

require(
    freeze["selection_status"] == "FROZEN",
    "Model is not frozen",
)

require(
    freeze["selected_architecture"] == "distilbert",
    "Unexpected model",
)

require(
    int(freeze["selected_seed"]) == 42,
    "Unexpected seed",
)

require(
    integrity["status"] == "PASS",
    "Integrity verification not passed",
)

require(
    evaluation["status"] == "COMPLETED",
    "Test evaluation incomplete",
)

require(
    evaluation["test_evaluation_performed"] is True,
    "Missing test evaluation confirmation",
)

# ------------------------------------------------------------
# 2. Verify frozen checkpoint hashes
# ------------------------------------------------------------

for record in freeze["checkpoint_files"]:
    file = MODEL / record["relative_path"]

    require(file.is_file(), f"Missing checkpoint file: {file}")

    require(
        file.stat().st_size == int(record["size_bytes"]),
        f"Checkpoint size mismatch: {file}",
    )

    require(
        sha256(file) == record["sha256"],
        f"Checkpoint SHA256 mismatch: {file}",
    )

# Verify frozen tokenizer package.
for record in integrity["tokenizer_files"]:
    file = TOKENIZER / record["relative_path"]

    require(
        file.is_file(),
        f"Missing tokenizer file: {file}",
    )

    require(
        file.stat().st_size == int(record["size_bytes"]),
        f"Tokenizer size mismatch: {file}",
    )

    require(
        sha256(file) == record["sha256"],
        f"Tokenizer SHA256 mismatch: {file}",
    )

# Verify saved prediction artifact without reading its rows.
prediction_path = (
    OUTPUTS / "predictions/01_final_test_predictions.csv"
)

require(
    sha256(prediction_path)
    == evaluation["prediction_file_sha256"],
    "Final prediction SHA256 mismatch",
)

print("\nINTEGRITY CHECKS")
print("Required research artifacts: PASS")
print("Model freeze: PASS")
print("Checkpoint hashes: PASS")
print("Tokenizer hashes: PASS")
print("Final prediction hash: PASS")

# ------------------------------------------------------------
# 3. Build the archive inventory
# ------------------------------------------------------------

# Archive inventory excludes its own output files.
excluded = {ARCHIVE, ARCHIVE_REPORT}

files = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file() and path not in excluded
    and "__pycache__" not in path.parts
)

records = []

for file in files:
    records.append({
        "relative_path": str(file.relative_to(ROOT)),
        "size_bytes": file.stat().st_size,
        "sha256": sha256(file),
    })

# ------------------------------------------------------------
# 4. Write archive metadata
# ------------------------------------------------------------

archive_data = {
    "archive": "FINAL_DISTILBERT_D2_V11",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": "VERIFIED",
    "model": "DistilBERT",
    "seed": 42,
    "dataset": "D2-v1.1",
    "test_requirements": 727,
    "checkpoint_frozen": True,
    "test_evaluation_completed": True,
    "model_retrained": False,
    "post_test_tuning": False,
    "inventory_file_count": len(records),
    "files": records,
}

ARCHIVE.write_text(
    json.dumps(archive_data, indent=2) + "\n",
    encoding="utf-8",
)

report = [
    "FINAL MODEL RESEARCH ARCHIVE",
    "=" * 76,
    "",
    "Model: DistilBERT",
    "Seed: 42",
    "Dataset: D2-v1.1",
    "Evaluation: Completed",
    "",
    "INTEGRITY",
    "Frozen checkpoint: VERIFIED",
    "Tokenizer: VERIFIED",
    "Prediction hash: VERIFIED",
    "",
    f"Files inventoried: {len(records)}",
    "",
    "FINAL TEST RESULTS",
    "Accuracy: 0.8927",
    "Balanced accuracy: 0.8870",
    "Macro F1: 0.8804",
    "NFR precision: 0.8157",
    "NFR recall: 0.8703",
    "NFR F1: 0.8421",
    "",
    "RESEARCH STATUS",
    "Final evaluation: COMPLETED",
    "Additional inference: NO",
    "Additional training: NO",
    "Post-test tuning: NO",
    "",
    "ARCHIVE MANIFEST",
    str(ARCHIVE),
]

ARCHIVE_REPORT.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\nARCHIVE SUMMARY")
print("Files inventoried:", len(records))
print("Archive status: VERIFIED")

print("\nRESEARCH STATUS")
print("Final research archive: COMPLETED")
print("Model selection: LOCKED")
print("Training: NO")
print("Inference: NO")
print("Post-test tuning: NO")

print("\nFILES GENERATED")
print(ARCHIVE)
print(ARCHIVE_REPORT)

print("\nScript 04 completed.")
