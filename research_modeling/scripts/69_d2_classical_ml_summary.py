
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

# ============================================================
# SCRIPT 69: CONSOLIDATED CLASSICAL ML BASELINE SUMMARY
#
# - Collect results from Scripts 57-68
# - Compare 17 fixed model configurations
# - Merge train-only CV and validation metrics
# - Preserve original experiment results
# - Do not train or evaluate any models
# - Do not access the frozen test partition
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "research_modeling/outputs"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

DATA = ROOT / "dataset_exploration/outputs/validated"

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
SPLIT_META = DATA / "54_d2_v11_split_metadata.json"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "experiment": 57,
        "family": "Logistic Regression",
        "validation": "57_d2_baseline_metrics.csv",
        "cv": "60_d2_cv_stability.csv",
    },
    {
        "experiment": 62,
        "family": "Linear SVM",
        "validation": "62_d2_svm_validation_metrics.csv",
        "cv": "62_d2_svm_cv_stability.csv",
    },
    {
        "experiment": 64,
        "family": "Naive Bayes / SGD",
        "validation": "64_d2_ml_validation_metrics.csv",
        "cv": "64_d2_ml_cv_stability.csv",
    },
    {
        "experiment": 65,
        "family": "RBF SVM",
        "validation": "65_d2_rbf_validation_metrics.csv",
        "cv": "65_d2_rbf_cv_stability.csv",
    },
    {
        "experiment": 66,
        "family": "Feature Representation",
        "validation": "66_d2_validation_metrics.csv",
        "cv": "66_d2_cv_stability.csv",
    },
    {
        "experiment": 68,
        "family": "Tree Ensembles",
        "validation": "68_d2_tree_validation_metrics.csv",
        "cv": "68_d2_tree_cv_stability.csv",
    },
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


# ------------------------------------------------------------
# 1. Verify dataset protocol
# ------------------------------------------------------------

meta = json.loads(
    SPLIT_META.read_text(encoding="utf-8")
)

if meta.get("protocol") != "D2-v1.1":
    raise RuntimeError("Unexpected dataset protocol")

if sha256(MANIFEST) != meta["output_sha256"][MANIFEST.name]:
    raise RuntimeError("Frozen manifest hash mismatch")

# ------------------------------------------------------------
# 2. Gather existing validation and CV results
# ------------------------------------------------------------

all_rows = []
input_hashes = {}

for source in SOURCES:

    validation_path = TABLES / source["validation"]
    cv_path = TABLES / source["cv"]

    for path in (validation_path, cv_path):
        if not path.exists():
            raise FileNotFoundError(path)

        input_hashes[path.name] = sha256(path)

    validation = pd.read_csv(validation_path)
    cv = pd.read_csv(cv_path)

    if not validation["model"].is_unique:
        raise RuntimeError(
            f"Duplicate validation models: {validation_path.name}"
        )

    if not cv["model"].is_unique:
        raise RuntimeError(
            f"Duplicate CV models: {cv_path.name}"
        )

    if set(validation["model"]) != set(cv["model"]):
        raise RuntimeError(
            f"CV and validation models differ: "
            f"{source['experiment']}"
        )

    merged = validation.merge(
        cv,
        on="model",
        how="inner",
        validate="one_to_one",
    )

    merged["experiment"] = source["experiment"]
    merged["family"] = source["family"]

    all_rows.append(merged)

leaderboard = pd.concat(
    all_rows,
    ignore_index=True,
)

if len(leaderboard) != 17:
    raise RuntimeError(
        f"Expected 17 configurations, got {len(leaderboard)}"
    )

if not leaderboard["model"].is_unique:
    raise RuntimeError("Duplicate model names")

# ------------------------------------------------------------
# 3. Select common reporting columns
# ------------------------------------------------------------

columns = [
    "experiment",
    "family",
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "nfr_precision",
    "nfr_recall",
    "nfr_f1",
    "average_precision",
    "roc_auc",
    "fp",
    "fn",
    "mean_nfr_f1",
    "std_nfr_f1",
    "mean_nfr_recall",
    "std_nfr_recall",
    "mean_macro_f1",
    "mean_balanced_accuracy",
]

missing = set(columns) - set(leaderboard.columns)

if missing:
    raise RuntimeError(
        f"Missing reporting columns: {sorted(missing)}"
    )

leaderboard = leaderboard[columns]

leaderboard = leaderboard.sort_values(
    ["nfr_f1", "macro_f1"],
    ascending=[False, False],
).reset_index(drop=True)

leaderboard.insert(
    0,
    "validation_rank",
    range(1, len(leaderboard) + 1),
)

# ------------------------------------------------------------
# 4. Save comparative tables
# ------------------------------------------------------------

leaderboard_path = (
    TABLES / "69_d2_classical_ml_leaderboard.csv"
)

cv_path = (
    TABLES / "69_d2_classical_ml_cv_comparison.csv"
)

leaderboard.to_csv(
    leaderboard_path,
    index=False,
)

cv_columns = [
    "model",
    "family",
    "mean_nfr_f1",
    "std_nfr_f1",
    "mean_nfr_recall",
    "std_nfr_recall",
    "mean_macro_f1",
    "mean_balanced_accuracy",
]

cv_summary = leaderboard[cv_columns].sort_values(
    "mean_nfr_f1",
    ascending=False,
)

cv_summary.to_csv(
    cv_path,
    index=False,
)

# ------------------------------------------------------------
# 5. Print summary
# ------------------------------------------------------------

print("=" * 76)
print("SCRIPT 69: CLASSICAL ML BASELINE CONSOLIDATION")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements: 3378")
print("Validation requirements: 728")
print("Frozen test requirements: 727")
print("Frozen manifest integrity: PASS")
print("Test evaluation: NOT PERFORMED")

print("\nEXPERIMENT COVERAGE")
print("Model configurations:", len(leaderboard))
print("Model groups:", len(SOURCES))
print("New model training: NO")
print("New validation evaluation: NO")

display = [
    "validation_rank",
    "model",
    "nfr_f1",
    "nfr_recall",
    "macro_f1",
    "balanced_accuracy",
    "mean_nfr_f1",
]

print("\nVALIDATION LEADERBOARD")
print(
    leaderboard[display]
    .round(4)
    .to_string(index=False)
)

print("\nTOP FIVE CROSS-VALIDATION CONFIGURATIONS")
print(
    cv_summary.head(5)
    .round(4)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 6. Save methodology and research record
# ------------------------------------------------------------

report_path = (
    REPORTS / "69_d2_classical_ml_summary.txt"
)

metadata_path = (
    REPORTS / "69_d2_classical_ml_summary.json"
)

best = leaderboard.iloc[0]

report = [
    "D2 v1.1 CLASSICAL MACHINE LEARNING SUMMARY",
    "=" * 76,
    "",
    "DATASET",
    "Total requirements: 4833",
    "Train: 3378",
    "Validation: 728",
    "Test: 727 (sealed)",
    "",
    "EVALUATION PROTOCOL",
    "Train-only five-fold similarity-group-aware CV",
    "Independent frozen validation partition",
    "No test evaluation",
    "Validation has been reused for model development",
    "",
    "VALIDATION LEADERBOARD",
    leaderboard[display]
    .round(4)
    .to_string(index=False),
    "",
    "CROSS-VALIDATION COMPARISON",
    cv_summary.round(4).to_string(index=False),
    "",
    "CURRENT LEADING VALIDATION CONFIGURATION",
    f"Model: {best['model']}",
    f"NFR F1: {best['nfr_f1']:.4f}",
    f"NFR recall: {best['nfr_recall']:.4f}",
    f"Macro-F1: {best['macro_f1']:.4f}",
    "",
    "METHODOLOGICAL LIMITATIONS",
    "- Validation results are development-stage evidence.",
    "- Models were compared repeatedly on validation.",
    "- No independent-project evaluation.",
    "- No independently verified annotation review.",
    "- Some models use different feature representations.",
    "- CV results are based on one five-fold partition.",
    "- Final test evaluation has not occurred.",
    "",
    "NEXT STAGE",
    "Compare selected neural/transformer baselines.",
    "Preserve D2 v1.1 and the test partition.",
    "Finalize model selection before test evaluation.",
]

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

metadata_path.write_text(
    json.dumps(
        {
            "experiment": 69,
            "protocol": "D2-v1.1",
            "generated_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "manifest_sha256": sha256(MANIFEST),
            "configurations": len(leaderboard),
            "source_sha256": input_hashes,
            "evaluation": "EXISTING_VALIDATION_RESULTS",
            "test_accessed": False,
            "new_model_training": False,
            "model_selection": "PENDING",
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Classical ML consolidation: COMPLETED")
print("Dataset protocol: FROZEN")
print("Model training in Script 69: NONE")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
print(leaderboard_path)
print(cv_path)
print(report_path)
print(metadata_path)

print("\nScript 69 completed.")
