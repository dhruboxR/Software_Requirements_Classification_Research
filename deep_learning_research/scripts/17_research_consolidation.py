
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

# ============================================================
# SCRIPT 17: DEEP LEARNING RESEARCH CONSOLIDATION
#
# No training.
# No model checkpoint modifications.
# No test-set access.
# No final model selection.
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"

SEED_RESULTS = TABLES / "09_transformer_seed_results.csv"
SEED_STABILITY = TABLES / "09_transformer_seed_stability.csv"
PREDICTION_STABILITY = (
    TABLES / "10_multiseed_prediction_stability.csv"
)
ERROR_SUMMARY = TABLES / "11_error_category_summary.csv"
ANNOTATION_SUMMARY = TABLES / "15_annotation_agreement_summary.csv"
AUDIT_SUMMARY = TABLES / "16_audit_summary.csv"

OUTPUT_REPORT = REPORTS / "17_research_consolidation.txt"
OUTPUT_JSON = REPORTS / "17_research_consolidation.json"
OUTPUT_TABLE = TABLES / "17_transformer_comparison.csv"
OUTPUT_INVENTORY = TABLES / "17_research_source_inventory.csv"

SOURCES = {
    "transformer_seed_results": SEED_RESULTS,
    "transformer_seed_stability": SEED_STABILITY,
    "prediction_stability": PREDICTION_STABILITY,
    "error_summary": ERROR_SUMMARY,
    "annotation_summary": ANNOTATION_SUMMARY,
    "audit_summary": AUDIT_SUMMARY,
}

ARCHITECTURES = {"distilbert", "bert", "roberta"}
SEEDS = {42, 17, 29}


def verify(condition, message):
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


def read_csv(path):
    verify(path.is_file(), f"Missing source: {path}")
    return pd.read_csv(path)


def format_table(frame):
    return frame.round(4).to_string(index=False)


print("=" * 76)
print("SCRIPT 17: DEEP LEARNING RESEARCH CONSOLIDATION")
print("=" * 76)

# ------------------------------------------------------------
# 1. Load and verify essential research outputs
# ------------------------------------------------------------

loaded = {
    name: read_csv(path)
    for name, path in SOURCES.items()
}

runs = loaded["transformer_seed_results"].copy()
stability = loaded["transformer_seed_stability"].copy()
prediction = loaded["prediction_stability"].copy()
errors = loaded["error_summary"].copy()
annotation = loaded["annotation_summary"].copy()
audit = loaded["audit_summary"].copy()

run_required = {
    "architecture",
    "seed",
    "best_epoch",
    "nfr_f1",
    "nfr_recall",
    "macro_f1",
    "fp",
    "fn",
}

stability_required = {
    "architecture",
    "seed_count",
    "nfr_f1_mean",
    "nfr_f1_std",
    "nfr_recall_mean",
    "nfr_recall_std",
    "macro_f1_mean",
    "macro_f1_std",
}

prediction_required = {
    "architecture",
    "requirements",
    "unanimous_predictions",
    "changing_predictions",
    "always_correct",
    "always_wrong",
    "majority_vote_nfr_f1_descriptive",
}

verify(
    run_required.issubset(runs.columns),
    "Incomplete Script 09 run table",
)

verify(
    stability_required.issubset(stability.columns),
    "Incomplete Script 09 stability table",
)

verify(
    prediction_required.issubset(prediction.columns),
    "Incomplete Script 10 stability table",
)

verify(
    len(runs) == 9,
    "Expected nine transformer runs",
)

verify(
    not runs.duplicated(["architecture", "seed"]).any(),
    "Duplicate transformer run",
)

verify(
    set(runs["architecture"]) == ARCHITECTURES,
    "Unexpected architectures",
)

for architecture in ARCHITECTURES:
    subset = runs.loc[
        runs["architecture"] == architecture
    ]

    verify(
        set(subset["seed"].astype(int)) == SEEDS,
        f"Incomplete seed coverage: {architecture}",
    )

verify(
    set(stability["architecture"]) == ARCHITECTURES,
    "Invalid stability architectures",
)

verify(
    set(prediction["architecture"]) == ARCHITECTURES,
    "Invalid prediction stability architectures",
)

verify(
    (stability["seed_count"].astype(int) == 3).all(),
    "Unexpected seed count",
)

verify(
    (prediction["requirements"].astype(int) == 728).all(),
    "Unexpected validation requirement count",
)

for frame, name in [
    (annotation, "annotation summary"),
    (audit, "audit summary"),
]:
    verify(
        len(frame) == 1,
        f"Expected one record in {name}",
    )

a = annotation.iloc[0]
b = audit.iloc[0]

verify(
    int(a["total_requirements"]) == 43,
    "Unexpected annotation review count",
)

verify(
    (
        int(a["agreement"]),
        int(a["disagreement"]),
        int(a["uncertain"]),
    ) == (11, 25, 7),
    "Annotation comparison mismatch",
)

verify(
    int(b["selected_for_adjudication"]) == 32,
    "Unexpected audit selection count",
)

verify(
    int(b["none_issue_disagreements"]) == 13,
    "Unexpected unexplained disagreements",
)

# ------------------------------------------------------------
# 2. Recalculate transformer means and standard deviations
# ------------------------------------------------------------

recalculated = (
    runs.groupby("architecture")
    .agg(
        nfr_f1_mean_check=("nfr_f1", "mean"),
        nfr_f1_std_check=("nfr_f1", "std"),
        nfr_recall_mean_check=("nfr_recall", "mean"),
        macro_f1_mean_check=("macro_f1", "mean"),
    )
    .reset_index()
)

check = stability.merge(
    recalculated,
    on="architecture",
    validate="one_to_one",
)

# The saved CSV may round displayed metrics, so allow
# minor rounding differences.
for original, calculated in [
    ("nfr_f1_mean", "nfr_f1_mean_check"),
    ("nfr_f1_std", "nfr_f1_std_check"),
    ("nfr_recall_mean", "nfr_recall_mean_check"),
    ("macro_f1_mean", "macro_f1_mean_check"),
]:
    verify(
        np.allclose(
            check[original],
            check[calculated],
            atol=0.0002,
            rtol=0,
        ),
        f"Seed aggregation discrepancy: {original}",
    )

# ------------------------------------------------------------
# 3. Merge transformer stability results
# ------------------------------------------------------------

comparison = stability.merge(
    prediction[
        [
            "architecture",
            "unanimous_predictions",
            "changing_predictions",
            "always_correct",
            "always_wrong",
            "majority_vote_nfr_f1_descriptive",
        ]
    ],
    on="architecture",
    validate="one_to_one",
)

comparison = comparison.sort_values(
    "nfr_f1_mean",
    ascending=False,
).reset_index(drop=True)

comparison.to_csv(
    OUTPUT_TABLE,
    index=False,
)

print("\nTRANSFORMER COMPARISON")
print(format_table(comparison))

# ------------------------------------------------------------
# 4. Audit source inventory
# ------------------------------------------------------------

inventory = []

for name, path in SOURCES.items():
    inventory.append({
        "source": name,
        "path": str(path),
        "sha256": sha256(path),
        "records": len(loaded[name]),
        "status": "VERIFIED",
    })

# Discover previous baseline outputs without guessing
# their exact filenames or interpreting unknown schemas.
baseline_files = sorted(
    set(TABLES.glob("03_*.csv"))
    | set(TABLES.glob("04_*.csv"))
)

for path in baseline_files:
    inventory.append({
        "source": "baseline_artifact",
        "path": str(path),
        "sha256": sha256(path),
        "records": "",
        "status": "INVENTORIED_NOT_SCHEMA_RECONCILED",
    })

inventory_df = pd.DataFrame(inventory)

inventory_df.to_csv(
    OUTPUT_INVENTORY,
    index=False,
)

# ------------------------------------------------------------
# 5. Previously recorded baseline figures
#
# These are historical console-output figures.
# They are not independently reverified from source
# metric files in this script.
# ------------------------------------------------------------

baseline_lines = [
    "NEURAL BASELINES: PREVIOUSLY RECORDED RESULTS",
    "Provenance: Earlier experimental console outputs.",
    "These figures are not independently revalidated",
    "against the original metrics CSVs by Script 17.",
    "",
    "C01 TextCNN standard:",
    "  Validation NFR F1 = 0.8058",
    "  Macro F1 = 0.8553",
    "",
    "C02 TextCNN weighted:",
    "  Validation NFR F1 = 0.8000",
    "  Macro F1 = 0.8485",
    "",
    "L01 BiLSTM standard:",
    "  Validation NFR F1 = 0.7889",
    "  Macro F1 = 0.8443",
    "",
    "L02 BiLSTM weighted:",
    "  Validation NFR F1 = 0.8016",
    "  Macro F1 = 0.8499",
]

# ------------------------------------------------------------
# 6. Research report
# ------------------------------------------------------------

report = [
    "DEEP LEARNING RESEARCH CONSOLIDATION",
    "=" * 76,
    "",
    "RESEARCH CONTEXT",
    "Task: Binary FR/NFR requirements classification",
    "Dataset protocol: D2-v1.1",
    "Training requirements: 3378",
    "Validation requirements: 728",
    "Sealed test requirements: 727",
    "",
    "All transformer results below use validation data.",
    "No test-set results are included.",
    "",
    "TRANSFORMER MULTI-SEED RESULTS",
    "Architectures: DistilBERT, BERT-base, RoBERTa-base",
    "Seeds: 42, 17, 29",
    "Runs: 9",
    "",
    format_table(runs),
    "",
    "MULTI-SEED AND PREDICTION STABILITY",
    format_table(comparison),
    "",
    *baseline_lines,
    "",
    "PERSISTENT ERRORS",
    "All nine transformer runs wrong: 29",
    "Persistently wrong in two or more architectures: 43",
    "Requirements with seed-dependent predictions: 104",
    "These counts are from Script 11 validation analysis.",
    "",
    "SCRIPT 11 ERROR CATEGORY SUMMARY",
    format_table(errors),
    "",
    "AI-ASSISTED ANNOTATION COMPARISON",
    "Selected difficult requirements: 43",
    "Agreement with original labels: 11",
    "Disagreement with original labels: 25",
    "Uncertain AI judgments: 7",
    "Disagreement audit cases: 32",
    "Disagreements with issue category NONE: 13",
    "",
    "Important: These are AI-assisted classifications.",
    "They are not independent expert ground truth.",
    "No original annotations have been corrected.",
    "",
    "PROVISIONAL MODELING INTERPRETATION",
    "DistilBERT had the highest observed mean",
    "validation NFR F1 across three seeds.",
    "The margin over BERT-base was small.",
    "Three seeds do not establish statistical superiority.",
    "A single validation partition was reused extensively.",
    "",
    "FINAL MODEL SELECTION",
    "Status: PENDING",
    "Final seed/checkpoint rule: NOT FROZEN",
    "Frozen test evaluation: NOT PERFORMED",
    "",
    "LIMITATIONS",
    "- Same validation set reused for many decisions.",
    "- Only three random seeds per transformer.",
    "- Dataset provenance remains a research concern.",
    "- Project-level independence not established.",
    "- Human annotation adjudication not completed.",
    "- AI disagreement does not prove dataset errors.",
    "- Majority-vote metrics are exploratory.",
    "- Full baseline source reconciliation remains open.",
    "",
    "NEXT RESEARCH STEPS",
    "1. Establish a final model-selection protocol.",
    "2. Record selected model and checkpoint SHA256.",
    "3. Freeze thresholds and preprocessing rules.",
    "4. Conduct one final sealed-test evaluation.",
    "5. Report generalization limits accurately.",
]

OUTPUT_REPORT.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

OUTPUT_JSON.write_text(
    json.dumps({
        "experiment": "DL-17",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": "D2-v1.1",
        "transformer_runs": 9,
        "architectures": sorted(ARCHITECTURES),
        "seeds": sorted(SEEDS),
        "best_mean_validation_nfr_f1_architecture":
            comparison.iloc[0]["architecture"],
        "test_accessed": False,
        "model_training": False,
        "final_model_selection": "PENDING",
        "source_hashes": {
            name: sha256(path)
            for name, path in SOURCES.items()
        },
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH CONSOLIDATION")
print("Transformer runs verified: 9")
print("Baseline outputs: inventoried")
print("Annotation audit: verified")
print("Research report: created")
print(
    "Leading mean validation NFR F1:",
    comparison.iloc[0]["architecture"],
)

print("\nRESEARCH STATUS")
print("Research consolidation: COMPLETED")
print("Final model selection: PENDING")
print("New training performed: NO")
print("Frozen test accessed: NO")

print("\nFILES GENERATED")
for path in [
    OUTPUT_TABLE,
    OUTPUT_INVENTORY,
    OUTPUT_REPORT,
    OUTPUT_JSON,
]:
    print(path)

print("\nScript 17 completed.")
