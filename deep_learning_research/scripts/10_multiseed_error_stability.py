
from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
)

# ============================================================
# SCRIPT 10: MULTI-SEED PREDICTION STABILITY
#
# - Reads nine saved validation prediction files
# - Checks IDs, labels, probability values
# - Analyzes within-architecture seed disagreement
# - Identifies persistent errors and unstable predictions
# - Computes same-seed cross-architecture comparisons
#
# No training, no model selection, no test access
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deep_learning_research"

DATA = ROOT / "dataset_exploration/outputs/validated"
MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
META = DATA / "54_d2_v11_split_metadata.json"

PREDS = PROJECT / "outputs/predictions"
TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 17, 29]
MODELS = ["distilbert", "bert", "roberta"]

FILES = {
    "distilbert": {
        42: "05_distilbert_validation_predictions.csv",
        17: "09_distilbert_seed17_validation_predictions.csv",
        29: "09_distilbert_seed29_validation_predictions.csv",
    },
    "bert": {
        42: "06_bert_validation_predictions.csv",
        17: "09_bert_seed17_validation_predictions.csv",
        29: "09_bert_seed29_validation_predictions.csv",
    },
    "roberta": {
        42: "07_roberta_validation_predictions.csv",
        17: "09_roberta_seed17_validation_predictions.csv",
        29: "09_roberta_seed29_validation_predictions.csv",
    },
}

LABEL_MAP = {"FR": 0, "NFR": 1}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------
# 1. Verify frozen manifest and load validation metadata
# ------------------------------------------------------------

metadata = json.loads(
    META.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError("Unexpected dataset protocol")

if sha256(MANIFEST) != metadata[
    "output_sha256"
][MANIFEST.name]:
    raise RuntimeError("Manifest hash mismatch")

parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    selected = chunk.loc[
        chunk["split"] == "validation",
        ["Requirement ID", "label", "similarity_group"],
    ].copy()

    if len(selected):
        parts.append(selected)

validation = pd.concat(
    parts,
    ignore_index=True
).rename(columns={
    "Requirement ID": "requirement_id",
    "label": "true_label",
})

if len(validation) != 728:
    raise RuntimeError("Unexpected validation count")

if not validation["requirement_id"].is_unique:
    raise RuntimeError("Duplicate requirement IDs")

if validation["similarity_group"].nunique() != 694:
    raise RuntimeError("Unexpected group count")

validation = validation.sort_values(
    "requirement_id"
).reset_index(drop=True)

merged = validation.copy()

# ------------------------------------------------------------
# 2. Load and verify all nine prediction files
# ------------------------------------------------------------

for model in MODELS:
    for seed in SEEDS:
        path = PREDS / FILES[model][seed]

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(
            path,
            dtype={"requirement_id": str},
        )

        required = {
            "requirement_id",
            "true_label",
            "predicted_label",
            "nfr_probability",
        }

        if not required.issubset(frame.columns):
            raise RuntimeError(
                f"Missing prediction columns: {path.name}"
            )

        if len(frame) != 728:
            raise RuntimeError(
                f"Incorrect row count: {path.name}"
            )

        if not frame["requirement_id"].is_unique:
            raise RuntimeError(
                f"Duplicate IDs: {path.name}"
            )

        if not frame["true_label"].isin(
            LABEL_MAP
        ).all():
            raise RuntimeError(
                f"Invalid true labels: {path.name}"
            )

        if not frame["predicted_label"].isin(
            LABEL_MAP
        ).all():
            raise RuntimeError(
                f"Invalid predictions: {path.name}"
            )

        probability = pd.to_numeric(
            frame["nfr_probability"],
            errors="raise",
        )

        if (
            not np.isfinite(probability).all()
            or not probability.between(0, 1).all()
        ):
            raise RuntimeError(
                f"Invalid probabilities: {path.name}"
            )

        if not frame["requirement_id"].isin(
            validation["requirement_id"]
        ).all():
            raise RuntimeError(
                f"Unknown requirement ID: {path.name}"
            )

        frame = frame.set_index(
            "requirement_id"
        ).reindex(
            validation["requirement_id"]
        )

        if frame["true_label"].isna().any():
            raise RuntimeError(
                f"Missing aligned predictions: {path.name}"
            )

        if not (
            frame["true_label"].to_numpy()
            == validation["true_label"].to_numpy()
        ).all():
            raise RuntimeError(
                f"True labels disagree: {path.name}"
            )

        column = f"{model}_seed{seed}"

        merged[column] = (
            frame["predicted_label"].to_numpy()
        )

        merged[f"{column}_prob"] = (
            frame["nfr_probability"]
            .astype(float)
            .to_numpy()
        )

print("=" * 76)
print("SCRIPT 10: MULTI-SEED ERROR STABILITY")
print("=" * 76)

print("\nDATA VALIDATION")
print("Validation requirements:", len(merged))
print("Similarity groups:", merged["similarity_group"].nunique())
print("Prediction files:", 9)
print("Prediction alignment: PASS")
print("True-label agreement: PASS")
print("Manifest integrity: PASS")
print("Test accessed: NO")

# ------------------------------------------------------------
# 3. Within-model prediction stability
# ------------------------------------------------------------

stability_rows = []
detail_rows = []

true = merged["true_label"].to_numpy()

for model in MODELS:
    columns = [
        f"{model}_seed{seed}"
        for seed in SEEDS
    ]

    predictions = merged[columns].to_numpy()

    correct = predictions == true[:, None]

    unanimous = (
        predictions[:, 0] == predictions[:, 1]
    ) & (
        predictions[:, 1] == predictions[:, 2]
    )

    always_correct = correct.all(axis=1)
    always_wrong = (~correct).all(axis=1)
    changing_predictions = ~unanimous

    majority_prediction = np.where(
        (predictions == "NFR").sum(axis=1) >= 2,
        "NFR",
        "FR",
    )

    seed_disagreements = []

    for first, second in combinations(
        range(len(SEEDS)), 2
    ):
        count = int(
            np.sum(
                predictions[:, first]
                != predictions[:, second]
            )
        )

        seed_disagreements.append(count)

    stability_rows.append({
        "architecture": model,
        "requirements": len(merged),
        "unanimous_predictions": int(
            unanimous.sum()
        ),
        "unanimous_percent": float(
            unanimous.mean() * 100
        ),
        "changing_predictions": int(
            changing_predictions.sum()
        ),
        "changing_percent": float(
            changing_predictions.mean() * 100
        ),
        "always_correct": int(
            always_correct.sum()
        ),
        "always_wrong": int(
            always_wrong.sum()
        ),
        "mean_pairwise_disagreements": float(
            np.mean(seed_disagreements)
        ),
        "majority_vote_nfr_f1_descriptive": f1_score(
            (true == "NFR").astype(int),
            (majority_prediction == "NFR").astype(int),
            zero_division=0,
        ),
    })

    for idx in range(len(merged)):
        detail_rows.append({
            "architecture": model,
            "requirement_id": merged.loc[
                idx, "requirement_id"
            ],
            "similarity_group": merged.loc[
                idx, "similarity_group"
            ],
            "true_label": true[idx],
            "seed42_prediction": predictions[idx, 0],
            "seed17_prediction": predictions[idx, 1],
            "seed29_prediction": predictions[idx, 2],
            "unanimous": bool(unanimous[idx]),
            "always_correct": bool(always_correct[idx]),
            "always_wrong": bool(always_wrong[idx]),
            "majority_prediction": majority_prediction[idx],
        })

stability = pd.DataFrame(stability_rows)
details = pd.DataFrame(detail_rows)

print("\nWITHIN-MODEL STABILITY")
print(
    stability.round(4).to_string(index=False)
)

# ------------------------------------------------------------
# 4. Pairwise comparisons within matching seeds
# ------------------------------------------------------------

comparison_rows = []

for seed in SEEDS:
    for a, b in combinations(MODELS, 2):

        p_a = merged[
            f"{a}_seed{seed}"
        ].map(LABEL_MAP).to_numpy()

        p_b = merged[
            f"{b}_seed{seed}"
        ].map(LABEL_MAP).to_numpy()

        y = merged[
            "true_label"
        ].map(LABEL_MAP).to_numpy()

        a_correct = p_a == y
        b_correct = p_b == y

        comparison_rows.append({
            "seed": seed,
            "model_a": a,
            "model_b": b,
            "changed_predictions": int(
                (p_a != p_b).sum()
            ),
            "a_only_correct": int(
                (a_correct & ~b_correct).sum()
            ),
            "b_only_correct": int(
                (~a_correct & b_correct).sum()
            ),
            "nfr_f1_a_minus_b": (
                f1_score(y, p_a)
                - f1_score(y, p_b)
            ),
            "nfr_recall_a_minus_b": (
                recall_score(y, p_a)
                - recall_score(y, p_b)
            ),
            "accuracy_a_minus_b": (
                accuracy_score(y, p_a)
                - accuracy_score(y, p_b)
            ),
        })

comparisons = pd.DataFrame(comparison_rows)

print("\nMATCHED-SEED MODEL COMPARISONS")
print(
    comparisons.round(4).to_string(index=False)
)

# ------------------------------------------------------------
# 5. Save all outputs
# ------------------------------------------------------------

stability_path = (
    TABLES / "10_multiseed_prediction_stability.csv"
)

details_path = (
    TABLES / "10_multiseed_requirement_details.csv"
)

comparison_path = (
    TABLES / "10_multiseed_matched_comparisons.csv"
)

report_path = (
    REPORTS / "10_multiseed_error_stability.txt"
)

metadata_path = (
    REPORTS / "10_multiseed_error_stability.json"
)

stability.to_csv(stability_path, index=False)
details.to_csv(details_path, index=False)
comparisons.to_csv(comparison_path, index=False)

report_path.write_text(
    "\n".join([
        "D2 v1.1 MULTI-SEED ERROR STABILITY",
        "=" * 76,
        "",
        "Dataset: D2-v1.1",
        "Validation requirements: 728",
        "Similarity groups: 694",
        "Seeds: 42, 17, 29",
        "Test accessed: NO",
        "",
        "WITHIN-MODEL STABILITY",
        stability.round(4).to_string(index=False),
        "",
        "MATCHED-SEED MODEL COMPARISONS",
        comparisons.round(4).to_string(index=False),
        "",
        "LIMITATIONS",
        "- The same validation partition is reused.",
        "- All results are exploratory.",
        "- Only three training seeds are available.",
        "- Majority-vote metrics are descriptive only.",
        "- No new ensemble has been approved or trained.",
        "- No independent-project evaluation.",
        "- Test partition remains sealed.",
    ]) + "\n",
    encoding="utf-8",
)

metadata_path.write_text(
    json.dumps({
        "experiment": "DL-10",
        "protocol": "D2-v1.1",
        "manifest_sha256": sha256(MANIFEST),
        "seeds": SEEDS,
        "architectures": MODELS,
        "validation_requirements": len(merged),
        "new_training": False,
        "test_accessed": False,
        "final_model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Multi-seed error analysis: COMPLETED")
print("New model training: NO")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    stability_path,
    details_path,
    comparison_path,
    report_path,
    metadata_path,
]:
    print(path)

print("\nScript 10 completed.")
