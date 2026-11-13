
from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

# ==========================================================
# SCRIPT 08: PAIRED TRANSFORMER COMPARISON
#
# D01: DistilBERT
# D02: BERT-base
# D03: RoBERTa-base
#
# Validation predictions only.
# Similarity-group bootstrap: 2000 repetitions.
# Exact McNemar test for paired correctness.
#
# No training or test-set access.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deep_learning_research"

PRED_DIR = PROJECT / "outputs/predictions"
TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"

DATA = ROOT / "dataset_exploration/outputs/validated"

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
META = DATA / "54_d2_v11_split_metadata.json"

FILES = {
    "D01_DistilBERT":
        PRED_DIR / "05_distilbert_validation_predictions.csv",
    "D02_BERTBase":
        PRED_DIR / "06_bert_validation_predictions.csv",
    "D03_RoBERTaBase":
        PRED_DIR / "07_roberta_validation_predictions.csv",
}

SEED = 42
BOOTSTRAP_REPETITIONS = 2000

LABEL_MAP = {"FR": 0, "NFR": 1}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1048576), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def nfr_f1(y, pred):
    return f1_score(
        y, pred, zero_division=0
    )


def nfr_recall(y, pred):
    return recall_score(
        y, pred, zero_division=0
    )


def confidence_interval(values):
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# ----------------------------------------------------------
# 1. Verify manifest integrity
# ----------------------------------------------------------

meta = json.loads(
    META.read_text(encoding="utf-8")
)

if meta.get("protocol") != "D2-v1.1":
    raise RuntimeError("Unexpected protocol")

if sha256(MANIFEST) != meta[
    "output_sha256"
][MANIFEST.name]:
    raise RuntimeError("Manifest integrity failed")

# Only validation rows are retained.
parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    part = chunk.loc[
        chunk["split"] == "validation",
        ["Requirement ID", "label", "similarity_group"]
    ].copy()

    if len(part):
        parts.append(part)

val = pd.concat(
    parts, ignore_index=True
).rename(columns={
    "Requirement ID": "requirement_id",
    "label": "manifest_label",
})

if len(val) != 728:
    raise RuntimeError("Unexpected validation size")

if not val["requirement_id"].is_unique:
    raise RuntimeError("Duplicate validation IDs")

if val["similarity_group"].nunique() != 694:
    raise RuntimeError("Unexpected group count")

if set(val["manifest_label"]) != {"FR", "NFR"}:
    raise RuntimeError("Unexpected true labels")

# ----------------------------------------------------------
# 2. Load saved predictions and align by ID
# ----------------------------------------------------------

merged = val.copy()

for name, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        dtype={"requirement_id": str}
    )

    expected = {
        "requirement_id",
        "true_label",
        "predicted_label",
        "nfr_probability",
    }

    if not expected.issubset(frame.columns):
        raise RuntimeError(
            f"Missing columns in {path.name}"
        )

    frame = frame[list(expected)].copy()

    if len(frame) != 728:
        raise RuntimeError(
            f"Wrong prediction count: {name}"
        )

    if not frame["requirement_id"].is_unique:
        raise RuntimeError(
            f"Duplicate prediction IDs: {name}"
        )

    if frame[
        ["true_label", "predicted_label"]
    ].isna().any().any():
        raise RuntimeError(f"Missing labels: {name}")

    if not set(frame["true_label"]).issubset(LABEL_MAP):
        raise RuntimeError(f"Unexpected true labels: {name}")

    if not set(frame["predicted_label"]).issubset(LABEL_MAP):
        raise RuntimeError(f"Unexpected predictions: {name}")

    frame["nfr_probability"] = pd.to_numeric(
        frame["nfr_probability"],
        errors="raise"
    )

    if not np.isfinite(
        frame["nfr_probability"]
    ).all():
        raise RuntimeError(
            f"Nonfinite probabilities: {name}"
        )

    if not frame[
        "nfr_probability"
    ].between(0, 1).all():
        raise RuntimeError(
            f"Invalid probabilities: {name}"
        )

    frame = frame.rename(columns={
        "true_label": f"{name}_true",
        "predicted_label": f"{name}_pred",
        "nfr_probability": f"{name}_prob",
    })

    merged = merged.merge(
        frame,
        on="requirement_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise RuntimeError(
            f"Missing prediction records: {name}"
        )

    merged = merged.drop(columns="_merge")

    if not (
        merged[f"{name}_true"]
        == merged["manifest_label"]
    ).all():
        raise RuntimeError(
            f"True labels disagree: {name}"
        )

print("=" * 76)
print("SCRIPT 08: PAIRED TRANSFORMER COMPARISON")
print("=" * 76)

print("\nDATA VALIDATION")
print("Validation requirements:", len(merged))
print(
    "Validation similarity groups:",
    merged["similarity_group"].nunique()
)
print("Prediction alignment: PASS")
print("True-label agreement: PASS")
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

# ----------------------------------------------------------
# 3. Individual model metrics
# ----------------------------------------------------------

y = merged["manifest_label"].map(
    LABEL_MAP
).to_numpy()

individual_rows = []

for name in FILES:

    p = merged[f"{name}_pred"].map(
        LABEL_MAP
    ).to_numpy()

    prob = merged[f"{name}_prob"].to_numpy()

    individual_rows.append({
        "model": name,
        "accuracy": accuracy_score(y, p),
        "nfr_f1": nfr_f1(y, p),
        "nfr_recall": nfr_recall(y, p),
        "average_precision":
            average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob),
        "errors": int(np.sum(y != p)),
    })

individual = pd.DataFrame(individual_rows)

print("\nINDIVIDUAL VALIDATION RESULTS")
print(
    individual.round(4).to_string(index=False)
)

# ----------------------------------------------------------
# 4. Paired comparison
# ----------------------------------------------------------

rng = np.random.default_rng(SEED)

groups = merged[
    "similarity_group"
].to_numpy()

unique_groups = np.unique(groups)

group_indices = [
    np.where(groups == group)[0]
    for group in unique_groups
]

comparison_rows = []
changed_rows = []

for name_a, name_b in combinations(FILES, 2):

    a = merged[f"{name_a}_pred"].map(
        LABEL_MAP
    ).to_numpy()

    b = merged[f"{name_b}_pred"].map(
        LABEL_MAP
    ).to_numpy()

    correct_a = a == y
    correct_b = b == y

    a_only_correct = int(
        np.sum(correct_a & ~correct_b)
    )

    b_only_correct = int(
        np.sum(~correct_a & correct_b)
    )

    discordant = (
        a_only_correct + b_only_correct
    )

    if discordant:
        mcnemar_p = binomtest(
            min(a_only_correct, b_only_correct),
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        mcnemar_p = 1.0

    f1_diffs = []
    recall_diffs = []

    for _ in range(BOOTSTRAP_REPETITIONS):

        selected = rng.integers(
            0,
            len(group_indices),
            size=len(group_indices),
        )

        indices = np.concatenate([
            group_indices[index]
            for index in selected
        ])

        sample_y = y[indices]

        # Skip samples without positive examples.
        if not np.any(sample_y == 1):
            continue

        f1_diffs.append(
            nfr_f1(sample_y, a[indices])
            - nfr_f1(sample_y, b[indices])
        )

        recall_diffs.append(
            nfr_recall(sample_y, a[indices])
            - nfr_recall(sample_y, b[indices])
        )

    f1_low, f1_high = confidence_interval(
        f1_diffs
    )

    recall_low, recall_high = confidence_interval(
        recall_diffs
    )

    comparison_rows.append({
        "model_a": name_a,
        "model_b": name_b,
        "changed_predictions": int(
            np.sum(a != b)
        ),
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "mcnemar_exact_p": mcnemar_p,
        "nfr_f1_difference_a_minus_b":
            nfr_f1(y, a) - nfr_f1(y, b),
        "nfr_f1_ci_low": f1_low,
        "nfr_f1_ci_high": f1_high,
        "recall_difference_a_minus_b":
            nfr_recall(y, a) - nfr_recall(y, b),
        "recall_ci_low": recall_low,
        "recall_ci_high": recall_high,
    })

    changed = np.where(a != b)[0]

    for idx in changed:
        changed_rows.append({
            "requirement_id":
                merged.loc[idx, "requirement_id"],
            "similarity_group":
                merged.loc[idx, "similarity_group"],
            "true_label":
                merged.loc[idx, "manifest_label"],
            "model_a": name_a,
            "model_b": name_b,
            "prediction_a":
                merged.loc[idx, f"{name_a}_pred"],
            "prediction_b":
                merged.loc[idx, f"{name_b}_pred"],
            "a_correct": bool(correct_a[idx]),
            "b_correct": bool(correct_b[idx]),
        })

comparison = pd.DataFrame(comparison_rows)
changed = pd.DataFrame(changed_rows)

print("\nPAIRED COMPARISONS")

for _, row in comparison.iterrows():

    print(
        f"\n{row['model_a']} VS {row['model_b']}"
    )

    print(
        "Predictions changed:",
        row["changed_predictions"]
    )

    print(
        "A only correct:",
        row["a_only_correct"]
    )

    print(
        "B only correct:",
        row["b_only_correct"]
    )

    print(
        "Exact McNemar p:",
        round(row["mcnemar_exact_p"], 6)
    )

    print(
        "NFR F1 difference (A-B):",
        round(row[
            "nfr_f1_difference_a_minus_b"
        ], 4)
    )

    print(
        "NFR F1 bootstrap 95% CI:",
        (
            round(row["nfr_f1_ci_low"], 4),
            round(row["nfr_f1_ci_high"], 4),
        )
    )

    print(
        "Recall difference (A-B):",
        round(row[
            "recall_difference_a_minus_b"
        ], 4)
    )

    print(
        "Recall bootstrap 95% CI:",
        (
            round(row["recall_ci_low"], 4),
            round(row["recall_ci_high"], 4),
        )
    )

# ----------------------------------------------------------
# 5. Save results
# ----------------------------------------------------------

individual_path = (
    TABLES / "08_transformer_individual_metrics.csv"
)

comparison_path = (
    TABLES / "08_transformer_paired_comparison.csv"
)

changed_path = (
    TABLES / "08_transformer_changed_predictions.csv"
)

report_path = (
    REPORTS / "08_transformer_paired_report.txt"
)

metadata_path = (
    REPORTS / "08_transformer_paired_metadata.json"
)

individual.to_csv(individual_path, index=False)
comparison.to_csv(comparison_path, index=False)
changed.to_csv(changed_path, index=False)

report_path.write_text(
    "\n".join([
        "SCRIPT 08: TRANSFORMER PAIRED COMPARISON",
        "=" * 76,
        "",
        "Dataset: D2-v1.1",
        "Validation requirements: 728",
        "Similarity groups: 694",
        "Test accessed: NO",
        "",
        "INDIVIDUAL RESULTS",
        individual.round(5).to_string(index=False),
        "",
        "PAIRED RESULTS",
        comparison.round(5).to_string(index=False),
        "",
        "STATISTICAL METHODS",
        "Exact McNemar test on paired correctness",
        "Similarity-group bootstrap: 2000 repetitions",
        "Percentile 95% confidence intervals",
        "",
        "LIMITATIONS",
        "- All models use one training seed.",
        "- Validation was reused for model development.",
        "- Comparisons are exploratory, not confirmatory.",
        "- No multiple-testing adjustment applied.",
        "- Bootstrap covers validation sampling variation.",
        "- Bootstrap does not measure training-seed variation.",
        "- Test partition remains sealed.",
    ]) + "\n",
    encoding="utf-8",
)

metadata_path.write_text(
    json.dumps({
        "experiment": "DL-08",
        "protocol": "D2-v1.1",
        "manifest_sha256": sha256(MANIFEST),
        "bootstrap_seed": SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "validation_count": len(merged),
        "validation_groups": len(unique_groups),
        "models": list(FILES),
        "new_training": False,
        "test_accessed": False,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Paired transformer comparison: COMPLETED")
print("Similarity-group bootstrap: COMPLETED")
print("New training: NO")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    individual_path,
    comparison_path,
    changed_path,
    report_path,
    metadata_path,
]:
    print(path)

print("\nScript 08 completed.")
