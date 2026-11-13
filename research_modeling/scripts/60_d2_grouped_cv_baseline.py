
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# SCRIPT 60: D2 v1.1 GROUP-AWARE CROSS-VALIDATION
#
# 5-fold CV on the TRAINING partition only.
#
# Compare:
# B01: Standard Logistic Regression
# B02: Balanced Logistic Regression
#
# TF-IDF fitted independently inside every fold.
# No validation or test predictions are used.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "dataset_exploration/outputs/validated"
OUTPUT = ROOT / "research_modeling/outputs"

TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
METADATA = DATA / "54_d2_v11_split_metadata.json"

SEED = 42
N_FOLDS = 5
THRESHOLDS = np.round(
    np.arange(0.20, 0.8001, 0.025), 3
)

MODELS = {
    "B01_LogisticRegression": None,
    "B02_BalancedLogisticRegression": "balanced"
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def metrics(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "nfr_precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "nfr_recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "nfr_f1": f1_score(
            y_true, pred, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, pred,
            average="macro",
            zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, pred
        ),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn)
    }


# ------------------------------------------------------------
# 1. Verify inputs
# ------------------------------------------------------------

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

if metadata["protocol"] != "D2-v1.1":
    raise RuntimeError("Wrong dataset protocol")

expected_hash = metadata["output_sha256"][MANIFEST.name]

if sha256(MANIFEST) != expected_hash:
    raise RuntimeError("Frozen manifest integrity failure")

# Read only training records.
train_parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000
):
    train_parts.append(
        chunk[chunk["split"] == "train"].copy()
    )

train = pd.concat(
    train_parts, ignore_index=True
)

train = train.sort_values("key").reset_index(drop=True)

if len(train) != 3378:
    raise RuntimeError("Unexpected training count")

if train["key"].duplicated().any():
    raise RuntimeError("Duplicate training texts")

if set(train["label"]) != {"FR", "NFR"}:
    raise RuntimeError("Unexpected class labels")

text_col = "Requirement Text (Original Requirement)"

if text_col not in train.columns:
    raise ValueError("Missing requirement text column")

texts = train[text_col].astype(str).to_numpy()

if any(not t.strip() for t in texts):
    raise ValueError("Blank training requirement")

y = train["label"].map({
    "FR": 0,
    "NFR": 1
}).to_numpy(dtype=int)

groups = train["similarity_group"].to_numpy()

print("=" * 76)
print("SCRIPT 60: D2 v1.1 GROUP-AWARE CROSS-VALIDATION")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements:", len(train))
print("FR:", int((y == 0).sum()))
print("NFR:", int((y == 1).sum()))
print("Training similarity groups:", len(set(groups)))
print("Cross-validation folds:", N_FOLDS)
print("External validation used: NO")
print("Test used: NO")

# ------------------------------------------------------------
# 2. Build deterministic grouped folds
# ------------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

folds = list(
    cv.split(texts, y, groups)
)

fold_records = []
fold_predictions = []

for fold_number, (tr_idx, va_idx) in enumerate(
    folds, start=1
):

    tr_groups = set(groups[tr_idx])
    va_groups = set(groups[va_idx])

    if tr_groups & va_groups:
        raise RuntimeError("CV group leakage")

    if len(set(y[tr_idx])) != 2:
        raise RuntimeError("Fold training class missing")

    if len(set(y[va_idx])) != 2:
        raise RuntimeError("Fold held-out class missing")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        max_features=100000,
        norm="l2"
    )

    X_tr = vectorizer.fit_transform(texts[tr_idx])
    X_va = vectorizer.transform(texts[va_idx])

    for name, class_weight in MODELS.items():

        model = LogisticRegression(
            C=1.0,
            class_weight=class_weight,
            max_iter=2000,
            solver="liblinear",
            random_state=SEED
        )

        model.fit(X_tr, y[tr_idx])

        probabilities = model.predict_proba(X_va)[:, 1]

        fold_metrics = metrics(
            y[va_idx],
            probabilities,
            threshold=0.50
        )

        fold_records.append({
            "fold": fold_number,
            "model": name,
            "train_size": len(tr_idx),
            "heldout_size": len(va_idx),
            "heldout_fr": int((y[va_idx] == 0).sum()),
            "heldout_nfr": int((y[va_idx] == 1).sum()),
            **fold_metrics
        })

        for local_i, probability in zip(
            va_idx, probabilities
        ):
            fold_predictions.append({
                "model": name,
                "fold": fold_number,
                "requirement_id": train.loc[
                    local_i, "Requirement ID"
                ],
                "true_label": int(y[local_i]),
                "nfr_probability": float(probability)
            })

        print(
            f"Fold {fold_number}, {name}: "
            f"NFR F1={fold_metrics['nfr_f1']:.4f}, "
            f"NFR recall={fold_metrics['nfr_recall']:.4f}"
        )

# ------------------------------------------------------------
# 3. Validate out-of-fold predictions
# ------------------------------------------------------------

fold_df = pd.DataFrame(fold_records)
oof = pd.DataFrame(fold_predictions)

for name in MODELS:
    subset = oof[oof["model"] == name]

    if len(subset) != len(train):
        raise RuntimeError(
            f"Incomplete OOF predictions: {name}"
        )

    if subset["requirement_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate OOF predictions: {name}"
        )

    if set(subset["requirement_id"]) != set(
        train["Requirement ID"]
    ):
        raise RuntimeError(
            f"Missing OOF requirement IDs: {name}"
        )

# ------------------------------------------------------------
# 4. Summarize fold stability
# ------------------------------------------------------------

stability = (
    fold_df.groupby("model")
    .agg(
        mean_nfr_f1=("nfr_f1", "mean"),
        std_nfr_f1=("nfr_f1", "std"),
        mean_nfr_recall=("nfr_recall", "mean"),
        std_nfr_recall=("nfr_recall", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_balanced_accuracy=(
            "balanced_accuracy", "mean"
        )
    )
    .reset_index()
)

# ------------------------------------------------------------
# 5. Threshold selection using training OOF scores
# ------------------------------------------------------------

threshold_records = []
ranking_records = []

for name in MODELS:

    subset = (
        oof[oof["model"] == name]
        .sort_values("requirement_id")
    )

    y_true = subset["true_label"].to_numpy(dtype=int)
    probs = subset["nfr_probability"].to_numpy(dtype=float)

    ranking_records.append({
        "model": name,
        "oof_average_precision": average_precision_score(
            y_true, probs
        ),
        "oof_roc_auc": roc_auc_score(
            y_true, probs
        )
    })

    for threshold in THRESHOLDS:
        row = metrics(y_true, probs, threshold)
        row["model"] = name
        threshold_records.append(row)

sweep = pd.DataFrame(threshold_records)

best_f1 = (
    sweep.sort_values(
        ["nfr_f1", "nfr_precision", "threshold"],
        ascending=[False, False, False]
    )
    .drop_duplicates("model")
    .sort_values("model")
)

recall_target = (
    sweep[sweep["nfr_recall"] >= 0.85]
    .sort_values(
        ["nfr_precision", "nfr_f1", "threshold"],
        ascending=[False, False, False]
    )
    .drop_duplicates("model")
    .sort_values("model")
)

ranking = pd.DataFrame(ranking_records)

# ------------------------------------------------------------
# 6. Save outputs
# ------------------------------------------------------------

fold_df.to_csv(
    TABLES / "60_d2_fold_metrics.csv",
    index=False
)

oof.to_csv(
    TABLES / "60_d2_oof_predictions.csv",
    index=False
)

stability.to_csv(
    TABLES / "60_d2_cv_stability.csv",
    index=False
)

sweep.to_csv(
    TABLES / "60_d2_oof_threshold_sweep.csv",
    index=False
)

best_f1.to_csv(
    TABLES / "60_d2_oof_best_f1.csv",
    index=False
)

recall_target.to_csv(
    TABLES / "60_d2_oof_recall_target.csv",
    index=False
)

ranking.to_csv(
    TABLES / "60_d2_oof_ranking_metrics.csv",
    index=False
)

columns = [
    "model", "threshold", "nfr_precision",
    "nfr_recall", "nfr_f1", "macro_f1",
    "balanced_accuracy", "fp", "fn"
]

print("\n" + "=" * 76)
print("CROSS-VALIDATION STABILITY")
print("=" * 76)
print(stability.round(4).to_string(index=False))

print("\nTRAIN-OOF BEST NFR F1 THRESHOLDS")
print(
    best_f1[columns].round(4).to_string(index=False)
)

print("\nTRAIN-OOF RECALL >= 0.85")
if len(recall_target):
    print(
        recall_target[columns]
        .round(4).to_string(index=False)
    )
else:
    print("No threshold achieved the recall target.")

print("\nTRAIN-OOF RANKING METRICS")
print(ranking.round(4).to_string(index=False))

report = [
    "D2 v1.1 GROUP-AWARE CROSS-VALIDATION",
    "=" * 65,
    "",
    f"Training requirements: {len(train)}",
    f"Folds: {N_FOLDS}",
    "Feature extraction: fitted separately in each fold",
    "External validation used: NO",
    "Test used: NO",
    "",
    "CROSS-VALIDATION STABILITY",
    stability.round(4).to_string(index=False),
    "",
    "TRAIN-OOF BEST NFR F1",
    best_f1[columns].round(4).to_string(index=False),
    "",
    "TRAIN-OOF RECALL TARGET",
    recall_target[columns].round(4).to_string(index=False),
    "",
    "TRAIN-OOF RANKING",
    ranking.round(4).to_string(index=False),
    "",
    "Limitations:",
    "- OOF thresholds are exploratory, not unbiased estimates.",
    "- Training folds share the same dataset source.",
    "- Original-project independence is not established.",
    "- No external validation or test predictions were used."
]

report_path = REPORTS / "60_d2_grouped_cv_report.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

metadata_path = REPORTS / "60_d2_grouped_cv_metadata.json"

metadata_path.write_text(
    json.dumps({
        "experiment": 60,
        "protocol": "D2-v1.1",
        "seed": SEED,
        "folds": N_FOLDS,
        "manifest_sha256": sha256(MANIFEST),
        "evaluation": "TRAIN_OOF_ONLY",
        "validation_used": False,
        "test_used": False,
        "threshold_selection": "EXPLORATORY",
        "model_selection": "NOT_FINALIZED"
    }, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Grouped cross-validation: COMPLETED")
print("External validation: UNTOUCHED")
print("Test set: SEALED")
print("Model selection: PENDING")

print("\nSaved:", report_path)
print("Script 60 completed.")
