
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# SCRIPT 62: D2 v1.1 LINEAR SVM BASELINE
#
# S01: TF-IDF + Linear SVM
# S02: TF-IDF + Balanced Linear SVM
#
# 5-fold group-aware training CV
# + evaluation on frozen validation partition
#
# No threshold optimization.
# No test-set access.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "dataset_exploration/outputs/validated"
)

OUTPUT_DIR = ROOT / "research_modeling/outputs"
TABLES = OUTPUT_DIR / "tables"
REPORTS = OUTPUT_DIR / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = (
    DATA_DIR / "54_d2_v11_frozen_split_manifest.csv"
)

METADATA = (
    DATA_DIR / "54_d2_v11_split_metadata.json"
)

SEED = 42
N_FOLDS = 5
TEXT_COL = "Requirement Text (Original Requirement)"

MODELS = {
    "S01_LinearSVM": None,
    "S02_BalancedLinearSVM": "balanced",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def make_vectorizer():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        max_features=100000,
        norm="l2",
    )


def evaluate(y_true, scores):
    # LinearSVC decision_function:
    # negative = FR, positive = NFR.
    pred = (scores >= 0).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, pred
        ),
        "macro_f1": f1_score(
            y_true, pred, average="macro",
            zero_division=0
        ),
        "nfr_precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "nfr_recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "nfr_f1": f1_score(
            y_true, pred, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_true, scores
        ),
        "roc_auc": roc_auc_score(
            y_true, scores
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ------------------------------------------------------------
# 1. Verify dataset protocol
# ------------------------------------------------------------

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

if metadata["protocol"] != "D2-v1.1":
    raise RuntimeError("Unexpected dataset protocol")

expected_hash = metadata[
    "output_sha256"
][MANIFEST.name]

if sha256(MANIFEST) != expected_hash:
    raise RuntimeError("Frozen manifest hash mismatch")

# Read only training and validation records.
parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    parts.append(
        chunk[
            chunk["split"].isin(
                ["train", "validation"]
            )
        ].copy()
    )

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL,
    "Requirement ID",
    "label",
    "split",
    "similarity_group",
    "key",
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

train = (
    df[df["split"] == "train"]
    .sort_values("key")
    .reset_index(drop=True)
)

val = (
    df[df["split"] == "validation"]
    .sort_values("key")
    .reset_index(drop=True)
)

if len(train) != 3378 or len(val) != 728:
    raise RuntimeError("Unexpected partition sizes")

if (
    set(train["similarity_group"])
    & set(val["similarity_group"])
):
    raise RuntimeError("Group overlap detected")

if (
    train[TEXT_COL].str.strip().eq("").any()
    or val[TEXT_COL].str.strip().eq("").any()
):
    raise ValueError("Blank requirement texts")

label_map = {"FR": 0, "NFR": 1}

y_train = train["label"].map(label_map)
y_val = val["label"].map(label_map)

if y_train.isna().any() or y_val.isna().any():
    raise ValueError("Unknown label")

y_train = y_train.to_numpy(dtype=int)
y_val = y_val.to_numpy(dtype=int)

texts_train = train[TEXT_COL].to_numpy()
texts_val = val[TEXT_COL].to_numpy()
groups = train["similarity_group"].to_numpy()

print("=" * 76)
print("SCRIPT 62: D2 v1.1 LINEAR SVM BASELINE")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements:", len(train))
print("Validation requirements:", len(val))
print("Training similarity groups:", len(set(groups)))
print("Test accessed: NO")
print("Frozen manifest integrity: PASS")

# ------------------------------------------------------------
# 2. Group-aware cross-validation
# ------------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

fold_records = []
oof_records = []

for fold, (tr_idx, hold_idx) in enumerate(
    cv.split(texts_train, y_train, groups),
    start=1
):
    if (
        set(groups[tr_idx])
        & set(groups[hold_idx])
    ):
        raise RuntimeError("CV group overlap")

    if (
        len(np.unique(y_train[tr_idx])) < 2
        or len(np.unique(y_train[hold_idx])) < 2
    ):
        raise RuntimeError("Missing class in CV fold")

    vectorizer = make_vectorizer()

    X_tr = vectorizer.fit_transform(
        texts_train[tr_idx]
    )
    X_hold = vectorizer.transform(
        texts_train[hold_idx]
    )

    for name, weight in MODELS.items():
        model = LinearSVC(
            C=1.0,
            class_weight=weight,
            max_iter=10000,
            random_state=SEED,
        )

        model.fit(X_tr, y_train[tr_idx])
        scores = model.decision_function(X_hold)

        result = evaluate(
            y_train[hold_idx], scores
        )

        fold_records.append({
            "fold": fold,
            "model": name,
            "train_size": len(tr_idx),
            "heldout_size": len(hold_idx),
            "heldout_nfr": int(
                (y_train[hold_idx] == 1).sum()
            ),
            **result,
        })

        for idx, score in zip(hold_idx, scores):
            oof_records.append({
                "model": name,
                "fold": fold,
                "requirement_id": train.loc[
                    idx, "Requirement ID"
                ],
                "true_label": int(y_train[idx]),
                "decision_score": float(score),
            })

        print(
            f"Fold {fold}, {name}: "
            f"NFR F1={result['nfr_f1']:.4f}, "
            f"NFR recall={result['nfr_recall']:.4f}"
        )

fold_df = pd.DataFrame(fold_records)
oof_df = pd.DataFrame(oof_records)

for name in MODELS:
    subset = oof_df[oof_df["model"] == name]

    if (
        len(subset) != len(train)
        or not subset["requirement_id"].is_unique
    ):
        raise RuntimeError(
            f"Invalid OOF coverage for {name}"
        )

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
        ),
    )
    .reset_index()
)

# ------------------------------------------------------------
# 3. Train on full training partition
# ------------------------------------------------------------

vectorizer = make_vectorizer()

X_train = vectorizer.fit_transform(texts_train)
X_val = vectorizer.transform(texts_val)

validation_records = []
validation_predictions = []

for name, weight in MODELS.items():
    model = LinearSVC(
        C=1.0,
        class_weight=weight,
        max_iter=10000,
        random_state=SEED,
    )

    model.fit(X_train, y_train)
    scores = model.decision_function(X_val)

    result = evaluate(y_val, scores)

    validation_records.append({
        "model": name,
        "decision_threshold": 0.0,
        **result,
    })

    for i, score in enumerate(scores):
        validation_predictions.append({
            "model": name,
            "requirement_id": val.loc[
                i, "Requirement ID"
            ],
            "true_label": val.loc[i, "label"],
            "decision_score": float(score),
            "predicted_label": (
                "NFR" if score >= 0 else "FR"
            ),
        })

validation_df = pd.DataFrame(validation_records)
prediction_df = pd.DataFrame(
    validation_predictions
)

# ------------------------------------------------------------
# 4. Save results
# ------------------------------------------------------------

fold_df.to_csv(
    TABLES / "62_d2_svm_cv_fold_metrics.csv",
    index=False,
)

oof_df.to_csv(
    TABLES / "62_d2_svm_oof_predictions.csv",
    index=False,
)

stability.to_csv(
    TABLES / "62_d2_svm_cv_stability.csv",
    index=False,
)

validation_df.to_csv(
    TABLES / "62_d2_svm_validation_metrics.csv",
    index=False,
)

prediction_df.to_csv(
    TABLES / "62_d2_svm_validation_predictions.csv",
    index=False,
)

display_cols = [
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
]

print("\n" + "=" * 76)
print("CROSS-VALIDATION STABILITY")
print("=" * 76)
print(stability.round(4).to_string(index=False))

print("\nVALIDATION COMPARISON")
print(
    validation_df[display_cols]
    .round(4)
    .to_string(index=False)
)

print("\nVALIDATION CONFUSION MATRICES")
print("Rows: actual FR, NFR")
print("Columns: predicted FR, NFR")

for _, row in validation_df.iterrows():
    print(f"\n{row['model']}")
    print([
        [int(row["tn"]), int(row["fp"])],
        [int(row["fn"]), int(row["tp"])],
    ])

report = [
    "D2 v1.1 LINEAR SVM BASELINE",
    "=" * 65,
    "",
    "Five-fold group-aware CV on train only",
    "Validation evaluated separately",
    "Test accessed: NO",
    "Decision threshold: 0.0",
    "C: 1.0",
    "",
    "CROSS-VALIDATION STABILITY",
    stability.round(4).to_string(index=False),
    "",
    "VALIDATION COMPARISON",
    validation_df[display_cols]
    .round(4).to_string(index=False),
    "",
    "No threshold optimization was performed.",
    "Decision scores are not calibrated probabilities.",
]

report_path = (
    REPORTS / "62_d2_svm_baseline_report.txt"
)

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

metadata_path = (
    REPORTS / "62_d2_svm_baseline_metadata.json"
)

metadata_path.write_text(
    json.dumps({
        "experiment": 62,
        "protocol": "D2-v1.1",
        "seed": SEED,
        "cv_folds": N_FOLDS,
        "manifest_sha256": sha256(MANIFEST),
        "models": list(MODELS),
        "C": 1.0,
        "decision_threshold": 0.0,
        "tfidf_fitted_on": (
            "TRAIN ONLY; independently per CV fold"
        ),
        "validation_evaluated": True,
        "test_accessed": False,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Linear SVM baseline: COMPLETED")
print("Group-aware CV: COMPLETED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
print(report_path)
print(metadata_path)

print("\nScript 62 completed.")
