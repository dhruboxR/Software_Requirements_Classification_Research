
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
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
# SCRIPT 64: ADDITIONAL CLASSICAL ML BASELINES
#
# M01: Multinomial Naive Bayes
# M02: Complement Naive Bayes
# M03: SGD Hinge
# M04: SGD Modified Huber
#
# D2 v1.1 frozen protocol
# 5-fold group-aware CV on TRAIN only
# Separate VALIDATION evaluation
# TEST remains sealed
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "dataset_exploration/outputs/validated"
OUT_DIR = ROOT / "research_modeling/outputs"

TABLES = OUT_DIR / "tables"
REPORTS = OUT_DIR / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA_DIR / "54_d2_v11_frozen_split_manifest.csv"
META = DATA_DIR / "54_d2_v11_split_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"

SEED = 42
FOLDS = 5

MODELS = [
    "M01_MultinomialNB",
    "M02_ComplementNB",
    "M03_SGD_Hinge",
    "M04_SGD_ModifiedHuber",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def vectorizer_factory():
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


def model_factory(name):
    if name == "M01_MultinomialNB":
        return MultinomialNB(alpha=1.0)

    if name == "M02_ComplementNB":
        return ComplementNB(alpha=1.0)

    if name == "M03_SGD_Hinge":
        return SGDClassifier(
            loss="hinge",
            alpha=0.0001,
            penalty="l2",
            max_iter=3000,
            tol=1e-4,
            random_state=SEED,
        )

    if name == "M04_SGD_ModifiedHuber":
        return SGDClassifier(
            loss="modified_huber",
            alpha=0.0001,
            penalty="l2",
            max_iter=3000,
            tol=1e-4,
            random_state=SEED,
        )

    raise ValueError(f"Unknown model: {name}")


def get_scores(model, X):
    # Every listed classifier has decision_function or
    # predict_proba. Higher scores must mean more NFR.
    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict_proba(X)[:, 1]


def evaluate(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "nfr_precision": precision_score(
            y_true, y_pred,
            zero_division=0,
        ),
        "nfr_recall": recall_score(
            y_true, y_pred,
            zero_division=0,
        ),
        "nfr_f1": f1_score(
            y_true, y_pred,
            zero_division=0,
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
# 1. Verify frozen protocol
# ------------------------------------------------------------

metadata = json.loads(
    META.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError("Wrong dataset protocol")

if sha256(MANIFEST) != metadata[
    "output_sha256"
][MANIFEST.name]:
    raise RuntimeError("Frozen manifest hash mismatch")

# Only train and validation records enter this experiment.
chunks = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    chunks.append(
        chunk[
            chunk["split"].isin(
                ["train", "validation"]
            )
        ].copy()
    )

df = pd.concat(chunks, ignore_index=True)

required = {
    TEXT_COL, "Requirement ID",
    "label", "split",
    "similarity_group", "key",
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

if not train["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate train requirement IDs")

if not val["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate validation requirement IDs")

if (
    set(train["similarity_group"])
    & set(val["similarity_group"])
):
    raise RuntimeError("Train/validation group overlap")

label_map = {"FR": 0, "NFR": 1}

y_train_series = train["label"].map(label_map)
y_val_series = val["label"].map(label_map)

if y_train_series.isna().any() or y_val_series.isna().any():
    raise RuntimeError("Unknown class label")

y_train = y_train_series.to_numpy(dtype=int)
y_val = y_val_series.to_numpy(dtype=int)

texts_train = train[TEXT_COL].astype(str).to_numpy()
texts_val = val[TEXT_COL].astype(str).to_numpy()

if any(not text.strip() for text in texts_train):
    raise RuntimeError("Blank training text")

if any(not text.strip() for text in texts_val):
    raise RuntimeError("Blank validation text")

groups = train["similarity_group"].to_numpy()

print("=" * 76)
print("SCRIPT 64: ADDITIONAL CLASSICAL ML BASELINES")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements:", len(train))
print("Validation requirements:", len(val))
print("Training similarity groups:", len(set(groups)))
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

# ------------------------------------------------------------
# 2. Group-aware training cross-validation
# ------------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=SEED,
)

fold_results = []
oof_rows = []

for fold, (tr_idx, hold_idx) in enumerate(
    cv.split(texts_train, y_train, groups),
    start=1,
):
    if set(groups[tr_idx]) & set(groups[hold_idx]):
        raise RuntimeError("Cross-validation group leakage")

    if len(np.unique(y_train[tr_idx])) != 2:
        raise RuntimeError("Missing training class")

    if len(np.unique(y_train[hold_idx])) != 2:
        raise RuntimeError("Missing held-out class")

    vectorizer = vectorizer_factory()

    X_tr = vectorizer.fit_transform(
        texts_train[tr_idx]
    )
    X_hold = vectorizer.transform(
        texts_train[hold_idx]
    )

    for model_name in MODELS:
        model = model_factory(model_name)
        model.fit(X_tr, y_train[tr_idx])

        predictions = model.predict(X_hold)
        scores = get_scores(model, X_hold)

        result = evaluate(
            y_train[hold_idx],
            predictions,
            scores,
        )

        fold_results.append({
            "fold": fold,
            "model": model_name,
            "train_size": len(tr_idx),
            "heldout_size": len(hold_idx),
            "heldout_nfr": int(
                (y_train[hold_idx] == 1).sum()
            ),
            **result,
        })

        for idx, prediction, score in zip(
            hold_idx, predictions, scores
        ):
            oof_rows.append({
                "model": model_name,
                "fold": fold,
                "requirement_id": train.loc[
                    idx, "Requirement ID"
                ],
                "true_label": int(y_train[idx]),
                "predicted_label": int(prediction),
                "ranking_score": float(score),
            })

        print(
            f"Fold {fold} | {model_name}: "
            f"NFR F1={result['nfr_f1']:.4f}, "
            f"Recall={result['nfr_recall']:.4f}"
        )

fold_df = pd.DataFrame(fold_results)
oof_df = pd.DataFrame(oof_rows)

for model_name in MODELS:
    part = oof_df[
        oof_df["model"] == model_name
    ]

    if (
        len(part) != len(train)
        or not part["requirement_id"].is_unique
    ):
        raise RuntimeError(
            f"Invalid OOF coverage: {model_name}"
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
# 3. Train full models and evaluate validation
# ------------------------------------------------------------

vectorizer = vectorizer_factory()

X_train = vectorizer.fit_transform(texts_train)
X_val = vectorizer.transform(texts_val)

validation_results = []
validation_predictions = []

for model_name in MODELS:
    model = model_factory(model_name)

    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    score = get_scores(model, X_val)

    result = evaluate(y_val, pred, score)

    validation_results.append({
        "model": model_name,
        **result,
    })

    for i, (prediction, ranking_score) in enumerate(
        zip(pred, score)
    ):
        validation_predictions.append({
            "model": model_name,
            "requirement_id": val.loc[
                i, "Requirement ID"
            ],
            "true_label": val.loc[i, "label"],
            "predicted_label": (
                "NFR" if prediction == 1 else "FR"
            ),
            "ranking_score": float(ranking_score),
        })

validation_df = pd.DataFrame(validation_results)

# ------------------------------------------------------------
# 4. Save outputs
# ------------------------------------------------------------

paths = {
    "fold": TABLES / "64_d2_ml_cv_fold_metrics.csv",
    "oof": TABLES / "64_d2_ml_oof_predictions.csv",
    "stability": TABLES / "64_d2_ml_cv_stability.csv",
    "validation": TABLES / "64_d2_ml_validation_metrics.csv",
    "predictions": TABLES / "64_d2_ml_validation_predictions.csv",
}

fold_df.to_csv(paths["fold"], index=False)
oof_df.to_csv(paths["oof"], index=False)
stability.to_csv(paths["stability"], index=False)
validation_df.to_csv(paths["validation"], index=False)

pd.DataFrame(validation_predictions).to_csv(
    paths["predictions"], index=False
)

columns = [
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
    validation_df[columns]
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
    "D2 v1.1 ADDITIONAL CLASSICAL ML BASELINES",
    "=" * 65,
    "",
    "Train-only, 5-fold group-aware CV",
    "Separate validation evaluation",
    "Test accessed: NO",
    "No threshold optimization",
    "",
    "CROSS-VALIDATION STABILITY",
    stability.round(4).to_string(index=False),
    "",
    "VALIDATION COMPARISON",
    validation_df[columns]
    .round(4).to_string(index=False),
    "",
    "LIMITATIONS",
    "- Validation has been used for prior model comparisons.",
    "- No hyperparameter search was conducted.",
    "- Group isolation does not establish project independence.",
    "- Ranking scores are not necessarily probabilities.",
    "- Test partition remains sealed.",
]

report_path = REPORTS / "64_d2_ml_baseline_report.txt"

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

metadata_path = REPORTS / "64_d2_ml_baseline_metadata.json"

metadata_path.write_text(
    json.dumps({
        "experiment": 64,
        "protocol": "D2-v1.1",
        "seed": SEED,
        "cv_folds": FOLDS,
        "manifest_sha256": sha256(MANIFEST),
        "models": MODELS,
        "tfidf": {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "fitted_separately_per_fold": True,
        },
        "validation_evaluated": True,
        "test_accessed": False,
        "threshold_tuning": False,
        "final_model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Additional ML baselines: COMPLETED")
print("Group-aware CV: COMPLETED")
print("Validation evaluation: COMPLETED")
print("Threshold optimization: NONE")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in paths.values():
    print(path)
print(report_path)
print(metadata_path)

print("\nScript 64 completed.")
