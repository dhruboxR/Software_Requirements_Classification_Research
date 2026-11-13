
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
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
# SCRIPT 68: D2 v1.1 TREE ENSEMBLE BASELINES
#
# T01: Random Forest, sparse word TF-IDF
# T02: Extra Trees, sparse word TF-IDF
# T03: HistGradientBoosting, TF-IDF + SVD
#
# Train-only group-aware 5-fold CV
# Validation evaluation
# Test set remains sealed
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "dataset_exploration/outputs/validated"
OUT = ROOT / "research_modeling/outputs"

TABLES = OUT / "tables"
REPORTS = OUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
METADATA = DATA / "54_d2_v11_split_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"

SEED = 42
N_FOLDS = 5
SVD_COMPONENTS = 100

MODEL_NAMES = [
    "T01_RandomForest",
    "T02_ExtraTrees",
    "T03_HistGradientBoosting",
]


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
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


def make_model(name):
    if name == "T01_RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1,
        )

    if name == "T02_ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1,
        )

    if name == "T03_HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=SEED,
        )

    raise ValueError(f"Unknown model: {name}")


def extract_features(train_texts, heldout_texts):
    vectorizer = make_vectorizer()

    X_train = vectorizer.fit_transform(train_texts)
    X_heldout = vectorizer.transform(heldout_texts)

    if min(X_train.shape) <= SVD_COMPONENTS:
        raise RuntimeError(
            "Insufficient dimensions for SVD"
        )

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        n_iter=5,
        random_state=SEED,
    )

    X_train_svd = svd.fit_transform(X_train)
    X_heldout_svd = svd.transform(X_heldout)

    return {
        "sparse": (X_train, X_heldout),
        "svd": (X_train_svd, X_heldout_svd),
    }


def evaluate(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "macro_f1": f1_score(
            y_true, predictions,
            average="macro",
            zero_division=0,
        ),
        "nfr_precision": precision_score(
            y_true, predictions,
            zero_division=0,
        ),
        "nfr_recall": recall_score(
            y_true, predictions,
            zero_division=0,
        ),
        "nfr_f1": f1_score(
            y_true, predictions,
            zero_division=0,
        ),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_and_evaluate(name, features, y_train, y_eval):
    representation = (
        "svd"
        if name == "T03_HistGradientBoosting"
        else "sparse"
    )

    X_train, X_eval = features[representation]

    model = make_model(name)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = model.predict(X_eval)

    metrics = evaluate(
        y_eval,
        predictions,
        probabilities,
    )

    return metrics, predictions, probabilities


# ------------------------------------------------------------
# 1. Load and verify training/validation data
# ------------------------------------------------------------

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError("Unexpected protocol")

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
    parts.append(
        chunk.loc[
            chunk["split"].isin(
                ["train", "validation"]
            )
        ].copy()
    )

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL,
    "Requirement ID",
    "key",
    "label",
    "similarity_group",
    "split",
}

if not required.issubset(df.columns):
    raise RuntimeError(
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
    raise RuntimeError("Duplicate training IDs")

if not val["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate validation IDs")

if (
    set(train["similarity_group"])
    & set(val["similarity_group"])
):
    raise RuntimeError("Train/validation group leakage")

label_map = {"FR": 0, "NFR": 1}

train_labels = train["label"].map(label_map)
val_labels = val["label"].map(label_map)

if train_labels.isna().any() or val_labels.isna().any():
    raise RuntimeError("Unexpected class labels")

y_train = train_labels.to_numpy(dtype=int)
y_val = val_labels.to_numpy(dtype=int)

texts_train = train[TEXT_COL].to_numpy()
texts_val = val[TEXT_COL].to_numpy()

if any(not str(t).strip() for t in texts_train):
    raise RuntimeError("Blank training text")

if any(not str(t).strip() for t in texts_val):
    raise RuntimeError("Blank validation text")

groups = train["similarity_group"].to_numpy()

print("=" * 76)
print("SCRIPT 68: D2 v1.1 TREE ENSEMBLE BASELINES")
print("=" * 76)

print("\nDATASET")
print("Training:", len(train))
print("Validation:", len(val))
print("Training groups:", len(set(groups)))
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

# ------------------------------------------------------------
# 2. Group-aware cross-validation
# ------------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

fold_rows = []
oof_rows = []

for fold, (tr_idx, held_idx) in enumerate(
    cv.split(texts_train, y_train, groups),
    start=1,
):
    if set(groups[tr_idx]) & set(groups[held_idx]):
        raise RuntimeError("CV group leakage")

    if (
        len(np.unique(y_train[tr_idx])) != 2
        or len(np.unique(y_train[held_idx])) != 2
    ):
        raise RuntimeError("Missing class in CV fold")

    features = extract_features(
        texts_train[tr_idx],
        texts_train[held_idx],
    )

    for name in MODEL_NAMES:
        metrics, predicted, probs = fit_and_evaluate(
            name,
            features,
            y_train[tr_idx],
            y_train[held_idx],
        )

        fold_rows.append({
            "fold": fold,
            "model": name,
            "train_size": len(tr_idx),
            "heldout_size": len(held_idx),
            **metrics,
        })

        for idx, pred, prob in zip(
            held_idx, predicted, probs
        ):
            oof_rows.append({
                "model": name,
                "fold": fold,
                "requirement_id": train.loc[
                    idx, "Requirement ID"
                ],
                "true_label": int(y_train[idx]),
                "predicted_label": int(pred),
                "nfr_probability": float(prob),
            })

        print(
            f"Fold {fold} | {name}: "
            f"NFR F1={metrics['nfr_f1']:.4f}, "
            f"Recall={metrics['nfr_recall']:.4f}"
        )

fold_df = pd.DataFrame(fold_rows)
oof_df = pd.DataFrame(oof_rows)

for name in MODEL_NAMES:
    subset = oof_df[oof_df["model"] == name]

    if (
        len(subset) != len(train)
        or not subset["requirement_id"].is_unique
    ):
        raise RuntimeError(
            f"Invalid OOF coverage: {name}"
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

features = extract_features(
    texts_train, texts_val
)

validation_rows = []
prediction_rows = []

for name in MODEL_NAMES:
    metrics, predicted, probs = fit_and_evaluate(
        name,
        features,
        y_train,
        y_val,
    )

    validation_rows.append({
        "model": name,
        **metrics,
    })

    for idx, prediction, probability in zip(
        range(len(val)), predicted, probs
    ):
        prediction_rows.append({
            "model": name,
            "requirement_id": val.loc[
                idx, "Requirement ID"
            ],
            "true_label": val.loc[idx, "label"],
            "predicted_label": (
                "NFR" if prediction == 1 else "FR"
            ),
            "nfr_probability": float(probability),
        })

validation_df = pd.DataFrame(validation_rows)

# ------------------------------------------------------------
# 4. Save outputs and report
# ------------------------------------------------------------

paths = {
    "fold": TABLES / "68_d2_tree_cv_fold_metrics.csv",
    "oof": TABLES / "68_d2_tree_oof_predictions.csv",
    "stability": TABLES / "68_d2_tree_cv_stability.csv",
    "validation": TABLES / "68_d2_tree_validation_metrics.csv",
    "predictions": TABLES / "68_d2_tree_validation_predictions.csv",
}

fold_df.to_csv(paths["fold"], index=False)
oof_df.to_csv(paths["oof"], index=False)
stability.to_csv(paths["stability"], index=False)
validation_df.to_csv(paths["validation"], index=False)

pd.DataFrame(prediction_rows).to_csv(
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

report_path = (
    REPORTS / "68_d2_tree_baseline_report.txt"
)

report_path.write_text(
    "\n".join([
        "SCRIPT 68: D2 v1.1 TREE ENSEMBLE BASELINES",
        "=" * 65,
        "",
        "Five-fold group-aware training CV",
        "Validation evaluated separately",
        "Test accessed: NO",
        "",
        "CROSS-VALIDATION STABILITY",
        stability.round(4).to_string(index=False),
        "",
        "VALIDATION COMPARISON",
        validation_df[columns]
        .round(4).to_string(index=False),
        "",
        "LIMITATIONS",
        "- Model pipelines use different representations.",
        "- SVD is fitted only on training data per fold.",
        "- Validation has been examined in earlier experiments.",
        "- No model or threshold tuning was performed.",
        "- Project independence is not established.",
        "- Test partition remains sealed.",
    ]) + "\n",
    encoding="utf-8",
)

metadata_path = (
    REPORTS / "68_d2_tree_baseline_metadata.json"
)

metadata_path.write_text(
    json.dumps({
        "experiment": 68,
        "protocol": "D2-v1.1",
        "seed": SEED,
        "folds": N_FOLDS,
        "manifest_sha256": sha256(MANIFEST),
        "models": MODEL_NAMES,
        "svd_components": SVD_COMPONENTS,
        "svd_fit_within_fold": True,
        "test_accessed": False,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Tree ensemble baselines: COMPLETED")
print("Group-aware CV: COMPLETED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in paths.values():
    print(path)
print(report_path)
print(metadata_path)
print("\nScript 68 completed.")
