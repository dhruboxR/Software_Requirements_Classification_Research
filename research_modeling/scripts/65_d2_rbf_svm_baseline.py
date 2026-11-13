
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# ==========================================================
# SCRIPT 65: D2 v1.1 RBF SVM BASELINE
#
# R01: Standard RBF SVM
# R02: Balanced RBF SVM
#
# Train-only 5-fold grouped CV
# Separate validation evaluation
# No threshold search
# No test evaluation
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "dataset_exploration/outputs/validated"
OUTPUT = ROOT / "research_modeling/outputs"

TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
METADATA = DATA / "54_d2_v11_split_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"
SEED = 42
N_FOLDS = 5

MODELS = {
    "R01_RBF_SVM": None,
    "R02_Balanced_RBF_SVM": "balanced",
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
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


def make_model(class_weight):
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight=class_weight,
        probability=False,
        cache_size=512,
        random_state=SEED,
    )


def evaluate(y_true, decision_scores):
    y_pred = (decision_scores >= 0).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "nfr_precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "nfr_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "nfr_f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_true, decision_scores
        ),
        "roc_auc": roc_auc_score(
            y_true, decision_scores
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ----------------------------------------------------------
# 1. Verify frozen protocol
# ----------------------------------------------------------

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError("Wrong dataset protocol")

if sha256(MANIFEST) != metadata[
    "output_sha256"
][MANIFEST.name]:
    raise RuntimeError("Frozen manifest hash mismatch")

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
            chunk["split"].isin(["train", "validation"])
        ].copy()
    )

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL,
    "Requirement ID",
    "key",
    "label",
    "split",
    "similarity_group",
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
    raise RuntimeError("Unexpected partition size")

if not train["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate training IDs")

if not val["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate validation IDs")

if set(train["similarity_group"]) & set(
    val["similarity_group"]
):
    raise RuntimeError("Similarity-group overlap")

label_map = {"FR": 0, "NFR": 1}

y_train = train["label"].map(label_map)
y_val = val["label"].map(label_map)

if y_train.isna().any() or y_val.isna().any():
    raise RuntimeError("Unknown class label")

y_train = y_train.to_numpy(dtype=int)
y_val = y_val.to_numpy(dtype=int)

texts_train = train[TEXT_COL].astype(str).to_numpy()
texts_val = val[TEXT_COL].astype(str).to_numpy()

if any(not text.strip() for text in texts_train):
    raise RuntimeError("Blank training text")

if any(not text.strip() for text in texts_val):
    raise RuntimeError("Blank validation text")

groups = train["similarity_group"].to_numpy()

print("=" * 76)
print("SCRIPT 65: D2 v1.1 RBF SVM BASELINE")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements:", len(train))
print("Validation requirements:", len(val))
print("Training groups:", len(set(groups)))
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

print("\nCONFIGURATION")
print("Kernel: RBF")
print("C: 1.0")
print("Gamma: scale")
print("CV folds:", N_FOLDS)
print("Threshold optimization: NONE")

# ----------------------------------------------------------
# 2. Group-aware cross-validation
# ----------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

fold_rows = []
oof_rows = []

for fold, (tr_idx, hold_idx) in enumerate(
    cv.split(texts_train, y_train, groups),
    start=1,
):
    if set(groups[tr_idx]) & set(groups[hold_idx]):
        raise RuntimeError("Cross-validation group leakage")

    if (
        len(np.unique(y_train[tr_idx])) != 2
        or len(np.unique(y_train[hold_idx])) != 2
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
        model = make_model(weight)
        model.fit(X_tr, y_train[tr_idx])

        scores = model.decision_function(X_hold)

        result = evaluate(
            y_train[hold_idx], scores
        )

        fold_rows.append({
            "fold": fold,
            "model": name,
            "train_size": len(tr_idx),
            "heldout_size": len(hold_idx),
            **result,
        })

        for idx, score in zip(hold_idx, scores):
            oof_rows.append({
                "model": name,
                "fold": fold,
                "requirement_id": train.loc[
                    idx, "Requirement ID"
                ],
                "true_label": int(y_train[idx]),
                "decision_score": float(score),
            })

        print(
            f"Fold {fold} | {name}: "
            f"NFR F1={result['nfr_f1']:.4f}, "
            f"Recall={result['nfr_recall']:.4f}"
        )

fold_df = pd.DataFrame(fold_rows)
oof_df = pd.DataFrame(oof_rows)

for name in MODELS:
    rows = oof_df[oof_df["model"] == name]

    if (
        len(rows) != len(train)
        or not rows["requirement_id"].is_unique
    ):
        raise RuntimeError(
            f"Incomplete OOF coverage: {name}"
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

# ----------------------------------------------------------
# 3. Full training and validation evaluation
# ----------------------------------------------------------

vectorizer = make_vectorizer()

X_train = vectorizer.fit_transform(texts_train)
X_val = vectorizer.transform(texts_val)

validation_rows = []
prediction_rows = []

for name, weight in MODELS.items():
    model = make_model(weight)
    model.fit(X_train, y_train)

    scores = model.decision_function(X_val)
    pred = (scores >= 0).astype(int)

    result = evaluate(y_val, scores)

    validation_rows.append({
        "model": name,
        "decision_threshold": 0.0,
        **result,
    })

    for i, (score, prediction) in enumerate(
        zip(scores, pred)
    ):
        prediction_rows.append({
            "model": name,
            "requirement_id": val.loc[
                i, "Requirement ID"
            ],
            "true_label": val.loc[i, "label"],
            "predicted_label": (
                "NFR" if prediction == 1 else "FR"
            ),
            "decision_score": float(score),
        })

validation_df = pd.DataFrame(validation_rows)

# ----------------------------------------------------------
# 4. Save experiment outputs
# ----------------------------------------------------------

outputs = {
    "fold": TABLES / "65_d2_rbf_cv_fold_metrics.csv",
    "oof": TABLES / "65_d2_rbf_oof_predictions.csv",
    "stability": TABLES / "65_d2_rbf_cv_stability.csv",
    "validation": TABLES / "65_d2_rbf_validation_metrics.csv",
    "predictions": TABLES / "65_d2_rbf_validation_predictions.csv",
}

fold_df.to_csv(outputs["fold"], index=False)
oof_df.to_csv(outputs["oof"], index=False)
stability.to_csv(outputs["stability"], index=False)
validation_df.to_csv(outputs["validation"], index=False)

pd.DataFrame(prediction_rows).to_csv(
    outputs["predictions"], index=False
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
    .round(4).to_string(index=False)
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
    "SCRIPT 65: D2 v1.1 RBF SVM BASELINE",
    "=" * 65,
    "",
    "Train-only 5-fold group-aware CV",
    "Separate validation evaluation",
    "No threshold search",
    "Test accessed: NO",
    "",
    "CROSS-VALIDATION STABILITY",
    stability.round(4).to_string(index=False),
    "",
    "VALIDATION COMPARISON",
    validation_df[columns].round(4).to_string(index=False),
    "",
    "LIMITATIONS",
    "- Existing validation partition reused for comparison.",
    "- Original-project independence is not established.",
    "- Decision scores are not calibrated probabilities.",
    "- No hyperparameter optimization performed.",
    "- Test partition remains sealed.",
]

report_path = REPORTS / "65_d2_rbf_baseline_report.txt"

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

metadata_path = (
    REPORTS / "65_d2_rbf_baseline_metadata.json"
)

metadata_path.write_text(
    json.dumps({
        "experiment": 65,
        "protocol": "D2-v1.1",
        "manifest_sha256": sha256(MANIFEST),
        "seed": SEED,
        "cv_folds": N_FOLDS,
        "models": list(MODELS),
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "tfidf_fitted_within_each_fold": True,
        "test_used": False,
        "threshold_optimized": False,
        "final_selection": "PENDING",
    }, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("RBF SVM experiment: COMPLETED")
print("Group-aware CV: COMPLETED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in outputs.values():
    print(path)
print(report_path)
print(metadata_path)

print("\nScript 65 completed.")
