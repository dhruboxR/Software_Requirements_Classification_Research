
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
# SCRIPT 66: WORD + CHARACTER TF-IDF EXPERIMENT
#
# F01: Character TF-IDF + Linear SVM
# F02: Character TF-IDF + Balanced Linear SVM
# F03: Combined TF-IDF + Linear SVM
# F04: Combined TF-IDF + Balanced Linear SVM
#
# Group-aware CV on TRAIN
# Separate validation evaluation
# TEST remains sealed
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset_exploration/outputs/validated"
OUT = ROOT / "research_modeling/outputs"

TABLES = OUT / "tables"
REPORTS = OUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
META = DATA / "54_d2_v11_split_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"
SEED = 42
FOLDS = 5

CONFIGS = {
    "F01_CharacterSVM": ("char", None),
    "F02_BalancedCharacterSVM": ("char", "balanced"),
    "F03_CombinedSVM": ("combined", None),
    "F04_BalancedCombinedSVM": ("combined", "balanced"),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def make_word_vectorizer():
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


def make_char_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        max_features=100000,
        norm="l2",
    )


def extract_features(train_texts, eval_texts):
    word = make_word_vectorizer()
    char = make_char_vectorizer()

    word_train = word.fit_transform(train_texts)
    word_eval = word.transform(eval_texts)

    char_train = char.fit_transform(train_texts)
    char_eval = char.transform(eval_texts)

    # Equal contribution of the two feature blocks,
    # followed by row normalization.
    combined_train = normalize(
        hstack(
            [word_train, char_train],
            format="csr",
        ),
        norm="l2",
    )

    combined_eval = normalize(
        hstack(
            [word_eval, char_eval],
            format="csr",
        ),
        norm="l2",
    )

    return {
        "char": (char_train, char_eval),
        "combined": (combined_train, combined_eval),
    }, {
        "word_features": word_train.shape[1],
        "character_features": char_train.shape[1],
    }


def evaluate(y_true, scores):
    predicted = (scores >= 0).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predicted
        ),
        "macro_f1": f1_score(
            y_true, predicted, average="macro",
            zero_division=0,
        ),
        "nfr_precision": precision_score(
            y_true, predicted, zero_division=0
        ),
        "nfr_recall": recall_score(
            y_true, predicted, zero_division=0
        ),
        "nfr_f1": f1_score(
            y_true, predicted, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_true, scores
        ),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_model(weight):
    return LinearSVC(
        C=1.0,
        class_weight=weight,
        max_iter=10000,
        random_state=SEED,
    )


# ----------------------------------------------------------
# 1. Verify and load data
# ----------------------------------------------------------

metadata = json.loads(META.read_text("utf-8"))

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
    TEXT_COL, "Requirement ID", "key",
    "label", "split", "similarity_group"
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

if set(train["similarity_group"]) & set(
    val["similarity_group"]
):
    raise RuntimeError("Similarity group overlap")

if (
    not train["Requirement ID"].is_unique
    or not val["Requirement ID"].is_unique
):
    raise RuntimeError("Duplicate IDs")

labels = {"FR": 0, "NFR": 1}

y_train = train["label"].map(labels)
y_val = val["label"].map(labels)

if y_train.isna().any() or y_val.isna().any():
    raise RuntimeError("Unknown class labels")

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
print("SCRIPT 66: WORD + CHARACTER TF-IDF EXPERIMENT")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training:", len(train))
print("Validation:", len(val))
print("Training groups:", len(set(groups)))
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

# ----------------------------------------------------------
# 2. Five-fold group-aware CV
# ----------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=FOLDS,
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
        raise RuntimeError("CV group overlap")

    if (
        len(np.unique(y_train[tr_idx])) != 2
        or len(np.unique(y_train[hold_idx])) != 2
    ):
        raise RuntimeError("Missing class in fold")

    features, dimensions = extract_features(
        texts_train[tr_idx],
        texts_train[hold_idx],
    )

    for name, (representation, weight) in CONFIGS.items():
        X_tr, X_hold = features[representation]

        model = make_model(weight)
        model.fit(X_tr, y_train[tr_idx])

        scores = model.decision_function(X_hold)
        result = evaluate(y_train[hold_idx], scores)

        fold_rows.append({
            "fold": fold,
            "model": name,
            "representation": representation,
            "train_size": len(tr_idx),
            "heldout_size": len(hold_idx),
            **dimensions,
            **result,
        })

        for idx, score in zip(hold_idx, scores):
            oof_rows.append({
                "fold": fold,
                "model": name,
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

for name in CONFIGS:
    part = oof_df[oof_df["model"] == name]

    if (
        len(part) != len(train)
        or not part["requirement_id"].is_unique
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

# ----------------------------------------------------------
# 3. Full training and validation
# ----------------------------------------------------------

features, dimensions = extract_features(
    texts_train, texts_val
)

validation_rows = []
validation_predictions = []

for name, (representation, weight) in CONFIGS.items():
    X_tr, X_val = features[representation]

    model = make_model(weight)
    model.fit(X_tr, y_train)

    scores = model.decision_function(X_val)
    predictions = (scores >= 0).astype(int)
    result = evaluate(y_val, scores)

    validation_rows.append({
        "model": name,
        "representation": representation,
        **result,
    })

    for i, (score, prediction) in enumerate(
        zip(scores, predictions)
    ):
        validation_predictions.append({
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
# 4. Save experiment records
# ----------------------------------------------------------

paths = {
    "fold_metrics": TABLES / "66_d2_cv_fold_metrics.csv",
    "oof": TABLES / "66_d2_oof_predictions.csv",
    "stability": TABLES / "66_d2_cv_stability.csv",
    "validation": TABLES / "66_d2_validation_metrics.csv",
    "predictions": TABLES / "66_d2_validation_predictions.csv",
}

fold_df.to_csv(paths["fold_metrics"], index=False)
oof_df.to_csv(paths["oof"], index=False)
stability.to_csv(paths["stability"], index=False)
validation_df.to_csv(paths["validation"], index=False)

pd.DataFrame(validation_predictions).to_csv(
    paths["predictions"], index=False
)

columns = [
    "model", "accuracy", "balanced_accuracy",
    "macro_f1", "nfr_precision", "nfr_recall",
    "nfr_f1", "average_precision", "roc_auc",
    "fp", "fn"
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
    "SCRIPT 66: WORD + CHARACTER TF-IDF EXPERIMENT",
    "=" * 65,
    "",
    f"Training: {len(train)}",
    f"Validation: {len(val)}",
    f"CV folds: {FOLDS}",
    "Test accessed: NO",
    "",
    "CONFIGURATION",
    "Character: char_wb (3,5), max_features=100000",
    "Word: word (1,2), max_features=100000",
    "Combined: concatenated and L2 normalized",
    "LinearSVC: C=1.0, decision threshold=0",
    "",
    "CROSS-VALIDATION STABILITY",
    stability.round(4).to_string(index=False),
    "",
    "VALIDATION COMPARISON",
    validation_df[columns].round(4).to_string(index=False),
    "",
    "LIMITATIONS",
    "- Validation partition previously used for comparisons.",
    "- Group isolation does not prove project independence.",
    "- No hyperparameter tuning performed.",
    "- Decision scores are not calibrated probabilities.",
    "- Frozen test partition remains sealed.",
]

report_path = REPORTS / "66_d2_feature_experiment.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

metadata_path = REPORTS / "66_d2_feature_metadata.json"
metadata_path.write_text(
    json.dumps({
        "experiment": 66,
        "protocol": "D2-v1.1",
        "manifest_sha256": sha256(MANIFEST),
        "seed": SEED,
        "cv_folds": FOLDS,
        "models": list(CONFIGS),
        "character_analyzer": "char_wb",
        "character_ngram_range": [3, 5],
        "word_ngram_range": [1, 2],
        "combined_normalization": "l2",
        "features_fit_within_each_fold": True,
        "test_used": False,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8"
)

print("\nFEATURE DIMENSIONS")
print("Word features:", dimensions["word_features"])
print("Character features:", dimensions["character_features"])

print("\nRESEARCH STATUS")
print("Feature experiment: COMPLETED")
print("Group-aware CV: COMPLETED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in paths.values():
    print(path)
print(report_path)
print(metadata_path)
print("\nScript 66 completed.")
