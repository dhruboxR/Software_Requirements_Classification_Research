
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# SCRIPT 57: D2 v1.1 BASELINE EXPERIMENT
#
# Models:
# B01: TF-IDF + Logistic Regression
# B02: TF-IDF + Balanced Logistic Regression
#
# Training and validation ONLY.
# No test-set evaluation.
# No threshold optimization.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "dataset_exploration/outputs"

MANIFEST = (
    DATA_ROOT / "validated"
    / "54_d2_v11_frozen_split_manifest.csv"
)

METADATA = (
    DATA_ROOT / "validated"
    / "54_d2_v11_split_metadata.json"
)

OUTPUT = ROOT / "research_modeling/outputs"

TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

for folder in (TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD = 0.50


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ------------------------------------------------------------
# 1. Verify frozen manifest integrity
# ------------------------------------------------------------

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

if metadata["protocol"] != "D2-v1.1":
    raise RuntimeError("Unexpected dataset protocol")

expected_hash = metadata["output_sha256"][MANIFEST.name]

if sha256(MANIFEST) != expected_hash:
    raise RuntimeError("Frozen manifest hash mismatch")

# Load only train/validation records into the
# modeling dataframe. Test records are discarded
# immediately and never used in model operations.
frames = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000
):
    selected = chunk[
        chunk["split"].isin(["train", "validation"])
    ].copy()

    frames.append(selected)

df = pd.concat(frames, ignore_index=True)

text_col = "Requirement Text (Original Requirement)"

required = {
    text_col,
    "Requirement ID",
    "label",
    "split",
    "similarity_group",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()

if len(train) != 3378 or len(val) != 728:
    raise RuntimeError("Unexpected train/validation sizes")

if set(train["similarity_group"]) & set(
    val["similarity_group"]
):
    raise RuntimeError(
        "Similarity-group overlap detected"
    )

if (
    train[text_col].str.strip().eq("").any()
    or val[text_col].str.strip().eq("").any()
):
    raise RuntimeError("Blank requirement text")

label_map = {"FR": 0, "NFR": 1}

y_train = train["label"].map(label_map).to_numpy()
y_val = val["label"].map(label_map).to_numpy()

if np.isnan(y_train).any() or np.isnan(y_val).any():
    raise RuntimeError("Unexpected class label")

y_train = y_train.astype(int)
y_val = y_val.astype(int)

print("=" * 76)
print("SCRIPT 57: D2 v1.1 BASELINE EXPERIMENT")
print("=" * 76)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements:", len(train))
print("Validation requirements:", len(val))
print("Test requirements accessed for modeling: 0")
print("Frozen manifest integrity: PASS")
print("Group isolation: PASS")

print("\nCLASS DISTRIBUTION")
print("Training:")
print(train["label"].value_counts().to_string())

print("\nValidation:")
print(val["label"].value_counts().to_string())

# ------------------------------------------------------------
# 2. Train-only TF-IDF feature extraction
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
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

X_train = vectorizer.fit_transform(
    train[text_col]
)

X_val = vectorizer.transform(
    val[text_col]
)

print("\nFEATURE EXTRACTION")
print("Method: Word TF-IDF")
print("N-grams: (1, 2)")
print("Vocabulary size:", X_train.shape[1])
print("TF-IDF fitted on: TRAIN ONLY")

# ------------------------------------------------------------
# 3. Fixed baseline configurations
# ------------------------------------------------------------

models = {
    "B01_LogisticRegression": LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=2000,
        solver="liblinear",
        random_state=SEED
    ),
    "B02_BalancedLogisticRegression": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=SEED
    ),
}

results = []
predictions = []

for name, model in models.items():

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    report = classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["FR", "NFR"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    )

    row = {
        "model": name,
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_val, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, y_pred
        ),
        "macro_f1": f1_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
        "fr_precision": report["FR"]["precision"],
        "fr_recall": report["FR"]["recall"],
        "fr_f1": report["FR"]["f1-score"],
        "nfr_precision": precision_score(
            y_val, y_pred,
            pos_label=1,
            zero_division=0
        ),
        "nfr_recall": recall_score(
            y_val, y_pred,
            pos_label=1,
            zero_division=0
        ),
        "nfr_f1": f1_score(
            y_val, y_pred,
            pos_label=1,
            zero_division=0
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "train_count": len(train),
        "validation_count": len(val),
    }

    results.append(row)

    prediction_df = pd.DataFrame({
        "requirement_id": val["Requirement ID"].to_numpy(),
        "true_label": val["label"].to_numpy(),
        "predicted_label": np.where(
            y_pred == 1, "NFR", "FR"
        ),
        "nfr_probability": probabilities,
        "model": name,
    })

    predictions.append(prediction_df)

    print(f"\n{name}")
    print("-" * 60)

    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["FR", "NFR"],
        digits=4,
        zero_division=0
    ))

    print("Macro-F1:", round(row["macro_f1"], 4))
    print(
        "Balanced accuracy:",
        round(row["balanced_accuracy"], 4)
    )
    print("Confusion matrix [FR, NFR]:")
    print(cm)

# ------------------------------------------------------------
# 4. Save validation-only results
# ------------------------------------------------------------

summary = pd.DataFrame(results)

summary = summary.sort_values(
    "model"
).reset_index(drop=True)

summary_path = (
    TABLES / "57_d2_baseline_metrics.csv"
)

predictions_path = (
    TABLES / "57_d2_validation_predictions.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

pd.concat(
    predictions,
    ignore_index=True
).to_csv(
    predictions_path,
    index=False
)

print("\n" + "=" * 76)
print("BASELINE COMPARISON")
print("=" * 76)

columns = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "nfr_precision",
    "nfr_recall",
    "nfr_f1",
    "fp",
    "fn"
]

print(
    summary[columns]
    .round(4)
    .to_string(index=False)
)

report_lines = [
    "D2 v1.1 BASELINE EXPERIMENT",
    "=" * 65,
    "",
    "Dataset protocol: D2-v1.1",
    f"Training examples: {len(train)}",
    f"Validation examples: {len(val)}",
    "Test examples used: 0",
    "",
    "TF-IDF: word (1,2), min_df=2",
    "Threshold: 0.50 (fixed)",
    "Model B01: Logistic Regression",
    "Model B02: Class-weighted Logistic Regression",
    "",
    summary[columns].round(4).to_string(index=False),
    "",
    "Research limitations:",
    "- Results are validation-only.",
    "- No transformer model was trained.",
    "- No threshold optimization was performed.",
    "- No test predictions were generated.",
    "- Independent-project generalization is not established.",
    "- Independent annotation verification was not performed.",
]

report_path = (
    REPORTS / "57_d2_baseline_report.txt"
)

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

run_metadata = {
    "experiment": 57,
    "dataset_protocol": "D2-v1.1",
    "seed": SEED,
    "manifest_sha256": sha256(MANIFEST),
    "train_examples": len(train),
    "validation_examples": len(val),
    "test_used": False,
    "vectorizer": {
        "type": "word_tfidf",
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
        "max_features": 100000
    },
    "models": {
        "B01": {
            "classifier": "LogisticRegression",
            "class_weight": None,
            "C": 1.0
        },
        "B02": {
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "C": 1.0
        }
    },
    "prediction_threshold": THRESHOLD,
    "evaluation_partition": "validation",
    "scientific_approval": "CONDITIONAL"
}

metadata_path = (
    REPORTS / "57_d2_baseline_metadata.json"
)

metadata_path.write_text(
    json.dumps(run_metadata, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Baseline experiments: COMPLETED")
print("Evaluation partition: VALIDATION ONLY")
print("Test set: SEALED")
print("Model selection: NOT FINALIZED")
print("Scientific approval: CONDITIONAL")

print("\nFILES GENERATED")
print(summary_path)
print(predictions_path)
print(report_path)
print(metadata_path)

print("\nScript 57 completed.")
