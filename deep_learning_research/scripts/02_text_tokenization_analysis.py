
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

import numpy as np
import pandas as pd

# ============================================================
# SCRIPT 02: TEXT LENGTH AND TOKENIZATION ANALYSIS
#
# Dataset: D2-v1.1
#
# - Loads TRAIN and VALIDATION text only
# - Verifies frozen manifest SHA256
# - Analyzes word and character lengths
# - Uses the pretrained DistilBERT tokenizer
# - Evaluates potential truncation
# - Selects a preliminary sequence length using TRAIN only
#
# No test-text access.
# No model-weight download.
# No model training.
# No dataset modification.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROJECT = ROOT / "deep_learning_research"
DATA = ROOT / "dataset_exploration/outputs/validated"

TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
SPLIT_META = DATA / "54_d2_v11_split_metadata.json"

TEXT_COL = "Requirement Text (Original Requirement)"

TOKENIZER_NAME = "distilbert/distilbert-base-uncased"

CANDIDATE_LENGTHS = [64, 96, 128, 160, 192, 256]
TARGET_COVERAGE = 0.99

SUMMARY_PATH = TABLES / "02_text_length_summary.csv"
CLASS_PATH = TABLES / "02_length_by_class.csv"
TRUNCATION_PATH = TABLES / "02_truncation_analysis.csv"
RECORDS_PATH = TABLES / "02_train_validation_lengths.csv"

REPORT_TXT = REPORTS / "02_tokenization_report.txt"
REPORT_JSON = REPORTS / "02_tokenization_report.json"


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def verify(condition, message):
    if not condition:
        raise RuntimeError(message)


def describe_lengths(frame, column):
    values = frame[column].to_numpy()

    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 3),
        "std": round(float(np.std(values)), 3),
        "minimum": int(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": int(np.max(values)),
    }


print("=" * 76)
print("SCRIPT 02: D2 v1.1 TEXT TOKENIZATION ANALYSIS")
print("=" * 76)

# ------------------------------------------------------------
# 1. Verify frozen manifest
# ------------------------------------------------------------

metadata = json.loads(
    SPLIT_META.read_text(encoding="utf-8")
)

verify(
    metadata.get("protocol") == "D2-v1.1",
    "Unexpected dataset protocol"
)

verify(
    sha256(MANIFEST)
    == metadata["output_sha256"][MANIFEST.name],
    "Frozen manifest integrity failed"
)

# Only TRAIN and VALIDATION rows enter this analysis.
parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    selected = chunk[
        chunk["split"].isin(["train", "validation"])
    ].copy()

    if not selected.empty:
        parts.append(selected)

df = pd.concat(parts, ignore_index=True)

required = {
    "Requirement ID",
    TEXT_COL,
    "key",
    "label",
    "split",
    "similarity_group",
}

verify(
    required.issubset(df.columns),
    f"Missing columns: {required - set(df.columns)}"
)

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()

verify(len(train) == 3378, "Unexpected training count")
verify(len(validation) == 728, "Unexpected validation count")

verify(
    df["Requirement ID"].is_unique,
    "Duplicate requirement IDs"
)

verify(
    df["key"].is_unique,
    "Duplicate normalized text keys"
)

verify(
    not (
        set(train["similarity_group"])
        & set(validation["similarity_group"])
    ),
    "Train-validation group overlap"
)

verify(
    set(df["label"]) == {"FR", "NFR"},
    "Unexpected class labels"
)

verify(
    df[TEXT_COL].str.strip().ne("").all(),
    "Blank requirement text"
)

df = df.sort_values(
    ["split", "Requirement ID"]
).reset_index(drop=True)

print("\nDATASET")
print("Protocol: D2-v1.1")
print("Training requirements:", len(train))
print("Validation requirements:", len(validation))
print("Total analyzed:", len(df))
print("Test text loaded: NO")
print("Frozen manifest integrity: PASS")

# ------------------------------------------------------------
# 2. Character and word length analysis
# ------------------------------------------------------------

df["character_count"] = (
    df[TEXT_COL].str.len()
)

df["word_count"] = (
    df[TEXT_COL]
    .apply(lambda text: len(re.findall(r"\S+", text)))
)

print("\nTEXT LENGTH STATISTICS")

summary_rows = []

for split in ["train", "validation"]:
    subset = df[df["split"] == split]

    for column in ["character_count", "word_count"]:
        summary_rows.append({
            "split": split,
            "measure": column,
            **describe_lengths(subset, column)
        })

summary = pd.DataFrame(summary_rows)

print(summary.to_string(index=False))

# ------------------------------------------------------------
# 3. Load pretrained tokenizer
# ------------------------------------------------------------

print("\nTOKENIZER CONFIGURATION")
print("Tokenizer:", TOKENIZER_NAME)

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

except Exception as error:
    print("\nTOKENIZER LOAD FAILED")
    print(type(error).__name__ + ":", str(error))

    raise RuntimeError(
        "Cannot complete subword token analysis. "
        "Check tokenizer availability or internet access. "
        "No model weights were downloaded."
    ) from error

print("Fast tokenizer:", tokenizer.is_fast)
print("Model maximum length:", tokenizer.model_max_length)

# Tokenize without truncation or padding so lengths
# describe the complete original requirements.
encoded = tokenizer(
    df[TEXT_COL].tolist(),
    add_special_tokens=True,
    truncation=False,
    padding=False,
    return_attention_mask=False,
)

df["token_count"] = [
    len(ids) for ids in encoded["input_ids"]
]

print("\nTOKEN LENGTH STATISTICS")

token_rows = []

for split in ["train", "validation"]:
    subset = df[df["split"] == split]

    token_rows.append({
        "split": split,
        "measure": "token_count",
        **describe_lengths(subset, "token_count")
    })

token_summary = pd.DataFrame(token_rows)

print(token_summary.to_string(index=False))

# ------------------------------------------------------------
# 4. Class-level length analysis
# ------------------------------------------------------------

class_rows = []

for split in ["train", "validation"]:
    for label in ["FR", "NFR"]:

        subset = df[
            (df["split"] == split)
            & (df["label"] == label)
        ]

        class_rows.append({
            "split": split,
            "label": label,
            **{
                f"tokens_{key}": value
                for key, value in describe_lengths(
                    subset, "token_count"
                ).items()
            }
        })

class_summary = pd.DataFrame(class_rows)

print("\nTOKEN LENGTH BY CLASS")
print(class_summary.to_string(index=False))

# ------------------------------------------------------------
# 5. Truncation analysis
# ------------------------------------------------------------

truncation_rows = []

for split in ["train", "validation"]:
    for label in ["ALL", "FR", "NFR"]:

        if label == "ALL":
            subset = df[df["split"] == split]
        else:
            subset = df[
                (df["split"] == split)
                & (df["label"] == label)
            ]

        lengths = subset["token_count"].to_numpy()

        for max_length in CANDIDATE_LENGTHS:

            truncated = int(
                np.sum(lengths > max_length)
            )

            coverage = float(
                np.mean(lengths <= max_length)
            )

            truncation_rows.append({
                "split": split,
                "label": label,
                "max_length": max_length,
                "requirements": len(subset),
                "truncated_count": truncated,
                "truncated_percent": round(
                    truncated / len(subset) * 100,
                    3
                ),
                "coverage_percent": round(
                    coverage * 100,
                    3
                ),
            })

truncation = pd.DataFrame(truncation_rows)

print("\nTRUNCATION ANALYSIS")

overall = truncation[
    truncation["label"] == "ALL"
]

print(overall.to_string(index=False))

# ------------------------------------------------------------
# 6. Preliminary length recommendation
# ------------------------------------------------------------

# Use TRAIN only for the recommendation.
train_lengths = df.loc[
    df["split"] == "train",
    "token_count"
].to_numpy()

recommended = None

for length in CANDIDATE_LENGTHS:
    coverage = np.mean(
        train_lengths <= length
    )

    if coverage >= TARGET_COVERAGE:
        recommended = length
        break

print("\nPRELIMINARY LENGTH DECISION")
print("Target training coverage:", TARGET_COVERAGE)
print("Candidate lengths:", CANDIDATE_LENGTHS)

if recommended is None:
    print(
        "Recommendation: NO CANDIDATE REACHES TARGET"
    )
    print(
        "Inspect truncation results before deciding."
    )
else:
    print(
        "Recommended length based on training:",
        recommended
    )

    train_coverage = np.mean(
        train_lengths <= recommended
    )

    print(
        "Training coverage (%):",
        round(train_coverage * 100, 3)
    )

# ------------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------------

summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)

class_summary.to_csv(
    CLASS_PATH,
    index=False,
    encoding="utf-8-sig"
)

truncation.to_csv(
    TRUNCATION_PATH,
    index=False,
    encoding="utf-8-sig"
)

# Only IDs, split labels, and length measurements;
# no original requirement text or test records.
df[
    [
        "Requirement ID",
        "split",
        "label",
        "character_count",
        "word_count",
        "token_count",
    ]
].to_csv(
    RECORDS_PATH,
    index=False,
    encoding="utf-8-sig"
)

report_metadata = {
    "script": "02_text_tokenization_analysis",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "protocol": "D2-v1.1",
    "manifest_sha256": sha256(MANIFEST),
    "tokenizer": TOKENIZER_NAME,
    "transformers_version": None,
    "train_count": len(train),
    "validation_count": len(validation),
    "test_text_accessed": False,
    "test_predictions_generated": False,
    "candidate_lengths": CANDIDATE_LENGTHS,
    "target_train_coverage": TARGET_COVERAGE,
    "recommended_max_length": recommended,
    "training_only_recommendation": True,
    "model_training": False,
}

import transformers
report_metadata["transformers_version"] = (
    transformers.__version__
)

REPORT_JSON.write_text(
    json.dumps(report_metadata, indent=2),
    encoding="utf-8"
)

report_lines = [
    "SCRIPT 02: D2 v1.1 TOKENIZATION REPORT",
    "=" * 76,
    "",
    f"Tokenizer: {TOKENIZER_NAME}",
    f"Training requirements: {len(train)}",
    f"Validation requirements: {len(validation)}",
    "Test text accessed: NO",
    "",
    "TEXT LENGTH STATISTICS",
    summary.to_string(index=False),
    "",
    "TOKEN LENGTH STATISTICS",
    token_summary.to_string(index=False),
    "",
    "TOKEN LENGTH BY CLASS",
    class_summary.to_string(index=False),
    "",
    "TRUNCATION ANALYSIS",
    truncation.to_string(index=False),
    "",
    "PRELIMINARY LENGTH DECISION",
    f"Target coverage: {TARGET_COVERAGE}",
    f"Recommended max length: {recommended}",
    "",
    "LIMITATIONS",
    "- This analysis uses the DistilBERT tokenizer.",
    "- Other tokenizer families may produce different lengths.",
    "- Word counts are whitespace-based.",
    "- Training lengths determine the recommendation.",
    "- No model performance was measured.",
    "- No test requirements were analyzed.",
]

REPORT_TXT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Text length analysis: COMPLETED")
print("Tokenizer analysis: COMPLETED")
print("Maximum sequence length: PRELIMINARY")
print("Test set: SEALED")
print("Model training: NOT STARTED")
print("Dataset modified: NO")

print("\nFILES GENERATED")
for path in [
    SUMMARY_PATH,
    CLASS_PATH,
    TRUNCATION_PATH,
    RECORDS_PATH,
    REPORT_TXT,
    REPORT_JSON,
]:
    print(path)

print("\nScript 02 completed.")
