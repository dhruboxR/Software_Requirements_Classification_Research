
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

# ==========================================================
# SCRIPT 11: PERSISTENT TRANSFORMER ERROR ANALYSIS
#
# Uses existing Script 10 prediction details.
# Inspects validation text only.
# Preserves dataset labels and partitions.
#
# No training, no test data, no label modifications.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deep_learning_research"

DATA = ROOT / "dataset_exploration/outputs/validated"
MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
META = DATA / "54_d2_v11_split_metadata.json"

DETAILS = (
    PROJECT / "outputs/tables"
    / "10_multiseed_requirement_details.csv"
)

TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Requirement Text (Original Requirement)"
MODELS = ["distilbert", "bert", "roberta"]

SUMMARY_PATH = TABLES / "11_error_category_summary.csv"
REVIEW_PATH = TABLES / "11_persistent_error_review.csv"
REPORT_PATH = REPORTS / "11_persistent_error_report.txt"
META_PATH = REPORTS / "11_persistent_error_metadata.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1048576), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def verify(condition, message):
    if not condition:
        raise RuntimeError(message)


print("=" * 76)
print("SCRIPT 11: PERSISTENT TRANSFORMER ERROR ANALYSIS")
print("=" * 76)

# ----------------------------------------------------------
# 1. Verify frozen dataset
# ----------------------------------------------------------

metadata = json.loads(
    META.read_text(encoding="utf-8")
)

verify(
    metadata.get("protocol") == "D2-v1.1",
    "Unexpected protocol",
)

manifest_hash = sha256(MANIFEST)

verify(
    manifest_hash == metadata[
        "output_sha256"
    ][MANIFEST.name],
    "Manifest SHA256 mismatch",
)

# Only validation records are retained.
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
        [
            "Requirement ID",
            TEXT_COL,
            "label",
            "similarity_group",
        ],
    ].copy()

    if not selected.empty:
        parts.append(selected)

validation = pd.concat(
    parts,
    ignore_index=True,
).rename(columns={
    "Requirement ID": "requirement_id",
    "label": "true_label",
})

verify(len(validation) == 728, "Wrong validation count")
verify(
    validation["requirement_id"].is_unique,
    "Duplicate validation IDs",
)
verify(
    validation["similarity_group"].nunique() == 694,
    "Wrong similarity-group count",
)

# ----------------------------------------------------------
# 2. Read previous seed-level stability analysis
# ----------------------------------------------------------

details = pd.read_csv(
    DETAILS,
    dtype={
        "requirement_id": str,
        "similarity_group": str,
    },
)

verify(
    len(details) == 728 * 3,
    "Unexpected Script 10 detail count",
)

verify(
    set(details["architecture"]) == set(MODELS),
    "Unexpected model list",
)

verify(
    not details.duplicated(
        ["architecture", "requirement_id"]
    ).any(),
    "Duplicate model-requirement records",
)

verify(
    details["true_label"].isin(["FR", "NFR"]).all(),
    "Unexpected labels",
)

for model in MODELS:
    subset = details[
        details["architecture"] == model
    ]

    verify(
        len(subset) == 728,
        f"Incomplete records for {model}",
    )

# ----------------------------------------------------------
# 3. Reshape model-level predictions and error status
# ----------------------------------------------------------

wide = validation.copy()

for model in MODELS:

    subset = details.loc[
        details["architecture"] == model,
        [
            "requirement_id",
            "true_label",
            "seed42_prediction",
            "seed17_prediction",
            "seed29_prediction",
        ],
    ].copy()

    subset = subset.rename(columns={
        "true_label": f"{model}_true",
        "seed42_prediction": f"{model}_seed42",
        "seed17_prediction": f"{model}_seed17",
        "seed29_prediction": f"{model}_seed29",
    })

    wide = wide.merge(
        subset,
        on="requirement_id",
        how="left",
        validate="one_to_one",
    )

    verify(
        wide[f"{model}_true"].notna().all(),
        f"Missing records: {model}",
    )

    verify(
        (
            wide[f"{model}_true"]
            == wide["true_label"]
        ).all(),
        f"Label mismatch: {model}",
    )

    seed_cols = [
        f"{model}_seed42",
        f"{model}_seed17",
        f"{model}_seed29",
    ]

    for column in seed_cols:
        verify(
            wide[column].isin(["FR", "NFR"]).all(),
            f"Invalid predictions in {column}",
        )

    wide[f"{model}_wrong_all_seeds"] = (
        wide[seed_cols]
        .ne(wide["true_label"], axis=0)
        .all(axis=1)
    )

    wide[f"{model}_unstable"] = (
        wide[seed_cols].nunique(axis=1) > 1
    )

# ----------------------------------------------------------
# 4. Identify persistent errors
# ----------------------------------------------------------

wrong_cols = [
    f"{model}_wrong_all_seeds"
    for model in MODELS
]

unstable_cols = [
    f"{model}_unstable"
    for model in MODELS
]

wide["architectures_always_wrong"] = (
    wide[wrong_cols].sum(axis=1)
)

wide["architectures_with_seed_variation"] = (
    wide[unstable_cols].sum(axis=1)
)

wide["wrong_all_nine_runs"] = (
    wide["architectures_always_wrong"] == 3
)

wide["wrong_at_least_two_architectures"] = (
    wide["architectures_always_wrong"] >= 2
)

wide["error_category"] = "Other"

wide.loc[
    wide["architectures_with_seed_variation"] > 0,
    "error_category",
] = "Seed-dependent prediction"

wide.loc[
    wide["architectures_always_wrong"] == 1,
    "error_category",
] = "Persistent error in one architecture"

wide.loc[
    wide["architectures_always_wrong"] == 2,
    "error_category",
] = "Persistent error in two architectures"

wide.loc[
    wide["wrong_all_nine_runs"],
    "error_category",
] = "Persistent error in all nine runs"

# Categories prioritize persistent errors;
# the seed-variation count is retained separately.

# ----------------------------------------------------------
# 5. Summarize errors by class and category
# ----------------------------------------------------------

summary = (
    wide.groupby(
        ["true_label", "error_category"],
        dropna=False,
    )
    .size()
    .reset_index(name="requirements")
)

print("\nPERSISTENT ERROR SUMMARY")
print(
    summary.to_string(index=False)
)

print("\nCONSENSUS ERROR COUNTS")
print(
    "Wrong across all nine runs:",
    int(wide["wrong_all_nine_runs"].sum()),
)

print(
    "Wrong across >=2 architectures, all their seeds:",
    int(
        wide["wrong_at_least_two_architectures"].sum()
    ),
)

print(
    "At least one architecture varies across seeds:",
    int(
        (
            wide[
                "architectures_with_seed_variation"
            ] > 0
        ).sum()
    ),
)

print("\nPERSISTENT ERRORS BY TRUE CLASS")

persistent_by_class = (
    wide.loc[wide["wrong_all_nine_runs"]]
    .groupby("true_label")
    .size()
    .reindex(["FR", "NFR"], fill_value=0)
)

print(persistent_by_class.to_string())

# ----------------------------------------------------------
# 6. Prepare manual review worksheet
# ----------------------------------------------------------

review = wide.loc[
    wide["wrong_at_least_two_architectures"]
    | wide["architectures_with_seed_variation"].gt(0)
].copy()

review = review.sort_values(
    [
        "architectures_always_wrong",
        "architectures_with_seed_variation",
        "requirement_id",
    ],
    ascending=[False, False, True],
)

# Empty fields are intentionally left for manual review.
review["reviewer_assessment"] = ""
review["possible_issue"] = ""
review["reviewer_notes"] = ""

review["review_status"] = "NOT_REVIEWED"

review_columns = [
    "requirement_id",
    "similarity_group",
    "true_label",
    TEXT_COL,
    "error_category",
    "architectures_always_wrong",
    "architectures_with_seed_variation",
    "distilbert_seed42",
    "distilbert_seed17",
    "distilbert_seed29",
    "bert_seed42",
    "bert_seed17",
    "bert_seed29",
    "roberta_seed42",
    "roberta_seed17",
    "roberta_seed29",
    "review_status",
    "reviewer_assessment",
    "possible_issue",
    "reviewer_notes",
]

review = review[review_columns]

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

review.to_csv(
    REVIEW_PATH,
    index=False,
    encoding="utf-8-sig",
)

# ----------------------------------------------------------
# 7. Save research report
# ----------------------------------------------------------

report_lines = [
    "SCRIPT 11: PERSISTENT ERROR ANALYSIS",
    "=" * 76,
    "",
    "Dataset: D2-v1.1",
    "Validation requirements: 728",
    "Architectures: DistilBERT, BERT-base, RoBERTa-base",
    "Seeds: 42, 17, 29",
    "Training performed: NO",
    "Test text accessed: NO",
    "",
    "ERROR CATEGORY SUMMARY",
    summary.to_string(index=False),
    "",
    "CONSENSUS ERRORS",
    "Wrong in all nine runs: "
    + str(int(wide["wrong_all_nine_runs"].sum())),
    "Wrong consistently in >=2 architectures: "
    + str(
        int(
            wide[
                "wrong_at_least_two_architectures"
            ].sum()
        )
    ),
    "",
    "PERSISTENT ERRORS BY CLASS",
    persistent_by_class.to_string(),
    "",
    "MANUAL REVIEW",
    f"Requirements selected for review: {len(review)}",
    "Annotations are not automatically corrected.",
    "Reviewer fields are intentionally blank.",
    "",
    "LIMITATIONS",
    "- Errors are defined relative to existing labels.",
    "- Persistent disagreement does not prove mislabeling.",
    "- Review selection uses validation predictions.",
    "- The same validation partition was reused.",
    "- Only three seeds per architecture.",
    "- No independent label adjudication conducted.",
    "- No test-partition analysis.",
]

REPORT_PATH.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

META_PATH.write_text(
    json.dumps({
        "experiment": "DL-11",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": "D2-v1.1",
        "manifest_sha256": manifest_hash,
        "validation_count": 728,
        "architectures": MODELS,
        "seeds": [42, 17, 29],
        "review_items": len(review),
        "wrong_all_nine": int(
            wide["wrong_all_nine_runs"].sum()
        ),
        "label_changes": False,
        "test_accessed": False,
        "new_training": False,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nMANUAL REVIEW")
print("Selected requirements:", len(review))
print("Original labels modified: NO")

print("\nRESEARCH STATUS")
print("Persistent error analysis: COMPLETED")
print("Manual annotation review: PENDING")
print("Model training: NO")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    SUMMARY_PATH,
    REVIEW_PATH,
    REPORT_PATH,
    META_PATH,
]:
    print(path)

print("\nScript 11 completed.")
