
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# SCRIPT 55: D2 v1.1 CROSS-PARTITION SIMILARITY AUDIT
#
# - Verify frozen v1.1 manifest integrity
# - Compare ALL cross-partition requirement pairs
# - Use character 3-5 gram TF-IDF cosine similarity
# - Count pairs above several thresholds
# - Save every pair with similarity >= 0.85
# - Preserve original v1.0 and frozen v1.1 files
#
# No model training.
# No dataset modification.
# No split regeneration.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

VALIDATED = BASE / "validated"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

MANIFEST = (
    VALIDATED / "54_d2_v11_frozen_split_manifest.csv"
)

METADATA = (
    VALIDATED / "54_d2_v11_split_metadata.json"
)

SUMMARY_OUTPUT = (
    TABLES / "55_d2_v11_similarity_summary.csv"
)

PAIRS_OUTPUT = (
    TABLES / "55_d2_v11_cross_partition_pairs.csv"
)

REPORT_OUTPUT = (
    REPORTS / "55_d2_v11_similarity_audit.txt"
)

METADATA_OUTPUT = (
    REPORTS / "55_d2_v11_similarity_audit.json"
)

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
REVIEW_THRESHOLD = 0.85
BLOCK_SIZE = 64

for folder in (TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_csv(path):
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


print("=" * 78)
print("SCRIPT 55: D2 v1.1 CROSS-PARTITION SIMILARITY AUDIT")
print("=" * 78)

# ------------------------------------------------------------
# 1. Load and verify frozen inputs
# ------------------------------------------------------------

df = load_csv(MANIFEST)

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError(
        "Unexpected protocol version"
    )

expected_manifest_hash = (
    metadata.get("output_sha256", {})
    .get(MANIFEST.name)
)

if sha256(MANIFEST) != expected_manifest_hash:
    raise RuntimeError(
        "Frozen manifest SHA256 verification failed"
    )

required = {
    "Requirement ID",
    "key",
    "label",
    "split",
    "similarity_group"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

if len(df) != 4833:
    raise ValueError(
        f"Unexpected dataset size: {len(df)}"
    )

if not df["Requirement ID"].is_unique:
    raise ValueError("Duplicate requirement IDs")

if not df["key"].is_unique:
    raise ValueError("Duplicate normalized texts")

if set(df["label"]) != {"FR", "NFR"}:
    raise ValueError("Unexpected class labels")

if set(df["split"]) != {
    "train",
    "validation",
    "test"
}:
    raise ValueError("Unexpected partition labels")

if (df["key"].str.strip() == "").any():
    raise ValueError("Blank normalized text detected")

group_partition_counts = (
    df.groupby("similarity_group")["split"]
    .nunique()
)

if (group_partition_counts > 1).any():
    raise RuntimeError(
        "Existing similarity groups cross partitions"
    )

df = df.sort_values("key").reset_index(drop=True)

print("\nDATASET INTEGRITY")
print("Protocol:", metadata["protocol"])
print("Requirements:", len(df))
print("Unique texts:", df["key"].nunique())
print("Similarity groups:", df["similarity_group"].nunique())
print("Frozen manifest SHA256: PASS")
print("Existing group isolation: PASS")

# ------------------------------------------------------------
# 2. Compute TF-IDF representation
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=False,
    norm="l2",
    dtype=np.float32
)

# This audit fits an unsupervised similarity representation
# across the fixed candidate texts. It does not use labels
# for training or model selection.
X = vectorizer.fit_transform(df["key"])

print("\nSIMILARITY CONFIGURATION")
print("TF-IDF features:", X.shape[1])
print("Character n-grams: 3 to 5")
print("Review threshold:", REVIEW_THRESHOLD)
print("Neighbor limit: NONE")
print("Frozen split regenerated: NO")

partition_indices = {
    name: np.flatnonzero(
        df["split"].to_numpy() == name
    )
    for name in ("train", "validation", "test")
}

comparisons = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]

# ------------------------------------------------------------
# 3. Exhaustive cross-partition comparison
# ------------------------------------------------------------

summary_rows = []
review_rows = []

for left_name, right_name in comparisons:

    left_indices = partition_indices[left_name]
    right_indices = partition_indices[right_name]

    comparison = (
        f"{left_name}_vs_{right_name}"
    )

    threshold_counts = {
        threshold: 0
        for threshold in THRESHOLDS
    }

    maximum_similarity = 0.0
    right_vectors = X[right_indices]

    total_pairs = (
        len(left_indices) * len(right_indices)
    )

    for start in range(
        0, len(left_indices), BLOCK_SIZE
    ):
        batch_indices = left_indices[
            start:start + BLOCK_SIZE
        ]

        similarities = (
            X[batch_indices] @ right_vectors.T
        ).toarray()

        if similarities.size:
            maximum_similarity = max(
                maximum_similarity,
                float(similarities.max())
            )

        for threshold in THRESHOLDS:
            threshold_counts[threshold] += int(
                np.count_nonzero(
                    similarities >= threshold
                )
            )

        rows, cols = np.where(
            similarities >= REVIEW_THRESHOLD
        )

        for r, c in zip(rows, cols):

            a = int(batch_indices[r])
            b = int(right_indices[c])

            review_rows.append({
                "comparison": comparison,
                "requirement_id_a": df.loc[
                    a, "Requirement ID"
                ],
                "requirement_id_b": df.loc[
                    b, "Requirement ID"
                ],
                "split_a": df.loc[a, "split"],
                "split_b": df.loc[b, "split"],
                "label_a": df.loc[a, "label"],
                "label_b": df.loc[b, "label"],
                "similarity": round(
                    float(similarities[r, c]), 6
                ),
                "group_a": df.loc[
                    a, "similarity_group"
                ],
                "group_b": df.loc[
                    b, "similarity_group"
                ],
                "text_a": df.loc[a, "key"],
                "text_b": df.loc[b, "key"],
                "review_status": "PENDING"
            })

    row = {
        "comparison": comparison,
        "pairs_compared": total_pairs,
        "maximum_similarity": round(
            maximum_similarity, 6
        )
    }

    for threshold in THRESHOLDS:
        column = f"pairs_ge_{threshold:.2f}"
        row[column] = threshold_counts[threshold]

    summary_rows.append(row)

    print(f"\n{comparison.upper()}")
    print("Pairs compared:", total_pairs)
    print(
        "Maximum similarity:",
        round(maximum_similarity, 6)
    )

    for threshold in THRESHOLDS:
        print(
            f"Pairs >= {threshold:.2f}:",
            threshold_counts[threshold]
        )

# ------------------------------------------------------------
# 4. Assemble audit results
# ------------------------------------------------------------

summary = pd.DataFrame(summary_rows)

pair_columns = [
    "comparison",
    "requirement_id_a",
    "requirement_id_b",
    "split_a",
    "split_b",
    "label_a",
    "label_b",
    "similarity",
    "group_a",
    "group_b",
    "text_a",
    "text_b",
    "review_status"
]

review = pd.DataFrame(
    review_rows,
    columns=pair_columns
)

if not review.empty:
    review = review.sort_values(
        "similarity",
        ascending=False
    ).reset_index(drop=True)

total_high = int(
    summary["pairs_ge_0.85"].sum()
)

total_80 = int(
    summary["pairs_ge_0.80"].sum()
)

test_high = int(
    summary.loc[
        summary["comparison"].isin([
            "train_vs_test",
            "validation_vs_test"
        ]),
        "pairs_ge_0.85"
    ].sum()
)

if len(review) != total_high:
    raise RuntimeError(
        "Review-pair count does not match summary"
    )

# ------------------------------------------------------------
# 5. Save new audit artifacts
# ------------------------------------------------------------

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

review.to_csv(
    PAIRS_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

audit_metadata = {
    "protocol": "D2-v1.1",
    "audit": "exhaustive_cross_partition_similarity",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "frozen_manifest_sha256": sha256(MANIFEST),
    "requirement_count": len(df),
    "similarity_features": X.shape[1],
    "method": "char_3_5_tfidf_cosine",
    "review_threshold": REVIEW_THRESHOLD,
    "thresholds": THRESHOLDS,
    "top_k_limit": None,
    "total_cross_partition_pairs": int(
        summary["pairs_compared"].sum()
    ),
    "pairs_ge_080": total_80,
    "pairs_ge_085": total_high,
    "test_involving_pairs_ge_085": test_high,
    "status": (
        "NO_PAIRS_AT_THRESHOLD"
        if total_high == 0
        else "REVIEW_REQUIRED"
    ),
    "output_sha256": {
        SUMMARY_OUTPUT.name: sha256(SUMMARY_OUTPUT),
        PAIRS_OUTPUT.name: sha256(PAIRS_OUTPUT)
    },
    "limitations": [
        "Character similarity is not semantic equivalence.",
        "Lower similarity does not guarantee independence.",
        "Original project independence remains unverified.",
        "No classifier was trained or evaluated.",
        "No frozen partition was modified."
    ]
}

METADATA_OUTPUT.write_text(
    json.dumps(audit_metadata, indent=2),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 6. Print audit conclusion
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("OVERALL AUDIT SUMMARY")
print("=" * 78)

print("Total requirements:", len(df))
print(
    "Total cross-partition pairs compared:",
    int(summary["pairs_compared"].sum())
)
print("Pairs >= 0.80:", total_80)
print("Pairs >= 0.85:", total_high)
print(
    "Test-involving pairs >= 0.85:",
    test_high
)

print("\nSUMMARY TABLE")
print(summary.to_string(index=False))

print("\nAUDIT CONCLUSION")

if total_high == 0:
    print("Similarity threshold screen: PASS")
    print(
        "No cross-partition pairs reached 0.85 "
        "under the current TF-IDF representation."
    )
else:
    print(
        "Similarity threshold screen: REVIEW_REQUIRED"
    )
    print(
        "Review the flagged pairs before considering "
        "the protocol ready for model training."
    )

print("Scientific approval: PENDING")
print("Frozen v1.1 manifest changed: NO")
print("Model training: NOT STARTED")

report_lines = [
    "D2 v1.1 CROSS-PARTITION SIMILARITY AUDIT",
    "=" * 65,
    "",
    f"Requirements: {len(df)}",
    f"TF-IDF features: {X.shape[1]}",
    f"Pairs >= 0.80: {total_80}",
    f"Pairs >= 0.85: {total_high}",
    f"Test-involving pairs >= 0.85: {test_high}",
    "",
    summary.to_string(index=False),
    "",
    "Audit status: "
    + (
        "PASS"
        if total_high == 0
        else "REVIEW_REQUIRED"
    ),
    "",
    "Limitations:",
    "- Similarity is lexical, not semantic.",
    "- Undetected paraphrases may remain.",
    "- Original-project independence is unverified.",
    "- AI-assisted annotation decisions are provisional.",
    "- No model training occurred.",
    "- No frozen split was modified."
]

REPORT_OUTPUT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(SUMMARY_OUTPUT)
print(PAIRS_OUTPUT)
print(REPORT_OUTPUT)
print(METADATA_OUTPUT)

print("\nScript 55 completed.")
