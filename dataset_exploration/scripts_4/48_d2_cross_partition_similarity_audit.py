
from pathlib import Path
from collections import defaultdict
import heapq
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# SCRIPT 48: CROSS-PARTITION SIMILARITY AUDIT
# Read-only. No model training or split modification.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

VALIDATED = BASE / "validated"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

MANIFEST = VALIDATED / "44_d2_frozen_split_manifest.csv"
METADATA = VALIDATED / "44_d2_split_metadata.json"

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
REVIEW_THRESHOLD = 0.85
MAX_REVIEW_PAIRS = 200
BLOCK_SIZE = 64

for folder in [TABLES, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

required = {
    "Requirement ID",
    "key",
    "split",
    "similarity_group"
}

if not required.issubset(df.columns):
    raise ValueError("Missing required manifest columns")

if df["key"].duplicated().any():
    raise ValueError("Duplicate text keys")

if set(df["split"]) != {
    "train", "validation", "test"
}:
    raise ValueError("Unexpected partitions")

if len(df) != 4835:
    raise ValueError("Unexpected dataset size")

# Keep deterministic ordering.
df = df.sort_values("key").reset_index(drop=True)

# TF-IDF vectors are normalized by default.
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=False,
    norm="l2",
    dtype=np.float32
)

X = vectorizer.fit_transform(df["key"])

partitions = {
    name: np.flatnonzero(
        df["split"].to_numpy() == name
    )
    for name in ["train", "validation", "test"]
}

comparisons = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]

counts = defaultdict(
    lambda: {t: 0 for t in THRESHOLDS}
)

max_scores = defaultdict(float)
top_pairs = []
serial = 0

print("=" * 78)
print("SCRIPT 48: D2 CROSS-PARTITION SIMILARITY AUDIT")
print("=" * 78)

print("\nDATASET")
print("Requirements:", len(df))
print("TF-IDF features:", X.shape[1])
print("Frozen split seed:", metadata["seed"])

for left_name, right_name in comparisons:
    left_ids = partitions[left_name]
    right_ids = partitions[right_name]

    pair_name = f"{left_name}_vs_{right_name}"

    print(f"\nChecking {pair_name} ...")

    right_matrix = X[right_ids]
    max_similarity = 0.0

    for start in range(0, len(left_ids), BLOCK_SIZE):
        batch_ids = left_ids[
            start:start + BLOCK_SIZE
        ]

        similarities = (
            X[batch_ids] @ right_matrix.T
        ).toarray()

        max_similarity = max(
            max_similarity,
            float(similarities.max(initial=0))
        )

        for threshold in THRESHOLDS:
            counts[pair_name][threshold] += int(
                np.count_nonzero(
                    similarities >= threshold
                )
            )

        rows, cols = np.where(
            similarities >= REVIEW_THRESHOLD
        )

        for r, c in zip(rows, cols):
            similarity = float(similarities[r, c])

            a = int(batch_ids[r])
            b = int(right_ids[c])

            serial += 1

            item = (
                similarity,
                serial,
                a,
                b,
                pair_name
            )

            if len(top_pairs) < MAX_REVIEW_PAIRS:
                heapq.heappush(top_pairs, item)

            elif similarity > top_pairs[0][0]:
                heapq.heapreplace(top_pairs, item)

    max_scores[pair_name] = max_similarity

    print(
        "Maximum cross-partition similarity:",
        round(max_similarity, 6)
    )

    for threshold in THRESHOLDS:
        print(
            f"Pairs >= {threshold:.2f}:",
            counts[pair_name][threshold]
        )

# ------------------------------------------------------------
# Save detailed review candidates (IDs only).
# ------------------------------------------------------------

review_rows = []

for similarity, _, a, b, pair_name in sorted(
    top_pairs,
    reverse=True
):
    review_rows.append({
        "comparison": pair_name,
        "requirement_id_a": df.loc[a, "Requirement ID"],
        "requirement_id_b": df.loc[b, "Requirement ID"],
        "split_a": df.loc[a, "split"],
        "split_b": df.loc[b, "split"],
        "similarity": round(similarity, 6),
        "similarity_group_a": df.loc[
            a, "similarity_group"
        ],
        "similarity_group_b": df.loc[
            b, "similarity_group"
        ]
    })

review_columns = [
    "comparison",
    "requirement_id_a",
    "requirement_id_b",
    "split_a",
    "split_b",
    "similarity",
    "similarity_group_a",
    "similarity_group_b"
]

review_df = pd.DataFrame(
    review_rows,
    columns=review_columns
)

review_path = TABLES / (
    "48_d2_cross_partition_review_candidates.csv"
)

review_df.to_csv(
    review_path,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Summarize all comparisons.
# ------------------------------------------------------------

summary_rows = []

for pair_name in counts:
    row = {
        "comparison": pair_name,
        "maximum_similarity": round(
            max_scores[pair_name], 6
        )
    }

    for threshold in THRESHOLDS:
        row[f"pairs_ge_{threshold:.2f}"] = (
            counts[pair_name][threshold]
        )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary_path = TABLES / (
    "48_d2_cross_partition_similarity_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

total_high = sum(
    counts[name][REVIEW_THRESHOLD]
    for name in counts
)

print("\n" + "=" * 78)
print("OVERALL SIMILARITY AUDIT")
print("=" * 78)

print("Total requirements:", len(df))
print(
    "Cross-partition pairs >= 0.85:",
    total_high
)
print(
    "Review candidates saved:",
    len(review_df)
)

print("\nSUMMARY")
print(summary.to_string(index=False))

print("\nAUDIT INTERPRETATION")

if total_high == 0:
    print(
        "No cross-partition pairs reached the "
        "0.85 similarity threshold under this audit."
    )
else:
    print(
        "Cross-partition pairs reached the 0.85 "
        "threshold. Structural review is required."
    )

print(
    "Semantic independence is not established "
    "by this test."
)

# ------------------------------------------------------------
# Save report.
# ------------------------------------------------------------

report = [
    "D2 CROSS-PARTITION SIMILARITY AUDIT",
    "=" * 65,
    "",
    f"Dataset size: {len(df)}",
    f"Features: {X.shape[1]}",
    f"Total pairs >= 0.85: {total_high}",
    "",
    summary.to_string(index=False),
    "",
    "LIMITATIONS",
    "- Similarity is measured with character TF-IDF.",
    "- The vocabulary is refitted on the frozen candidate set.",
    "- Scores may differ from the earlier similarity search.",
    "- High similarity does not prove semantic equivalence.",
    "- Low similarity does not prove semantic independence.",
    "- No classifier was trained.",
    "- No frozen partitions were changed."
]

report_path = REPORTS / (
    "48_d2_cross_partition_similarity_audit.txt"
)

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(summary_path)
print(review_path)
print(report_path)

print("\nScript 48 completed.")
