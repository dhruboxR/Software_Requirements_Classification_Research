
from pathlib import Path
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    ROOT / "A Unified Multi-Source Dataset of Software Require"
)

DATA_FILE = DATASET_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Requirement Text (Original Requirement)"
SOURCE_COL = "Datasets"
LABEL_COL = "Class"

SOURCES = ["D2", "D3", "D4"]
THRESHOLD = 0.85
NEIGHBORS = 10


def normalize(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def clean_label(value):
    if pd.isna(value) or not str(value).strip():
        return "[BLANK]"

    return str(value).strip().upper()


df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig",
    low_memory=False
)

df["source"] = df[SOURCE_COL].astype(str).str.strip()
df["label"] = df[LABEL_COL].map(clean_label)
df["normalized_text"] = df[TEXT_COL].map(normalize)

results = []
summaries = []

print("=" * 75)
print("SCRIPT 30: NEAR-DUPLICATE REQUIREMENT AUDIT")
print("=" * 75)

for source in SOURCES:
    subset = df[
        (df["source"] == source)
        & (df["normalized_text"] != "")
    ].copy()

    # One record per unique text, retaining every
    # distinct label associated with that text.
    grouped = (
        subset.groupby("normalized_text")
        .agg(
            labels=("label", lambda s: "|".join(sorted(set(s)))),
            text=(TEXT_COL, "first"),
            occurrences=("label", "size")
        )
        .reset_index()
    )

    print(f"\nSOURCE: {source}")
    print("Unique texts:", len(grouped))

    texts = grouped["normalized_text"].tolist()

    if len(texts) < 2:
        continue

    # Character n-grams identify similar wording
    # without requiring a neural model.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        dtype="float32",
    )

    matrix = vectorizer.fit_transform(texts)

    k = min(NEIGHBORS + 1, len(grouped))

    neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    )

    neighbors.fit(matrix)

    # Process in batches to avoid holding the
    # entire neighbor matrix in memory.
    seen = set()
    source_pairs = []

    batch_size = 128

    for start in range(0, len(grouped), batch_size):
        stop = min(start + batch_size, len(grouped))

        distances, indices = neighbors.kneighbors(
            matrix[start:stop],
            return_distance=True
        )

        for offset, (dist_row, idx_row) in enumerate(
            zip(distances, indices)
        ):
            i = start + offset

            for distance, j in zip(dist_row, idx_row):
                j = int(j)

                if i == j:
                    continue

                similarity = 1.0 - float(distance)

                if similarity < THRESHOLD:
                    continue

                pair = tuple(sorted((i, j)))

                if pair in seen:
                    continue

                seen.add(pair)

                left = grouped.iloc[pair[0]]
                right = grouped.iloc[pair[1]]

                left_labels = set(left["labels"].split("|"))
                right_labels = set(right["labels"].split("|"))

                different_labels = left_labels != right_labels
                disjoint_labels = left_labels.isdisjoint(
                    right_labels
                )

                source_pairs.append({
                    "source": source,
                    "similarity": round(similarity, 5),
                    "text_a": left["text"],
                    "labels_a": left["labels"],
                    "occurrences_a": left["occurrences"],
                    "text_b": right["text"],
                    "labels_b": right["labels"],
                    "occurrences_b": right["occurrences"],
                    "different_label_sets": different_labels,
                    "disjoint_labels": disjoint_labels,
                    "manual_review": "PENDING",
                })

    summaries.append({
        "source": source,
        "unique_texts": len(grouped),
        "candidate_pairs": len(source_pairs),
        "different_label_pairs": sum(
            row["different_label_sets"]
            for row in source_pairs
        ),
        "disjoint_label_pairs": sum(
            row["disjoint_labels"]
            for row in source_pairs
        ),
    })

    results.extend(source_pairs)

    print("Candidate pairs:", len(source_pairs))
    print(
        "Different-label pairs:",
        sum(r["different_label_sets"] for r in source_pairs)
    )
    print(
        "Disjoint-label pairs:",
        sum(r["disjoint_labels"] for r in source_pairs)
    )

candidates = pd.DataFrame(results)

if len(candidates):
    candidates = candidates.sort_values(
        ["disjoint_labels", "similarity"],
        ascending=[False, False]
    )

candidates_path = TABLES / "30_near_duplicate_candidates.csv"

candidates.to_csv(
    candidates_path,
    index=False,
    encoding="utf-8-sig"
)

summary = pd.DataFrame(summaries)
summary_path = TABLES / "30_near_duplicate_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 75)
print("COMPARATIVE SUMMARY")
print("=" * 75)

print(summary.to_string(index=False))

report_path = REPORTS / "30_near_duplicate_audit.txt"

report_path.write_text(
    "PHASE 1D: NEAR-DUPLICATE AUDIT\n"
    + "=" * 65 + "\n\n"
    + summary.to_string(index=False)
    + "\n\nLimitations:\n"
    + "- Similarity is based on character n-gram TF-IDF.\n"
    + "- Candidate pairs are not confirmed semantic duplicates.\n"
    + "- Only the nearest neighbors above the threshold are reported.\n"
    + "- Paraphrases with different wording may be missed.\n"
    + "- Analysis is within each source, not across sources.\n"
    + "- No records or labels have been modified.\n",
    encoding="utf-8"
)

print("\nSaved:", candidates_path)
print("Saved:", summary_path)
print("Saved:", report_path)
