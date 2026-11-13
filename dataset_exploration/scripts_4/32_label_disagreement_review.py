
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

DATA_FILE = DATA_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

PAIRS_FILE = TABLES / "31_priority_near_duplicate_review.csv"

def normalize(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()

def join_values(values):
    return "|".join(sorted({
        str(v).strip()
        for v in values
        if pd.notna(v) and str(v).strip()
    }))

df = pd.read_csv(
    DATA_FILE, dtype=str, keep_default_na=False
)

pairs = pd.read_csv(
    PAIRS_FILE, keep_default_na=False
)

df["key"] = df[
    "Requirement Text (Original Requirement)"
].map(normalize)

df["source"] = df["Datasets"].str.strip()

lookup = {}

for (source, key), group in df.groupby(["source", "key"]):
    lookup[(source, key)] = {
        "ids": join_values(group["Requirement ID"]),
        "projects": join_values(group["Project ID"]),
        "project_names": join_values(group["Project Name"]),
        "labels": join_values(group["Class"]),
        "rows": len(group),
    }

records = []

for index, row in pairs.iterrows():
    source = row["source"]
    a = lookup.get((source, normalize(row["text_a"])), {})
    b = lookup.get((source, normalize(row["text_b"])), {})

    records.append({
        "review_id": f"{source}-{index + 1:04d}",
        "source": source,
        "similarity": row["similarity"],

        "text_a": row["text_a"],
        "label_a": row["labels_a"],
        "requirement_ids_a": a.get("ids", ""),
        "project_ids_a": a.get("projects", ""),
        "project_names_a": a.get("project_names", ""),

        "text_b": row["text_b"],
        "label_b": row["labels_b"],
        "requirement_ids_b": b.get("ids", ""),
        "project_ids_b": b.get("projects", ""),
        "project_names_b": b.get("project_names", ""),

        "both_texts_found": bool(a) and bool(b),
        "same_reported_project": (
            bool(a) and bool(b)
            and a.get("projects") == b.get("projects")
        ),

        "semantic_relationship": "PENDING",
        "annotation_assessment": "PENDING",
        "proposed_action": "PENDING",
        "reviewer_reason": "",
    })

review = pd.DataFrame(records)

review = review.sort_values(
    ["source", "similarity"],
    ascending=[True, False]
)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

output = TABLES / "32_label_disagreement_review.csv"
review.to_csv(output, index=False, encoding="utf-8-sig")

summary = review.groupby("source").agg(
    review_pairs=("review_id", "size"),
    both_texts_found=("both_texts_found", "sum"),
    same_reported_project=("same_reported_project", "sum"),
).reset_index()

summary_path = TABLES / "32_disagreement_summary.csv"
summary.to_csv(summary_path, index=False)

print("=" * 75)
print("SCRIPT 32: LABEL DISAGREEMENT REVIEW")
print("=" * 75)

print("\nSUMMARY")
print(summary.to_string(index=False))

print("\nTOTAL REVIEW PAIRS:", len(review))

print("\nTRACEABILITY CHECK")
print(
    "Pairs with missing source records:",
    int((~review["both_texts_found"]).sum())
)

report = [
    "PHASE 1D: LABEL DISAGREEMENT REVIEW",
    "=" * 65,
    "",
    summary.to_string(index=False),
    "",
    "Manual review fields:",
    "semantic_relationship: SAME / RELATED / DIFFERENT / UNCERTAIN",
    "annotation_assessment: CONSISTENT / CONFLICT / UNCERTAIN",
    "proposed_action: RETAIN / QUARANTINE / REVIEW_SOURCE",
    "",
    "Notes:",
    "- Reported project IDs may be unreliable.",
    "- Similarity does not prove semantic equivalence.",
    "- No label corrections have been applied.",
    "- No records have been deleted.",
]

report_path = REPORTS / "32_label_disagreement_review.txt"
report_path.write_text("\n".join(report) + "\n")

print("\nSaved:", output)
print("Saved:", summary_path)
print("Saved:", report_path)
print("All decisions remain pending.")
