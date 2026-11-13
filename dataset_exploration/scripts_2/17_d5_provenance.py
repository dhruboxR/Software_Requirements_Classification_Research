
from pathlib import Path
import pandas as pd

from common import (
    load_main,
    normalize_text,
    TABLES,
    REPORTS
)

TEXT_COL = "Requirement Text (Original Requirement)"

df = load_main("preprocessing").copy()

df["normalized_text"] = df[TEXT_COL].map(normalize_text)

d5 = df[df["Datasets"].str.strip() == "D5"].copy()
others = df[df["Datasets"].str.strip() != "D5"].copy()

reference = (
    others.groupby("normalized_text")
    .agg(
        matched_sources=(
            "Datasets",
            lambda x: "|".join(sorted(set(x)))
        ),
        matched_project_names=(
            "Project Name",
            lambda x: "|".join(sorted(set(x)))
        ),
        matched_labels=(
            "Class",
            lambda x: "|".join(sorted(set(x)))
        ),
        matching_rows=("Requirement ID", "size")
    )
    .reset_index()
)

comparison = d5.merge(
    reference,
    on="normalized_text",
    how="left",
    validate="many_to_one",
    indicator=True
)

comparison["match_status"] = comparison["_merge"].map({
    "both": "matched",
    "left_only": "unmatched",
    "right_only": "not_applicable"
}).astype("string")

print("=" * 75)
print("D5 PROVENANCE INVESTIGATION")
print("=" * 75)

print("D5 rows:", len(d5))
print("Unique D5 texts:", d5["normalized_text"].nunique())

print("\nMATCH STATUS")
print(comparison["match_status"].value_counts().to_string())

matched = comparison[
    comparison["match_status"] == "matched"
]

print("\nMATCHED SOURCE COMBINATIONS")
print(
    matched["matched_sources"]
    .value_counts()
    .to_string()
)

print("\nFIRST 15 MATCHED REQUIREMENTS")
print(
    matched[
        [
            "Requirement ID",
            "Project Name",
            "Class",
            "matched_sources",
            "matched_project_names",
            "matched_labels"
        ]
    ].head(15).to_string(
        index=False,
        max_colwidth=100
    )
)

comparison.to_csv(
    TABLES / "17_d5_provenance_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

report = [
    "D5 PROVENANCE INVESTIGATION",
    f"Total D5 rows: {len(d5)}",
    f"Matched D5 rows: {len(matched)}",
    f"Unmatched D5 rows: {len(d5) - len(matched)}",
    "",
    "Matching uses normalized requirement text.",
    "A matching requirement does not establish original ownership.",
    "Original dataset files remain unchanged."
]

(REPORTS / "17_d5_provenance_summary.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved D5 provenance reports.")
