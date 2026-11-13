
from pathlib import Path
import pandas as pd

from common import load_main, TABLES, REPORTS

SOURCE_PATH = (
    TABLES / "15_d4_source_generation_records.csv"
)

TEXT_COL = "Requirement Text (Original Requirement)"

source = pd.read_csv(
    SOURCE_PATH,
    dtype="string",
    keep_default_na=False
)

d4 = load_main("preprocessing")
d4 = d4[d4["Datasets"].str.strip() == "D4"].copy()

# Use the same normalization used in Script 15.
import re
import unicodedata

def normalize(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().strip("'\"‘’“”")
    return re.sub(r"\s+", " ", text).casefold()

d4["normalized_text"] = d4[TEXT_COL].map(normalize)
d4["combined_label"] = d4["Class"].str.strip()

source["source_classification"] = (
    source["source_classification"].str.strip()
)

# For each D4 requirement, retain all source labels.
reference = (
    source.groupby("normalized_text")
    .agg(
        source_labels=(
            "source_classification",
            lambda values: "|".join(
                sorted(set(v for v in values if v))
            )
        ),
        source_label_count=(
            "source_classification",
            lambda values: len(
                set(v for v in values if v)
            )
        )
    )
    .reset_index()
)

comparison = d4.merge(
    reference,
    on="normalized_text",
    how="left",
    validate="many_to_one"
)

# Separate requirements with one source classification
# from those with multiple source classifications.
single = comparison[
    comparison["source_label_count"] == 1
].copy()

multiple = comparison[
    comparison["source_label_count"] > 1
].copy()

mapping = (
    single.groupby(
        ["source_labels", "combined_label"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("=" * 75)
print("D4 LABEL TRANSFORMATION ANALYSIS")
print("=" * 75)

print("Total D4 rows:", len(d4))
print("Single-source-label rows:", len(single))
print("Multiple-source-label rows:", len(multiple))
print(
    "Missing source-label rows:",
    len(comparison) - len(single) - len(multiple)
)

print("\nSOURCE TO COMBINED LABEL MATRIX")
print(mapping.to_string(index=False))

print("\nMULTIPLE-LABEL EXAMPLES")
print(
    multiple[
        [
            "Requirement ID",
            "Project Name",
            "combined_label",
            "source_labels"
        ]
    ].head(20).to_string(index=False)
)

mapping.to_csv(
    TABLES / "16_d4_label_transformation_matrix.csv",
    index=False,
    encoding="utf-8-sig"
)

comparison.to_csv(
    TABLES / "16_d4_label_mapping_records.csv",
    index=False,
    encoding="utf-8-sig"
)

(REPORTS / "16_d4_label_mapping_summary.txt").write_text(
    "\n".join([
        "D4 LABEL TRANSFORMATION ANALYSIS",
        f"Total D4 rows: {len(d4)}",
        f"Single-source-label rows: {len(single)}",
        f"Multiple-source-label rows: {len(multiple)}",
        "",
        "The matrix describes observed label associations.",
        "It does not prove the publisher's transformation rules.",
        "Original records remain unchanged."
    ]) + "\n",
    encoding="utf-8"
)

print("\nSaved Script 16 outputs.")
