
from pathlib import Path
import pandas as pd
import re
import unicodedata

from common import load_main, TABLES, REPORTS

ROOT = Path(__file__).resolve().parents[2]

ZENODO = next(
    ROOT.rglob(
        "Zenodo - Selected Software Requirements Specification SRS"
    )
)

WORKBOOK = (
    ZENODO
    / "Process_data"
    / "Classified_AI_Requirements.xlsx"
)

TEXT_COL = "Requirement Text (Original Requirement)"


def normalize(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().strip("'\"‘’“”")
    return re.sub(r"\s+", " ", text).casefold()


d4 = load_main("preprocessing")
d4 = d4[d4["Datasets"].str.strip() == "D4"].copy()

d4["normalized_text"] = d4[TEXT_COL].map(normalize)
d4["combined_label"] = d4["Class"].str.strip()

records = []

excel = pd.ExcelFile(WORKBOOK)

for sheet in excel.sheet_names:
    data = pd.read_excel(
        WORKBOOK,
        sheet_name=sheet,
        dtype="string"
    )

    if "text" not in data.columns:
        continue

    for _, row in data.iterrows():
        records.append({
            "sheet": sheet,
            "normalized_text": normalize(row["text"]),
            "llm": str(row.get("llm", "")),
            "round": str(row.get("round", "")),
            "source_classification": str(
                row.get("classification", "")
            ).strip()
        })

source = pd.DataFrame(records)

source = source[source["normalized_text"] != ""]

# Keep all source classifications for each text.
reference = (
    source.groupby("normalized_text")
    .agg(
        source_labels=(
            "source_classification",
            lambda x: "|".join(
                sorted(set(v for v in x if v))
            )
        ),
        source_llms=(
            "llm",
            lambda x: "|".join(
                sorted(set(v for v in x if v))
            )
        ),
        source_rounds=(
            "round",
            lambda x: "|".join(
                sorted(set(v for v in x if v))
            )
        ),
        source_occurrences=("sheet", "size")
    )
    .reset_index()
)

comparison = d4.merge(
    reference,
    on="normalized_text",
    how="left",
    validate="many_to_one",
    indicator=True
)

matched = comparison[
    comparison["_merge"] == "both"
]

print("=" * 75)
print("D4 GENERATION AND LABEL AUDIT")
print("=" * 75)

print("D4 rows:", len(d4))
print("Matched classified-source rows:", len(matched))
print("Unmatched rows:", len(d4) - len(matched))

print("\nCOMBINED LABEL DISTRIBUTION")
print(d4["combined_label"].value_counts().to_string())

print("\nSOURCE CLASSIFICATION VALUES")
print(
    source["source_classification"]
    .value_counts()
    .head(30)
    .to_string()
)

print("\nMULTIPLE SOURCE CLASSIFICATIONS")

multiple = comparison[
    comparison["source_labels"].fillna("").str.contains(
        "|", regex=False
    )
]

print("D4 rows with multiple source classifications:",
      len(multiple))

print("\nSOURCE LLM DISTRIBUTION")
print(source["llm"].value_counts().to_string())

print("\nSOURCE ROUND DISTRIBUTION")
print(source["round"].value_counts().to_string())

comparison.to_csv(
    TABLES / "15_d4_generation_label_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

source.to_csv(
    TABLES / "15_d4_source_generation_records.csv",
    index=False,
    encoding="utf-8-sig"
)

report = [
    "D4 GENERATION AND LABEL AUDIT",
    f"D4 rows: {len(d4)}",
    f"Matched classified-source rows: {len(matched)}",
    f"Unmatched rows: {len(d4) - len(matched)}",
    f"Rows with multiple source classifications: {len(multiple)}",
    "",
    "Source classification values are not automatically",
    "assumed equivalent to FR/NFR.",
    "Original data and labels remain unchanged."
]

(REPORTS / "15_d4_generation_label_summary.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved D4 generation and label audit outputs.")
