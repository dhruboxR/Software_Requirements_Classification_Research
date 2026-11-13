
from pathlib import Path
from collections import defaultdict
import re
import unicodedata
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MAIN = DATA_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

workbooks = list(
    DATA_ROOT.rglob("Classified_AI_Requirements.xlsx")
)

if len(workbooks) != 1:
    raise RuntimeError(
        f"Expected one source workbook, found {len(workbooks)}"
    )


def normalize(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean(value):
    return str(value).strip() if value is not None else ""


df = pd.read_csv(
    MAIN, dtype=str, keep_default_na=False
)

d4 = df[df["Datasets"].str.strip() == "D4"].copy()
d4["key"] = d4[
    "Requirement Text (Original Requirement)"
].map(normalize)
d4["label"] = d4["Class"].str.strip().str.upper()

reference = defaultdict(list)

wb = load_workbook(
    workbooks[0], read_only=True, data_only=True
)

pattern = re.compile(
    r"^(claude|google|llama|mistral|openai)_Round_[1-5]$",
    re.IGNORECASE
)

for ws in wb:
    if not pattern.fullmatch(ws.title):
        continue

    for row_number, row in enumerate(
        ws.iter_rows(min_row=2, values_only=True),
        start=2
    ):
        values = list(row) + [None] * 6
        llm, rnd, project, text, col5, col6 = values[:6]

        key = normalize(text)
        if not key:
            continue

        reference[key].append({
            "model": clean(llm).lower(),
            "round": clean(rnd),
            "project": clean(project),
            "sheet": ws.title,
            "row": row_number,
            "column_5": clean(col5),
            "column_6": clean(col6),
        })

wb.close()

records = []

for key, group in d4.groupby("key"):
    matches = reference.get(key, [])

    models = sorted({
        x["model"] for x in matches if x["model"]
    })
    rounds = sorted({
        x["round"] for x in matches if x["round"]
    })
    projects = sorted({
        x["project"] for x in matches if x["project"]
    })

    labels = sorted(set(group["label"]))

    records.append({
        "normalized_text": key,
        "unified_rows": len(group),
        "unified_labels": "|".join(labels),
        "original_source_rows": len(matches),
        "models": "|".join(models),
        "model_count": len(models),
        "rounds": "|".join(rounds),
        "round_count": len(rounds),
        "original_project_names": "|".join(projects),
        "original_project_name_count": len(projects),
        "cross_model": len(models) > 1,
        "cross_round": len(rounds) > 1,
        "conflicting_unified_labels": len(labels) > 1,
    })

result = pd.DataFrame(records)

result.to_csv(
    TABLES / "38_d4_generation_dependency_by_text.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = []

for label in ["FR", "NFR"]:
    subset = result[
        result["unified_labels"].map(
            lambda x: label in x.split("|")
        )
    ]

    summary.append({
        "label": label,
        "unique_texts_with_label": len(subset),
        "matched_source_texts": int(
            (subset["original_source_rows"] > 0).sum()
        ),
        "cross_model_texts": int(
            subset["cross_model"].sum()
        ),
        "cross_round_texts": int(
            subset["cross_round"].sum()
        ),
        "conflicting_label_texts": int(
            subset["conflicting_unified_labels"].sum()
        ),
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    TABLES / "38_d4_generation_dependency_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 75)
print("SCRIPT 38: D4 GENERATION DEPENDENCY AUDIT")
print("=" * 75)

print("\nOVERALL")
print("D4 rows:", len(d4))
print("Unique texts:", len(result))
print(
    "Texts matched to original source:",
    int((result["original_source_rows"] > 0).sum())
)
print(
    "Texts associated with multiple models:",
    int(result["cross_model"].sum())
)
print(
    "Texts associated with multiple rounds:",
    int(result["cross_round"].sum())
)
print(
    "Texts with conflicting unified labels:",
    int(result["conflicting_unified_labels"].sum())
)

print("\nCLASS-LEVEL DEPENDENCY")
print(summary_df.to_string(index=False))

print("\nMODEL COUNT DISTRIBUTION")
print(
    result["model_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

report = [
    "D4 GENERATION DEPENDENCY AUDIT",
    "=" * 65,
    "",
    f"Rows: {len(d4)}",
    f"Unique texts: {len(result)}",
    "",
    summary_df.to_string(index=False),
    "",
    "Limitations:",
    "- Matching is based on normalized exact text.",
    "- Source occurrences are not independent examples.",
    "- Shared models or rounds do not alone prove leakage.",
    "- Project names may not represent actual source projects.",
    "- This does not independently validate binary labels.",
    "- No original data, labels or splits were changed.",
]

path = REPORTS / "38_d4_generation_dependency.txt"
path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", path)
print("Script 38 completed.")
