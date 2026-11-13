
from pathlib import Path
from collections import defaultdict
import re
import unicodedata
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SOURCE = list(DATASET_ROOT.rglob("Classified_AI_Requirements.xlsx"))
if len(SOURCE) != 1:
    raise RuntimeError(f"Expected one source workbook, found {len(SOURCE)}")

REVIEW_PATH = TABLES / "27_conflict_review_packet.csv"
review = pd.read_csv(
    REVIEW_PATH, dtype=str, keep_default_na=False
)
review = review[review["source"] == "D4"].copy()

def normalize(value):
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()

def binary_label(value):
    v = str(value or "").strip().upper()
    if v in ("FR", "NFR"):
        return v
    if v in ("FUNCTIONAL", "FUNCTIONAL REQUIREMENT"):
        return "FR"
    if v in ("NON-FUNCTIONAL", "NON FUNCTIONAL",
             "NONFUNCTIONAL", "NON-FUNCTIONAL REQUIREMENT"):
        return "NFR"
    return ""

targets = {
    normalize(text)
    for text in review["requirement_text"]
}

matches = []
wb = load_workbook(SOURCE[0], read_only=True, data_only=True)

sheet_pattern = re.compile(
    r"^(claude|google|llama|mistral|openai)_Round_[1-5]$",
    re.IGNORECASE
)

for ws in wb:
    if not sheet_pattern.fullmatch(ws.title):
        continue

    for row_num, row in enumerate(
        ws.iter_rows(min_row=2, values_only=True),
        start=2
    ):
        values = list(row)
        values += [None] * max(0, 6 - len(values))

        llm, rnd, project, text, classification, cls = values[:6]
        key = normalize(text)

        if key not in targets:
            continue

        label_5 = binary_label(classification)
        label_6 = binary_label(cls)

        if label_5 and label_6 and label_5 != label_6:
            derived = "COLUMN_DISAGREEMENT"
        else:
            derived = label_5 or label_6 or "UNRESOLVED"

        matches.append({
            "normalized_text": key,
            "requirement_text": str(text),
            "sheet": ws.title,
            "excel_row": row_num,
            "llm": str(llm or ""),
            "round": str(rnd or ""),
            "project": str(project or ""),
            "column_5_raw": str(classification or ""),
            "column_6_raw": str(cls or ""),
            "source_binary_label": derived,
        })

wb.close()

details = pd.DataFrame(matches)

details_path = TABLES / "29_d4_conflict_source_records.csv"
details.to_csv(
    details_path, index=False, encoding="utf-8-sig"
)

grouped = defaultdict(list)
for item in matches:
    grouped[item["normalized_text"]].append(item)

results = []

for _, item in review.iterrows():
    key = normalize(item["requirement_text"])
    records = grouped[key]

    labels = sorted({
        r["source_binary_label"]
        for r in records
    })

    results.append({
        "requirement_text": item["requirement_text"],
        "unified_labels": item["combined_labels"],
        "unified_affected_rows": item["affected_rows"],
        "source_matched_rows": len(records),
        "source_labels": "|".join(labels) if labels else "UNMATCHED",
        "models": "|".join(sorted({
            r["llm"] for r in records
        })),
        "rounds": "|".join(sorted({
            r["round"] for r in records
        })),
        "projects": "|".join(sorted({
            r["project"] for r in records
        })),
        "review_status": "PENDING",
    })

summary = pd.DataFrame(results)
summary_path = TABLES / "29_d4_conflict_provenance_summary.csv"

summary.to_csv(
    summary_path, index=False, encoding="utf-8-sig"
)

print("=" * 75)
print("SCRIPT 29: D4 CONFLICT PROVENANCE")
print("=" * 75)

print("Conflicting groups:", len(summary))
print("Original source records matched:", len(details))
print(
    "Groups without source matches:",
    int((summary["source_matched_rows"] == 0).sum())
)

print("\nCOMPACT CONFLICT SUMMARY")

print(
    summary[
        [
            "unified_labels",
            "unified_affected_rows",
            "source_matched_rows",
            "source_labels",
            "models",
            "rounds",
        ]
    ].to_string(index=False)
)

report = [
    "D4 ORIGINAL-SOURCE CONFLICT VERIFICATION",
    "=" * 65,
    f"Conflicting groups: {len(summary)}",
    f"Matched original rows: {len(details)}",
    "",
    "Important:",
    "- FR/NFR labels are extracted from the two annotation columns.",
    "- Conflicting binary values in the same row are flagged.",
    "- A broad category such as Security is not automatically NFR.",
    "- Multiple model records do not imply independent requirements.",
    "- All manual decisions remain pending.",
    "- Original data was not changed.",
]

report_path = REPORTS / "29_d4_conflict_provenance.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", details_path)
print("Saved:", summary_path)
print("Saved:", report_path)
