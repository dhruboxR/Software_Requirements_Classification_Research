
from pathlib import Path
import sys
import re
import unicodedata
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_main, TABLES, REPORTS

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"

zenodo_dirs = list(DATASET_ROOT.rglob(
    "Zenodo - Selected Software Requirements Specification SRS"
))

if len(zenodo_dirs) != 1:
    raise RuntimeError("Expected exactly one Zenodo directory")

ZENODO = zenodo_dirs[0]


def normalize(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().strip("'\"‘’“”")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


df = load_main("preprocessing")
d4 = df[df["Datasets"].str.strip() == "D4"].copy()

TEXT_COL = "Requirement Text (Original Requirement)"
d4["normalized_text"] = d4[TEXT_COL].map(normalize)

# Avoid counting blank strings as matches.
d4_texts = set(d4["normalized_text"]) - {""}

results = []
matched_by_file = {}

print("=" * 75)
print("D4 ZENODO PROVENANCE ANALYSIS")
print("=" * 75)
print("D4 records:", len(d4))
print("Unique D4 texts:", len(d4_texts))

excel_files = sorted(
    p for p in ZENODO.rglob("*")
    if p.is_file() and p.suffix.lower() in {".xlsx", ".xls"}
)

for path in excel_files:
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        print("Cannot open:", path.name, repr(exc))
        continue

    for sheet in workbook.sheet_names:
        try:
            source = pd.read_excel(
                path,
                sheet_name=sheet,
                dtype="string"
            )

            columns = {
                str(c).strip().casefold(): c
                for c in source.columns
            }

            # Only use explicitly identifiable text columns.
            candidates = [
                "requirement text",
                "text",
                "requirement"
            ]

            text_col = next(
                (columns[c] for c in candidates if c in columns),
                None
            )

            if text_col is None:
                continue

            source_texts = {
                normalize(value)
                for value in source[text_col]
            } - {""}

            matches = d4_texts & source_texts

            matched_by_file[(str(path.relative_to(ZENODO)), sheet)] = matches

            results.append({
                "file": str(path.relative_to(ZENODO)),
                "sheet": sheet,
                "source_rows": len(source),
                "source_unique_texts": len(source_texts),
                "matched_unique_d4_texts": len(matches)
            })

        except Exception as exc:
            print("Error:", path.name, sheet, repr(exc))

summary = pd.DataFrame(results)

if not summary.empty:
    summary = summary.sort_values(
        "matched_unique_d4_texts",
        ascending=False
    )

print("\nSOURCE MATCH SUMMARY")
print(summary.head(25).to_string(index=False))

all_matches = set()
for matches in matched_by_file.values():
    all_matches.update(matches)

d4["matched_zenodo_source"] = d4["normalized_text"].isin(all_matches)

matched_rows = int(d4["matched_zenodo_source"].sum())

print("\nOVERALL RESULTS")
print("D4 rows:", len(d4))
print("Matched D4 rows:", matched_rows)
print("Unmatched D4 rows:", len(d4) - matched_rows)
print("Distinct matched texts:", len(all_matches))

summary.to_csv(
    TABLES / "22_d4_zenodo_source_matches.csv",
    index=False,
    encoding="utf-8-sig"
)

d4.to_csv(
    TABLES / "22_d4_record_matching_status.csv",
    index=False,
    encoding="utf-8-sig"
)

report = [
    "D4 ZENODO PROVENANCE ANALYSIS",
    f"D4 rows: {len(d4)}",
    f"Matched rows: {matched_rows}",
    f"Unmatched rows: {len(d4) - matched_rows}",
    f"Distinct matched texts: {len(all_matches)}",
    "",
    "Exact normalized-text matching only.",
    "Matches do not establish that Zenodo was the sole origin.",
    "Unmatched records may contain transformed text.",
    "No original dataset was modified."
]

(REPORTS / "22_d4_zenodo_provenance_summary.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nReports saved successfully.")
