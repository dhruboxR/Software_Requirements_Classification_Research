
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"
BASE = ROOT / "dataset_exploration/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"
VALIDATED = BASE / "validated"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MAIN = DATA_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

REVIEW = VALIDATED / "phase1d_69_pair_preliminary_review.csv"


def norm(value):
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", s).strip().casefold()


def clean_label(value):
    if pd.isna(value) or not str(value).strip():
        return "[BLANK]"
    return str(value).strip().upper()


df = pd.read_csv(MAIN, dtype=str, keep_default_na=False)
review = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)

d2 = df[df["Datasets"].str.strip() == "D2"].copy()
d2["key"] = d2["Requirement Text (Original Requirement)"].map(norm)
d2["label"] = d2["Class"].map(clean_label)

# Match the eligibility rules used in Script 39.
conflicting = set(
    d2.groupby("key")["label"]
    .nunique()
    .loc[lambda s: s > 1]
    .index
)

flagged = set()
for _, row in review[review["source"] == "D2"].iterrows():
    flagged.add(norm(row["text_a"]))
    flagged.add(norm(row["text_b"]))

eligible = d2[
    d2["label"].isin(["FR", "NFR"])
    & d2["key"].ne("")
    & ~d2["key"].isin(conflicting)
    & ~d2["key"].isin(flagged)
].drop_duplicates("key").copy()

sources = list(DATA_ROOT.rglob("FR_NFR_Dataset Original.xlsx"))
if len(sources) != 1:
    raise RuntimeError("Expected exactly one original FR_NFR workbook")

original = pd.read_excel(
    sources[0],
    sheet_name="datasetFR_NFR_full",
    engine="openpyxl",
    dtype=str
)

original["key"] = original["Requirement Text"].map(norm)
original["source_label"] = original["Type"].map(clean_label)

source_groups = (
    original[original["key"] != ""]
    .groupby("key")
    .agg(
        source_labels=(
            "source_label",
            lambda s: "|".join(sorted(set(s)))
        ),
        source_rows=("source_label", "size")
    )
    .reset_index()
)

audit = eligible.merge(
    source_groups,
    on="key",
    how="left",
    validate="one_to_one"
)

def assign_status(row):
    if pd.isna(row["source_labels"]):
        return "UNMATCHED_SOURCE_TEXT"

    labels = set(row["source_labels"].split("|"))

    if len(labels) != 1:
        return "AMBIGUOUS_SOURCE_LABEL"

    if "[BLANK]" in labels:
        return "BLANK_SOURCE_LABEL"

    if row["label"] in labels:
        return "SOURCE_LABEL_AGREES"

    return "SOURCE_LABEL_DISAGREES"


audit["source_label_status"] = audit.apply(
    assign_status, axis=1
)

audit["approval_status"] = "PENDING_MANUAL_REVIEW"

details_path = TABLES / "41_d2_source_label_audit.csv"
audit.to_csv(details_path, index=False, encoding="utf-8-sig")

summary = (
    audit.groupby(["label", "source_label_status"])
    .size()
    .reset_index(name="unique_texts")
    .sort_values(["label", "source_label_status"])
)

summary_path = TABLES / "41_d2_label_reliability_summary.csv"
summary.to_csv(summary_path, index=False)

print("=" * 78)
print("SCRIPT 41: D2 FINAL LABEL RELIABILITY")
print("=" * 78)

print("\nELIGIBILITY")
print("Original D2 rows:", len(d2))
print("Eligible unique texts:", len(eligible))

print("\nSOURCE LABEL RELIABILITY")
print(summary.to_string(index=False))

print("\nOVERALL STATUS")
print(audit["source_label_status"].value_counts().to_string())

print("\nPOTENTIAL SOURCE-LABEL ISSUES")
issues = audit[
    audit["source_label_status"] != "SOURCE_LABEL_AGREES"
]

print("Unique texts requiring further review:", len(issues))
print(
    "FR requiring review:",
    int((issues["label"] == "FR").sum())
)
print(
    "NFR requiring review:",
    int((issues["label"] == "NFR").sum())
)

issues.to_csv(
    TABLES / "41_d2_unresolved_label_cases.csv",
    index=False,
    encoding="utf-8-sig"
)

report = [
    "D2 SOURCE-LABEL RELIABILITY AUDIT",
    "=" * 65,
    "",
    f"Eligible unique texts: {len(eligible)}",
    "",
    summary.to_string(index=False),
    "",
    "Limitations:",
    "- Matching source labels do not prove correctness.",
    "- Original annotations may contain errors.",
    "- Ambiguous and unmatched records require review.",
    "- No labels have been changed.",
    "- No records have been permanently excluded.",
    "- Dataset approval remains pending.",
]

report_path = REPORTS / "41_d2_label_reliability.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", details_path)
print("Saved:", summary_path)
print("Saved:", report_path)
print("Script 41 completed.")
