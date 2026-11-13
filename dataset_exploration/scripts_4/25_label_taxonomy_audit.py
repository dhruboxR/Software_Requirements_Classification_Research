
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "A Unified Multi-Source Dataset of Software Require"

MAIN_FILE = (
    DATASET_DIR
    / "Software Requirement Dataset (SR vs NSR) - After the Preprocessing.csv"
)

OUTPUT_DIR = ROOT / "dataset_exploration" / "outputs"
TABLES = OUTPUT_DIR / "tables"
REPORTS = OUTPUT_DIR / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SOURCE_COL = "Datasets"
LABEL_COL = "Class"
TEXT_COL = "Requirement Text (Original Requirement)"


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def clean_label(value):
    if pd.isna(value):
        return "[BLANK]"

    text = str(value).strip().upper()
    return text if text else "[BLANK]"


print("=" * 80)
print("SCRIPT 25: SOURCE-SPECIFIC LABEL TAXONOMY AUDIT")
print("=" * 80)

df = pd.read_csv(MAIN_FILE, encoding="utf-8-sig", low_memory=False)

required = [SOURCE_COL, LABEL_COL, TEXT_COL]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["source"] = df[SOURCE_COL].astype("string").str.strip()
df["label"] = df[LABEL_COL].map(clean_label)
df["normalized_text"] = df[TEXT_COL].map(normalize_text)

print("\nTotal rows:", len(df))
print("Sources:", sorted(df["source"].dropna().unique().tolist()))

# --------------------------------------------------
# 1. Audit all source-specific label distributions
# --------------------------------------------------

distribution = (
    df.groupby(["source", "label"], dropna=False)
    .agg(
        rows=("label", "size"),
        unique_texts=("normalized_text", "nunique"),
    )
    .reset_index()
)

source_totals = df.groupby("source").size()

distribution["source_pct"] = distribution.apply(
    lambda row: round(
        100 * row["rows"] / source_totals[row["source"]], 3
    ),
    axis=1,
)

distribution = distribution.sort_values(
    ["source", "rows"], ascending=[True, False]
)

distribution.to_csv(
    TABLES / "25_source_label_distribution.csv",
    index=False,
)

print("\nSOURCE LABEL DISTRIBUTIONS")
print(distribution.to_string(index=False))

# --------------------------------------------------
# 2. Find exact-text label conflicts WITHIN a source
# --------------------------------------------------

groups = (
    df.groupby(["source", "normalized_text"], dropna=False)
    .agg(
        rows=("label", "size"),
        labels=("label", lambda s: "|".join(sorted(set(s)))),
        distinct_labels=("label", "nunique"),
    )
    .reset_index()
)

conflict_groups = groups[
    (groups["distinct_labels"] > 1)
    & (groups["normalized_text"] != "")
].copy()

conflict_groups.to_csv(
    TABLES / "25_within_source_conflict_groups.csv",
    index=False,
)

conflict_keys = conflict_groups[
    ["source", "normalized_text"]
]

conflict_rows = df.merge(
    conflict_keys,
    on=["source", "normalized_text"],
    how="inner",
    validate="many_to_many",
)

conflict_rows.to_csv(
    TABLES / "25_within_source_conflict_review.csv",
    index=False,
)

print("\nWITHIN-SOURCE CONFLICT SUMMARY")

for source in sorted(df["source"].dropna().unique()):
    subset = conflict_groups[conflict_groups["source"] == source]
    records = conflict_rows[conflict_rows["source"] == source]

    print(
        f"{source}: {len(subset)} conflicting text groups, "
        f"{len(records)} affected rows"
    )

# --------------------------------------------------
# 3. D2 binary-label readiness
# --------------------------------------------------

d2 = df[df["source"] == "D2"].copy()

d2_binary = d2[d2["label"].isin(["FR", "NFR"])].copy()
d2_other = d2[~d2["label"].isin(["FR", "NFR"])].copy()

d2_other.to_csv(
    TABLES / "25_d2_nonbinary_label_review.csv",
    index=False,
)

d2_binary_groups = (
    d2_binary.groupby("normalized_text")["label"]
    .agg(lambda s: "|".join(sorted(set(s))))
    .reset_index(name="labels")
)

d2_fr_nfr_conflicts = d2_binary_groups[
    d2_binary_groups["labels"] == "FR|NFR"
].copy()

d2_fr_nfr_conflicts.to_csv(
    TABLES / "25_d2_fr_nfr_conflicts.csv",
    index=False,
)

print("\nD2 BINARY READINESS")
print("D2 total:", len(d2))
print("FR/NFR rows:", len(d2_binary))
print("Other/blank rows:", len(d2_other))
print(
    "Distinct texts with both FR and NFR:",
    len(d2_fr_nfr_conflicts),
)

# --------------------------------------------------
# 4. D3 multiclass-label readiness
# --------------------------------------------------

d3 = df[df["source"] == "D3"].copy()

expected_d3 = {
    "NSR", "THE", "CON", "THO", "MAI",
    "IMM", "IND", "AVA", "AUD", "SUR", "INT",
}

unexpected_d3 = d3[~d3["label"].isin(expected_d3)]

unexpected_d3.to_csv(
    TABLES / "25_d3_unexpected_labels.csv",
    index=False,
)

d3_conflicts = conflict_groups[
    conflict_groups["source"] == "D3"
]

print("\nD3 MULTICLASS READINESS")
print("D3 total:", len(d3))
print("Observed classes:", d3["label"].nunique())
print("Unexpected labels:", len(unexpected_d3))
print("Within-source conflicting groups:", len(d3_conflicts))

# --------------------------------------------------
# 5. D4 binary-label readiness
# --------------------------------------------------

d4 = df[df["source"] == "D4"].copy()

d4_groups = (
    d4.groupby("normalized_text")
    .agg(
        rows=("label", "size"),
        labels=("label", lambda s: "|".join(sorted(set(s)))),
    )
    .reset_index()
)

d4_conflicting = d4_groups[
    d4_groups["labels"].str.contains(r"\|", regex=True)
].copy()

d4_conflicting.to_csv(
    TABLES / "25_d4_conflicting_texts.csv",
    index=False,
)

print("\nD4 BINARY READINESS")
print("D4 total:", len(d4))
print("Unique texts:", d4["normalized_text"].nunique())
print("Conflicting text groups:", len(d4_conflicting))
print(
    "Rows associated with conflicting texts:",
    int(d4_conflicting["rows"].sum()),
)

# --------------------------------------------------
# 6. Write consolidated report
# --------------------------------------------------

report_lines = [
    "PHASE 1D: LABEL TAXONOMY AUDIT",
    "=" * 65,
    "",
    f"Original records: {len(df)}",
    "",
    "Within-source exact-text conflict groups:",
]

for source in sorted(df["source"].dropna().unique()):
    n = len(conflict_groups[conflict_groups["source"] == source])
    report_lines.append(f"  {source}: {n}")

report_lines += [
    "",
    "D2:",
    f"  FR/NFR records: {len(d2_binary)}",
    f"  Other or blank records: {len(d2_other)}",
    f"  Exact-text FR/NFR conflicts: {len(d2_fr_nfr_conflicts)}",
    "",
    "D3:",
    f"  Unexpected labels: {len(unexpected_d3)}",
    f"  Within-source conflicts: {len(d3_conflicts)}",
    "",
    "D4:",
    f"  Unique normalized texts: {d4['normalized_text'].nunique()}",
    f"  Conflicting text groups: {len(d4_conflicting)}",
    f"  Conflicting-group rows: {int(d4_conflicting['rows'].sum())}",
    "",
    "Limitations:",
    "- Exact-text conflicts are not automatically label errors.",
    "- Different source taxonomies are not automatically equivalent.",
    "- This script does not compare against original-source annotations.",
    "- Near-duplicate and semantic conflicts are not tested here.",
    "- No original records were modified.",
    "- No dataset is approved for training by this audit alone.",
]

report_path = REPORTS / "25_label_taxonomy_audit.txt"

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("REPORT")
print("=" * 80)
print("\n".join(report_lines))
print("\nSaved:", report_path)
print("Script 25 completed.")
