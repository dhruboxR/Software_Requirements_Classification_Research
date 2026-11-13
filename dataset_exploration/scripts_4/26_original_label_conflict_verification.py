
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    ROOT / "A Unified Multi-Source Dataset of Software Require"
)

TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def normalize(value):
    if pd.isna(value):
        return ""

    s = unicodedata.normalize("NFKC", str(value)).strip()

    if len(s) >= 2 and (
        (s[0] == "'" and s[-1] == "'")
        or (s[0] == '"' and s[-1] == '"')
    ):
        s = s[1:-1]

    return re.sub(r"\s+", " ", s).strip().casefold()


def label(value):
    if pd.isna(value) or not str(value).strip():
        return "[BLANK]"
    return str(value).strip().upper()


def find_one(filename):
    matches = list(DATASET_ROOT.rglob(filename))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {filename}; found {len(matches)}"
        )

    return matches[0]


def summarize_reference(df, text_col, label_col):
    temp = df[[text_col, label_col]].copy()

    temp["key"] = temp[text_col].map(normalize)
    temp["reference_label"] = temp[label_col].map(label)

    temp = temp[temp["key"] != ""]

    return (
        temp.groupby("key")
        .agg(
            reference_rows=("reference_label", "size"),
            reference_labels=(
                "reference_label",
                lambda s: "|".join(sorted(set(s)))
            ),
        )
        .reset_index()
    )


def load_d2_reference():
    path = find_one("FR_NFR_Dataset Original.xlsx")

    source = pd.read_excel(
        path,
        sheet_name="datasetFR_NFR_full",
        engine="openpyxl"
    )

    expected = {"Requirement Text", "Type"}

    if not expected.issubset(source.columns):
        raise ValueError(
            f"D2 reference columns: {list(source.columns)}"
        )

    print("D2 original workbook rows:", len(source))

    return summarize_reference(
        source, "Requirement Text", "Type"
    )


def load_d3_reference():
    path = find_one("DOSSPRE 1.0Original.csv")

    source = pd.read_csv(
        path,
        encoding="cp1252",
        dtype=str,
        keep_default_na=False
    )

    expected = {"Requirement", "Class"}

    if not expected.issubset(source.columns):
        raise ValueError(
            f"D3 reference columns: {list(source.columns)}"
        )

    print("D3 original CSV rows:", len(source))

    return summarize_reference(
        source, "Requirement", "Class"
    )


print("=" * 78)
print("SCRIPT 26: ORIGINAL LABEL CONFLICT VERIFICATION")
print("=" * 78)

path = TABLES / "25_within_source_conflict_review.csv"

combined = pd.read_csv(
    path,
    dtype=str,
    keep_default_na=False
)

required = {
    "source", "label", "normalized_text"
}

if not required.issubset(combined.columns):
    raise ValueError(
        f"Missing columns: {required - set(combined.columns)}"
    )

# One record per conflicting requirement text.
groups = (
    combined.groupby(["source", "normalized_text"])
    .agg(
        combined_rows=("label", "size"),
        combined_labels=(
            "label",
            lambda s: "|".join(sorted(set(s)))
        ),
    )
    .reset_index()
)

groups["reference_rows"] = pd.NA
groups["reference_labels"] = pd.NA
groups["reference_match"] = "NOT_CHECKED"
groups["review_decision"] = "PENDING"
groups["review_notes"] = ""

references = {
    "D2": load_d2_reference(),
    "D3": load_d3_reference()
}

for source_name, reference in references.items():
    mask = groups["source"] == source_name

    indices = groups.index[mask]

    lookup = reference.set_index("key")

    for idx in indices:
        key = groups.at[idx, "normalized_text"]

        if key not in lookup.index:
            groups.at[idx, "reference_match"] = "UNMATCHED"
            continue

        record = lookup.loc[key]

        groups.at[idx, "reference_rows"] = int(
            record["reference_rows"]
        )
        groups.at[idx, "reference_labels"] = (
            record["reference_labels"]
        )

        if "|" in record["reference_labels"]:
            status = "AMBIGUOUS_REFERENCE_LABELS"
        else:
            status = "SINGLE_REFERENCE_LABEL"

        groups.at[idx, "reference_match"] = status

# D4 requires separate analysis of generated-source
# model, generation round, and annotation metadata.
groups.loc[
    groups["source"] == "D4", "reference_match"
] = "D4_GENERATION_AUDIT_PENDING"

groups = groups.sort_values(
    ["source", "normalized_text"]
)

output_path = (
    TABLES / "26_original_conflict_reference_audit.csv"
)

groups.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# Retain the original text and source metadata for
# direct manual examination. No annotations are changed.
review_rows = combined.merge(
    groups[
        [
            "source", "normalized_text",
            "reference_labels", "reference_match",
            "review_decision", "review_notes"
        ]
    ],
    on=["source", "normalized_text"],
    how="left",
    validate="many_to_one"
)

review_path = (
    TABLES / "26_conflict_records_for_manual_review.csv"
)

review_rows.to_csv(
    review_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nREFERENCE VERIFICATION SUMMARY")
print("-" * 78)

summary = (
    groups.groupby(
        ["source", "reference_match"],
        dropna=False
    )
    .size()
    .reset_index(name="conflicting_text_groups")
)

print(summary.to_string(index=False))

print("\nD2 ORIGINAL LABEL COMPARISONS")

d2 = groups[groups["source"] == "D2"]

print(
    d2[
        [
            "combined_labels",
            "reference_labels",
            "reference_match"
        ]
    ].to_string(index=False)
)

print("\nD3 ORIGINAL LABEL COMPARISONS")

d3 = groups[groups["source"] == "D3"]

print(
    d3[
        [
            "combined_labels",
            "reference_labels",
            "reference_match"
        ]
    ].to_string(index=False)
)

lines = [
    "PHASE 1D: SOURCE LABEL VERIFICATION",
    "=" * 65,
    "",
    f"Conflicting text groups reviewed: {len(groups)}",
    "",
    summary.to_string(index=False),
    "",
    "Interpretation cautions:",
    "- A reference label is not automatically correct.",
    "- Multiple original labels must be reviewed.",
    "- D3 original and unified taxonomies differ.",
    "- D4 generation-source annotation audit is pending.",
    "- No records or annotations were modified.",
]

report_path = REPORTS / "26_source_label_verification.txt"

report_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("\nSaved:", output_path)
print("Saved:", review_path)
print("Saved:", report_path)
print("\nScript 26 completed.")
