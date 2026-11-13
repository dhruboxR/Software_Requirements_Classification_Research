
from pathlib import Path
from collections import Counter
import re
import unicodedata
import pandas as pd

from common import load_main, TABLES, REPORTS

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    ROOT / "A Unified Multi-Source Dataset of Software Require"
)

PROMISE_PATH = next(DATASET_ROOT.rglob("PROMISE_exp.txt"))

TEXT_COL = "Requirement Text (Original Requirement)"


def normalize(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip()

    # Remove matching enclosing quotes only.
    if len(text) >= 2:
        if (text[0], text[-1]) in [
            ("'", "'"),
            ('"', '"'),
            ("‘", "’"),
            ("“", "”"),
        ]:
            text = text[1:-1].strip()

    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def relaxed_normalize(value):
    text = normalize(value)

    # Secondary comparison only: remove punctuation
    # and whitespace. Matches require manual review.
    return "".join(
        c for c in text if c.isalnum()
    )


def parse_promise(path):
    lines = path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    records = []
    errors = []
    in_data = False

    for line_number, line in enumerate(lines, start=1):
        value = line.strip()

        if not in_data:
            if value.lower() == "@data":
                in_data = True
            continue

        if not value or value.startswith("%"):
            continue

        # First and last commas delimit the fields.
        # Requirement text may contain internal commas.
        parts = value.split(",", 1)

        if len(parts) != 2:
            errors.append((line_number, "Missing first comma"))
            continue

        project_id, remainder = parts

        if "," not in remainder:
            errors.append((line_number, "Missing last comma"))
            continue

        requirement, label = remainder.rsplit(",", 1)

        project_id = project_id.strip()
        requirement = requirement.strip()
        label = label.strip()

        # Reject malformed records rather than silently
        # treating their fields as valid.
        if not project_id or not requirement or not label:
            errors.append((line_number, "Empty required field"))
            continue

        records.append({
            "promise_line": line_number,
            "original_project_id": project_id,
            "original_requirement": requirement,
            "original_label": label,
            "normalized_text": normalize(requirement),
            "relaxed_text": relaxed_normalize(requirement),
        })

    if not in_data:
        raise ValueError("@DATA section not found")

    return pd.DataFrame(records), errors


print("=" * 75)
print("SCRIPT 18: PROMISE OVERLAP VERIFICATION")
print("=" * 75)

promise, errors = parse_promise(PROMISE_PATH)

combined = load_main("preprocessing").copy()

combined["normalized_text"] = combined[TEXT_COL].map(normalize)
combined["relaxed_text"] = combined[TEXT_COL].map(relaxed_normalize)

print("\nPROMISE DATA")
print("Parsed records:", len(promise))
print("Parsing errors:", len(errors))
print("Unique project IDs:", promise["original_project_id"].nunique())

print("\nPROMISE CLASS DISTRIBUTION")
print(promise["original_label"].value_counts().to_string())

# Preserve every original label and project for
# repeated normalized requirement texts.
reference = (
    promise.groupby("normalized_text")
    .agg(
        promise_labels=(
            "original_label",
            lambda x: "|".join(sorted(set(x)))
        ),
        promise_projects=(
            "original_project_id",
            lambda x: "|".join(sorted(set(x)))
        ),
        promise_occurrences=(
            "original_label",
            "size"
        )
    )
    .reset_index()
)

comparison = combined.merge(
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

matched = comparison[
    comparison["match_status"] == "matched"
].copy()

matched["label_agreement"] = matched.apply(
    lambda row: (
        str(row["Class"]).strip()
        in row["promise_labels"].split("|")
    ),
    axis=1
)

print("\n" + "=" * 75)
print("PROMISE OVERLAP BY COMBINED SOURCE")
print("=" * 75)

summary = []

for source, group in comparison.groupby("Datasets"):
    matches = group[group["match_status"] == "matched"]

    summary.append({
        "source": source,
        "total_rows": len(group),
        "matched_rows": len(matches),
        "unmatched_rows": len(group) - len(matches),
        "match_pct": round(
            100 * len(matches) / len(group), 2
        ),
        "distinct_matched_texts":
            matches["normalized_text"].nunique()
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

print("\nOVERALL EXACT MATCHES")
print("Matched combined rows:", len(matched))
print(
    "Distinct matched texts:",
    matched["normalized_text"].nunique()
)
print(
    "Matching labels:",
    int(matched["label_agreement"].sum())
)
print(
    "Nonmatching labels:",
    int((~matched["label_agreement"]).sum())
)

print("\nMATCHED LABEL COMBINATIONS")

label_mapping = (
    matched.groupby(["promise_labels", "Class"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(label_mapping.head(30).to_string(index=False))

# Secondary relaxed comparison for texts not found
# through standard normalization.
promise_relaxed = set(promise["relaxed_text"]) - {""}

unmatched = comparison[
    comparison["match_status"] == "unmatched"
].copy()

unmatched["relaxed_candidate"] = (
    unmatched["relaxed_text"].isin(promise_relaxed)
)

relaxed_summary = (
    unmatched.groupby("Datasets")["relaxed_candidate"]
    .sum()
    .reset_index(name="relaxed_candidate_rows")
)

print("\nRELAXED MATCH CANDIDATES")
print(relaxed_summary.to_string(index=False))

# Save reports.
summary_df.to_csv(
    TABLES / "18_promise_overlap_by_source.csv",
    index=False,
    encoding="utf-8-sig"
)

matched.to_csv(
    TABLES / "18_promise_matched_records.csv",
    index=False,
    encoding="utf-8-sig"
)

unmatched[unmatched["relaxed_candidate"]].to_csv(
    TABLES / "18_promise_relaxed_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

label_mapping.to_csv(
    TABLES / "18_promise_label_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)

promise.to_csv(
    TABLES / "18_promise_parsed_reference.csv",
    index=False,
    encoding="utf-8-sig"
)

report = [
    "PROMISE OVERLAP VERIFICATION",
    f"Parsed records: {len(promise)}",
    f"Parsing errors: {len(errors)}",
    f"Exact matched combined rows: {len(matched)}",
    f"Distinct matched texts: {matched['normalized_text'].nunique()}",
    "",
    "Exact matches use Unicode and whitespace normalization.",
    "Relaxed matches are candidates, not verified provenance.",
    "Project IDs and labels are not automatically remapped.",
    "Original dataset files were not modified.",
    "",
    "Parse errors:",
    *[f"Line {n}: {message}" for n, message in errors]
]

(REPORTS / "18_promise_overlap_summary.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved Script 18 outputs.")
