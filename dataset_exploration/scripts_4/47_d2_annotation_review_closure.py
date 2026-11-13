
from pathlib import Path
import re
import unicodedata
import pandas as pd

# ============================================================
# SCRIPT 47: D2 ANNOTATION REVIEW CLOSURE AUDIT
# Read-only audit. Does not change labels or frozen splits.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
VALIDATED = BASE / "validated"
REPORTS = BASE / "reports"

DATA = (
    ROOT
    / "A Unified Multi-Source Dataset of Software Require"
    / "Software Requirement Dataset (SR vs NSR) - "
      "After the Preprocessing.csv"
)

PAIRS_FILE = TABLES / "32_label_disagreement_review.csv"
REVIEW_FILE = (
    VALIDATED / "phase1d_69_pair_preliminary_review.csv"
)
CANDIDATE_FILE = (
    VALIDATED / "42_d2_source_consistent_candidates.csv"
)
WITHHELD_FILE = (
    VALIDATED / "42_d2_withheld_label_cases.csv"
)
MANIFEST_FILE = (
    VALIDATED / "44_d2_frozen_split_manifest.csv"
)

for folder in (TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def normalize(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(
        r"\s+", " ", text
    ).strip().casefold()


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


print("=" * 76)
print("SCRIPT 47: D2 ANNOTATION REVIEW CLOSURE AUDIT")
print("=" * 76)

main = read_csv(DATA)
pairs = read_csv(PAIRS_FILE)
review = read_csv(REVIEW_FILE)
candidate = read_csv(CANDIDATE_FILE)
withheld = read_csv(WITHHELD_FILE)
manifest = read_csv(MANIFEST_FILE)

for name, frame in [
    ("Pair review", pairs),
    ("Preliminary review", review)
]:
    needed = {"source", "text_a", "text_b"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}"
        )

if not {"key", "label"}.issubset(candidate.columns):
    raise ValueError("Candidate columns missing")

if not {"key", "split"}.issubset(manifest.columns):
    raise ValueError("Manifest columns missing")

# ------------------------------------------------------------
# 1. Confirm frozen candidate integrity
# ------------------------------------------------------------

candidate_keys = set(candidate["key"].map(normalize))
withheld_keys = set(withheld["key"].map(normalize))
manifest_keys = set(manifest["key"].map(normalize))

if candidate_keys != manifest_keys:
    raise RuntimeError(
        "Candidate and frozen manifest do not match"
    )

if candidate_keys & withheld_keys:
    raise RuntimeError(
        "Withheld texts are present in candidate data"
    )

d2 = main[
    main["Datasets"].str.strip() == "D2"
].copy()

d2["key"] = d2[
    "Requirement Text (Original Requirement)"
].map(normalize)

d2["label"] = d2["Class"].str.strip().str.upper()

original_keys = set(d2["key"])

original_labels = (
    d2.groupby("key")["label"]
    .agg(lambda values: "|".join(
        sorted(set(values))
    ))
    .to_dict()
)

# ------------------------------------------------------------
# 2. Identify where reviewed texts ended up
# ------------------------------------------------------------

def location(key):
    if key in candidate_keys:
        return "RETAINED_IN_FROZEN_DATA"

    if key in withheld_keys:
        return "WITHHELD_SOURCE_LABEL_REVIEW"

    if key in original_keys:
        return "EXCLUDED_EARLIER"

    return "NOT_FOUND_IN_D2"


d2_pairs = pairs[
    pairs["source"].str.strip() == "D2"
].copy()

records = []

for index, row in d2_pairs.iterrows():
    a = normalize(row["text_a"])
    b = normalize(row["text_b"])

    status_a = location(a)
    status_b = location(b)

    records.append({
        "pair_row": index,
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "label_a_in_d2": original_labels.get(a, ""),
        "label_b_in_d2": original_labels.get(b, ""),
        "status_a": status_a,
        "status_b": status_b,
        "either_retained": (
            status_a == "RETAINED_IN_FROZEN_DATA"
            or status_b == "RETAINED_IN_FROZEN_DATA"
        ),
        "both_retained": (
            status_a == "RETAINED_IN_FROZEN_DATA"
            and status_b == "RETAINED_IN_FROZEN_DATA"
        ),
        "both_traceable_in_d2": (
            a in original_keys and b in original_keys
        )
    })

audit = pd.DataFrame(records)

if audit.empty:
    raise RuntimeError("No D2 review pairs found")

# ------------------------------------------------------------
# 3. Verify original 69-pair review coverage
# ------------------------------------------------------------

def pair_identifier(row):
    return (
        str(row["source"]).strip(),
        *sorted([
            normalize(row["text_a"]),
            normalize(row["text_b"])
        ])
    )


original_pair_keys = {
    pair_identifier(row)
    for _, row in pairs.iterrows()
}

review_pair_keys = {
    pair_identifier(row)
    for _, row in review.iterrows()
}

missing_reviews = original_pair_keys - review_pair_keys
extra_reviews = review_pair_keys - original_pair_keys

review_d2 = review[
    review["source"].str.strip() == "D2"
]

# Show optional annotation columns without assuming
# that a preliminary comment equals human approval.
review_metadata_columns = [
    col for col in review.columns
    if col not in pairs.columns
]

# ------------------------------------------------------------
# 4. Check review exposure in frozen dataset
# ------------------------------------------------------------

flagged_keys = set()

for _, row in d2_pairs.iterrows():
    flagged_keys.add(normalize(row["text_a"]))
    flagged_keys.add(normalize(row["text_b"]))

flagged_keys.discard("")

retained_flagged = flagged_keys & manifest_keys

flagged_status = pd.DataFrame([
    {
        "normalized_text": key,
        "status": location(key),
        "original_d2_labels": original_labels.get(key, "")
    }
    for key in sorted(flagged_keys)
])

# ------------------------------------------------------------
# 5. Save audit outputs
# ------------------------------------------------------------

audit_path = TABLES / "47_d2_pair_closure_audit.csv"
flagged_path = TABLES / "47_d2_flagged_text_status.csv"

audit.to_csv(
    audit_path,
    index=False,
    encoding="utf-8-sig"
)

flagged_status.to_csv(
    flagged_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nREVIEW COVERAGE")
print("Original review pairs:", len(pairs))
print("Preliminary review rows:", len(review))
print("D2 original review pairs:", len(d2_pairs))
print("D2 preliminary review rows:", len(review_d2))
print("Missing review pair matches:", len(missing_reviews))
print("Extra review pair matches:", len(extra_reviews))

print("\nD2 FLAGGED REQUIREMENT STATUS")
print("Unique flagged D2 texts:", len(flagged_keys))
print(
    flagged_status["status"]
    .value_counts()
    .to_string()
)

print("\nFROZEN DATA EXPOSURE")
print(
    "D2 review pairs with either member retained:",
    int(audit["either_retained"].sum())
)
print(
    "D2 review pairs with both members retained:",
    int(audit["both_retained"].sum())
)
print(
    "Flagged texts retained in frozen manifest:",
    len(retained_flagged)
)
print(
    "D2 pairs traceable to original D2:",
    int(audit["both_traceable_in_d2"].sum())
)

print("\nPRELIMINARY REVIEW METADATA")
print(
    "Additional review columns:",
    review_metadata_columns
)
print(
    "Human annotation approval:",
    "NOT VERIFIED BY THIS SCRIPT"
)

# Technical screening only, not independent
# scientific annotation approval.
screening_pass = (
    len(missing_reviews) == 0
    and len(retained_flagged) == 0
    and bool(audit["both_traceable_in_d2"].all())
)

print("\nAUDIT CONCLUSION")
print(
    "Disputed-text exclusion screen:",
    "PASS" if screening_pass else "REVIEW_REQUIRED"
)
print("Frozen partitions changed: NO")
print("Scientific label approval: PENDING")

report = [
    "SCRIPT 47: D2 ANNOTATION REVIEW CLOSURE",
    "=" * 65,
    "",
    f"Original review pairs: {len(pairs)}",
    f"Preliminary review rows: {len(review)}",
    f"D2 review pairs: {len(d2_pairs)}",
    f"Unique D2 flagged texts: {len(flagged_keys)}",
    f"Missing preliminary pair matches: {len(missing_reviews)}",
    f"Flagged texts retained: {len(retained_flagged)}",
    "",
    "D2 FLAGGED REQUIREMENT STATUS",
    flagged_status["status"].value_counts().to_string(),
    "",
    "SCREENING",
    "PASS" if screening_pass else "REVIEW_REQUIRED",
    "",
    "SCIENTIFIC APPROVAL",
    "PENDING",
    "",
    "Limitations:",
    "- Pair matching does not establish annotation correctness.",
    "- Preliminary annotations are not independently approved.",
    "- Exclusion does not resolve the original disagreements.",
    "- Undetected semantic conflicts may remain.",
    "- The frozen split was not changed."
]

report_path = REPORTS / "47_d2_annotation_review_closure.txt"

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(audit_path)
print(flagged_path)
print(report_path)

print("\nScript 47 completed.")
