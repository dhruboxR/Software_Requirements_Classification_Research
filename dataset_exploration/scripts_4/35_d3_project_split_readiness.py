
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "A Unified Multi-Source Dataset of Software Require"
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

MAIN = DATA_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

MAPPING = TABLES / "21_d3_original_project_recovery.csv"
PAIRS = TABLES / "30_near_duplicate_candidates.csv"

def normalize(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()

df = pd.read_csv(MAIN, dtype=str, keep_default_na=False)
mapping = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
pairs = pd.read_csv(PAIRS, dtype=str, keep_default_na=False)

d3 = df[df["Datasets"].str.strip() == "D3"].copy()

mapping = mapping.rename(columns={
    "Original Project IDs": "original_project",
    "Mapping Status": "mapping_status"
})

if mapping["Requirement ID"].duplicated().any():
    raise ValueError("Duplicate Requirement IDs in recovery file")

d3 = d3.merge(
    mapping[[
        "Requirement ID",
        "original_project",
        "mapping_status"
    ]],
    on="Requirement ID",
    how="left",
    validate="one_to_one"
)

d3["text_key"] = d3[
    "Requirement Text (Original Requirement)"
].map(normalize)

d3["label"] = d3["Class"].str.strip()

d3["verified_unique"] = (
    d3["mapping_status"].eq("unique_original_project")
    & d3["original_project"].notna()
    & d3["original_project"].ne("")
    & ~d3["original_project"].str.contains(
        "|", regex=False, na=False
    )
)

# Exclude texts containing contradictory labels
# from the provisional support calculation.
conflicting_texts = set(
    d3.groupby("text_key")["label"]
    .nunique()
    .loc[lambda x: x > 1]
    .index
)

d3["text_label_conflict"] = (
    d3["text_key"].isin(conflicting_texts)
)

# Provisional eligible rows have unique source-project
# recovery and no exact-text label contradiction.
d3["provisionally_eligible"] = (
    d3["verified_unique"]
    & ~d3["text_label_conflict"]
    & d3["text_key"].ne("")
)

eligible = d3[d3["provisionally_eligible"]].copy()

class_rows = []

for label, group in d3.groupby("label"):
    clean = eligible[eligible["label"] == label]

    unique_projects = clean["original_project"].nunique()

    project_counts = (
        clean.groupby("original_project").size()
    )

    class_rows.append({
        "label": label,
        "total_rows": len(group),
        "unique_project_rows": int(
            group["verified_unique"].sum()
        ),
        "provisionally_eligible_rows": len(clean),
        "eligible_unique_texts": clean["text_key"].nunique(),
        "eligible_original_projects": unique_projects,
        "largest_project_rows": (
            int(project_counts.max())
            if len(project_counts) else 0
        ),
        "largest_project_share_pct": (
            round(100 * project_counts.max() / len(clean), 2)
            if len(clean) else 0
        ),
        "minimum_20_test_examples_met": (
            "NOT_TESTED"
        ),
    })

class_summary = pd.DataFrame(class_rows)

# Identify texts that correspond to multiple
# recovered original projects.
project_sets = (
    eligible.groupby("text_key")["original_project"]
    .agg(lambda x: sorted(set(x)))
)

cross_project_exact = {
    text for text, projects in project_sets.items()
    if len(projects) > 1
}

# Inspect near-duplicate connections across projects.
d3_pairs = pairs[pairs["source"] == "D3"].copy()

text_projects = {
    key: set(group["original_project"])
    for key, group in eligible.groupby("text_key")
}

pair_rows = []

for _, row in d3_pairs.iterrows():
    a = normalize(row["text_a"])
    b = normalize(row["text_b"])

    projects_a = text_projects.get(a, set())
    projects_b = text_projects.get(b, set())

    if not projects_a or not projects_b:
        continue

    pair_rows.append({
        "similarity": row["similarity"],
        "labels_a": row["labels_a"],
        "labels_b": row["labels_b"],
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "projects_a": "|".join(sorted(projects_a)),
        "projects_b": "|".join(sorted(projects_b)),
        "different_project_sets": projects_a != projects_b,
        "disjoint_projects": projects_a.isdisjoint(projects_b),
        "review_status": "PENDING",
    })

cross_pairs = pd.DataFrame(pair_rows)

class_summary.to_csv(
    TABLES / "35_d3_original_project_class_support.csv",
    index=False, encoding="utf-8-sig"
)

cross_pairs.to_csv(
    TABLES / "35_d3_cross_project_similarity.csv",
    index=False, encoding="utf-8-sig"
)

print("=" * 78)
print("SCRIPT 35: D3 ORIGINAL-PROJECT SPLIT READINESS")
print("=" * 78)

print("\nPROJECT RECOVERY")
print("Total D3 rows:", len(d3))
print("Unique project recovery rows:", int(d3["verified_unique"].sum()))
print("Provisional eligible rows:", len(eligible))
print("Exact-text conflicting groups:", len(conflicting_texts))
print(
    "Eligible texts in multiple original projects:",
    len(cross_project_exact)
)

print("\nCLASS-LEVEL PROJECT SUPPORT")
print(class_summary.to_string(index=False))

print("\nNEAR-DUPLICATE PROJECT RELATIONSHIPS")
print("Pairs checked:", len(cross_pairs))

if len(cross_pairs):
    print(
        "Pairs with disjoint recovered project sets:",
        int(cross_pairs["disjoint_projects"].sum())
    )
else:
    print("No eligible pairs available")

report = [
    "PHASE 1E: D3 PROJECT SPLIT READINESS",
    "=" * 65,
    f"Total rows: {len(d3)}",
    f"Unique-project recovered rows: {int(d3['verified_unique'].sum())}",
    f"Provisional eligible rows: {len(eligible)}",
    f"Exact-text conflicts: {len(conflicting_texts)}",
    f"Cross-project exact-text groups: {len(cross_project_exact)}",
    "",
    class_summary.to_string(index=False),
    "",
    "Limitations:",
    "- Original project matches are provisional source evidence.",
    "- Near-duplicate pairs can connect different projects.",
    "- Eligibility does not mean manual label approval.",
    "- Minority test support has not been simulated.",
    "- No train/validation/test partition was created.",
]

path = REPORTS / "35_d3_project_split_readiness.txt"
path.write_text("\n".join(report) + "\n", encoding="utf-8")

print("\nSaved:", path)
print("Script 35 completed.")
