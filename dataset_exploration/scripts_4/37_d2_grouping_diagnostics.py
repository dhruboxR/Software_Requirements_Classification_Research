
from pathlib import Path
from collections import Counter
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

MAIN = (
    ROOT
    / "A Unified Multi-Source Dataset of Software Require"
    / "Software Requirement Dataset (SR vs NSR) - After the Preprocessing.csv"
)

PAIRS = TABLES / "30_near_duplicate_candidates.csv"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def normalize(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


df = pd.read_csv(MAIN, dtype=str, keep_default_na=False)
pairs = pd.read_csv(PAIRS, dtype=str, keep_default_na=False)

d2 = df[
    (df["Datasets"].str.strip() == "D2")
    & (df["Class"].str.strip().isin(["FR", "NFR"]))
].copy()

d2["text_key"] = d2[
    "Requirement Text (Original Requirement)"
].map(normalize)

d2["project_name"] = d2["Project Name"].map(normalize)
d2["label"] = d2["Class"].str.strip()

projects = (
    d2.groupby("project_name")
    .agg(
        rows=("label", "size"),
        unique_texts=("text_key", "nunique"),
        fr=("label", lambda s: int((s == "FR").sum())),
        nfr=("label", lambda s: int((s == "NFR").sum())),
    )
    .reset_index()
)

projects["minority_label_present"] = projects["nfr"] > 0
projects["both_labels"] = (
    (projects["fr"] > 0) & (projects["nfr"] > 0)
)

text_projects = (
    d2.groupby("text_key")["project_name"]
    .agg(lambda s: set(s))
    .to_dict()
)

cross_project = []
for _, row in pairs[pairs["source"] == "D2"].iterrows():
    a = normalize(row["text_a"])
    b = normalize(row["text_b"])

    pa = text_projects.get(a, set())
    pb = text_projects.get(b, set())

    if not pa or not pb:
        continue

    cross_project.append({
        "similarity": float(row["similarity"]),
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "projects_a": "|".join(sorted(pa)),
        "projects_b": "|".join(sorted(pb)),
        "disjoint_project_names": pa.isdisjoint(pb),
    })

cross = pd.DataFrame(cross_project)

projects.to_csv(
    TABLES / "37_d2_project_name_distribution.csv",
    index=False, encoding="utf-8-sig"
)

cross.to_csv(
    TABLES / "37_d2_cross_project_similarity.csv",
    index=False, encoding="utf-8-sig"
)

print("=" * 75)
print("SCRIPT 37: D2 GROUPING DIAGNOSTICS")
print("=" * 75)

print("\nPROJECT-NAME COVERAGE")
print("Binary-labeled rows:", len(d2))
print("Distinct project names:", len(projects))
print("Projects containing FR:", int((projects["fr"] > 0).sum()))
print("Projects containing NFR:", int((projects["nfr"] > 0).sum()))
print("Projects containing both:", int(projects["both_labels"].sum()))
print(
    "Rows in largest reported project:",
    int(projects["rows"].max())
)
print(
    "Largest project share:",
    round(100 * projects["rows"].max() / len(d2), 2),
    "%"
)

print("\nNEAR-DUPLICATE PROJECT RELATIONSHIPS")
print("Candidate pairs checked:", len(cross))
print(
    "Pairs with disjoint reported project names:",
    int(cross["disjoint_project_names"].sum())
    if len(cross) else 0
)

print("\nTOP 10 PROJECT NAMES BY ROW COUNT")
print(
    projects.sort_values("rows", ascending=False)
    .head(10)
    .to_string(index=False)
)

notes = [
    "D2 PROJECT GROUPING DIAGNOSTICS",
    "=" * 65,
    "",
    f"Binary rows: {len(d2)}",
    f"Reported project names: {len(projects)}",
    f"Projects with NFR: {int((projects['nfr'] > 0).sum())}",
    f"Projects with both labels: {int(projects['both_labels'].sum())}",
    "",
    "Limitations:",
    "- Reported project names are not verified original project identities.",
    "- Matching names may refer to unrelated requirements.",
    "- Different names may refer to the same underlying project.",
    "- The source workbook lacks reliable project metadata.",
    "- Project-name groups cannot yet support an independent-project claim.",
    "- No train/test splits or label changes were made.",
]

report = REPORTS / "37_d2_grouping_diagnostics.txt"
report.write_text("\n".join(notes) + "\n", encoding="utf-8")

print("\nSaved:", report)
print("Script 37 completed.")

