
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    ROOT / "A Unified Multi-Source Dataset of Software Require"
)
TABLES = ROOT / "dataset_exploration/outputs/tables"
REPORTS = ROOT / "dataset_exploration/outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_ROOT / (
    "Software Requirement Dataset (SR vs NSR) - "
    "After the Preprocessing.csv"
)

df = pd.read_csv(
    DATA_FILE,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

required = {
    "Datasets", "Project ID", "Project Name", "Class"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

print("=" * 78)
print("SCRIPT 34: PROJECT IDENTITY VERIFICATION")
print("=" * 78)

summary = []
ambiguous = []

for source in ["D1", "D2", "D3", "D4", "D5"]:
    part = df[df["Datasets"].str.strip() == source].copy()

    part["project_id"] = part["Project ID"].str.strip()
    part["project_name"] = part["Project Name"].str.strip()

    valid = part[
        (part["project_id"] != "")
        & (part["project_name"] != "")
    ].copy()

    by_id = (
        valid.groupby("project_id")
        .agg(
            rows=("project_name", "size"),
            distinct_names=("project_name", "nunique")
        )
        .reset_index()
    )

    ambiguous_ids = by_id[
        by_id["distinct_names"] > 1
    ].copy()

    for _, record in ambiguous_ids.iterrows():
        pid = record["project_id"]

        names = (
            valid.loc[
                valid["project_id"] == pid,
                "project_name"
            ]
            .value_counts()
        )

        ambiguous.append({
            "source": source,
            "project_id": pid,
            "rows": record["rows"],
            "distinct_names": record["distinct_names"],
            "example_names": " | ".join(
                names.index[:5].tolist()
            )
        })

    ambiguous_row_count = int(
        valid["project_id"].isin(
            ambiguous_ids["project_id"]
        ).sum()
    )

    summary.append({
        "source": source,
        "rows": len(part),
        "distinct_combined_project_ids":
            part["project_id"].replace("", pd.NA).nunique(),
        "distinct_project_names":
            part["project_name"].replace("", pd.NA).nunique(),
        "ids_with_multiple_names": len(ambiguous_ids),
        "rows_under_ambiguous_ids": ambiguous_row_count,
        "missing_project_id_rows": int(
            (part["project_id"] == "").sum()
        ),
        "missing_project_name_rows": int(
            (part["project_name"] == "").sum()
        ),
        "project_identity_verified": "NO",
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    TABLES / "34_project_metadata_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

ambiguous_df = pd.DataFrame(ambiguous)

ambiguous_df.to_csv(
    TABLES / "34_ambiguous_project_ids.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCOMBINED PROJECT METADATA SUMMARY")
print(summary_df.to_string(index=False))

print("\nHIGHEST-RISK PROJECT IDS")

if len(ambiguous_df):
    print(
        ambiguous_df.sort_values(
            ["rows", "distinct_names"],
            ascending=False
        ).head(12).to_string(index=False)
    )
else:
    print("No ID-to-name collisions detected.")

# Inspect previously recovered original DOSSPRE
# project mapping without assuming its column names.
candidates = [
    TABLES / "21_d3_original_project_recovery.csv",
    TABLES / "20_d3_project_mapping_candidates.csv",
]

print("\nD3 ORIGINAL PROJECT RECOVERY FILES")

for path in candidates:
    if not path.exists():
        print("Not found:", path.name)
        continue

    recovered = pd.read_csv(
        path, dtype=str, keep_default_na=False
    )

    print("\nFile:", path.name)
    print("Rows:", len(recovered))
    print("Columns:", recovered.columns.tolist())

    print("First 3 records:")
    for record in recovered.head(3).to_dict("records"):
        compact = {
            k: str(v)[:100]
            for k, v in record.items()
        }
        print(compact)

report = [
    "PHASE 1E: PROJECT IDENTITY VERIFICATION",
    "=" * 70,
    "",
    summary_df.to_string(index=False),
    "",
    "Interpretation:",
    "- Multiple names per ID indicate metadata ambiguity.",
    "- One name per ID does not establish true identity.",
    "- D2 combined project IDs are not approved for grouping.",
    "- D3 original project mappings require separate verification.",
    "- D4 generation model and round may create additional dependencies.",
    "- No train/test partition has been generated.",
]

report_path = REPORTS / "34_project_identity_verification.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", report_path)
print("Script 34 completed.")
