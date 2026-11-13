
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"
VALIDATED = BASE / "validated"

INPUT = TABLES / "41_d2_source_label_audit.csv"

for folder in (TABLES, REPORTS, VALIDATED):
    folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    INPUT, dtype=str, keep_default_na=False,
    encoding="utf-8-sig"
)

required = {
    "Requirement ID", "key", "label",
    "source_label_status"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["key"].duplicated().any():
    raise ValueError("Duplicate normalized text keys detected")

if len(df) != 4871:
    raise ValueError(
        f"Expected 4871 audited texts, found {len(df)}"
    )

accepted = df[
    df["source_label_status"] == "SOURCE_LABEL_AGREES"
].copy()

pending = df[
    df["source_label_status"] != "SOURCE_LABEL_AGREES"
].copy()

accepted["candidate_status"] = (
    "SOURCE_CONSISTENT_NOT_APPROVED"
)

pending["candidate_status"] = (
    "WITHHELD_PENDING_REVIEW"
)

accepted.to_csv(
    VALIDATED / "42_d2_source_consistent_candidates.csv",
    index=False, encoding="utf-8-sig"
)

pending.to_csv(
    VALIDATED / "42_d2_withheld_label_cases.csv",
    index=False, encoding="utf-8-sig"
)

summary = (
    df.assign(
        decision=df["source_label_status"].map(
            lambda s: (
                "CANDIDATE"
                if s == "SOURCE_LABEL_AGREES"
                else "WITHHELD"
            )
        )
    )
    .groupby(["label", "decision"])
    .size()
    .reset_index(name="unique_texts")
)

summary.to_csv(
    TABLES / "42_d2_candidate_summary.csv",
    index=False, encoding="utf-8-sig"
)

print("=" * 70)
print("SCRIPT 42: D2 CANDIDATE MANIFEST")
print("=" * 70)

print("\nSOURCE-CONSISTENT CANDIDATES")
print("Total:", len(accepted))
print(accepted["label"].value_counts().to_string())

print("\nWITHHELD FOR FURTHER REVIEW")
print("Total:", len(pending))
print(
    pending["source_label_status"]
    .value_counts()
    .to_string()
)

print("\nCLASS SUMMARY")
print(summary.to_string(index=False))

print("\nCHECKS")
print("Candidate text keys unique:",
      accepted["key"].is_unique)

print("Overlap between candidate and withheld keys:",
      len(set(accepted["key"]) & set(pending["key"])))

print("All audited texts accounted for:",
      len(accepted) + len(pending) == len(df))

report = [
    "D2 SOURCE-CONSISTENT CANDIDATE MANIFEST",
    "=" * 65,
    f"Audited texts: {len(df)}",
    f"Candidates: {len(accepted)}",
    f"Withheld: {len(pending)}",
    "",
    summary.to_string(index=False),
    "",
    "CAUTIONS",
    "- Source agreement does not prove label correctness.",
    "- Prior manual-review pairs remain unresolved.",
    "- Near-duplicate grouping must be recomputed.",
    "- No final train/validation/test split is approved.",
    "- Original data is unchanged.",
]

(REPORTS / "42_d2_candidate_manifest.txt").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)

print("\nScript 42 completed.")
