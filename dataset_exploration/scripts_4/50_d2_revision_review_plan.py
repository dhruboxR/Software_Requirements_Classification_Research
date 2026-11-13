
from pathlib import Path
import pandas as pd

# ============================================================
# SCRIPT 50: D2 REVISION REVIEW PLAN
# Read-only. No modifications to dataset or frozen split.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
REPORTS = BASE / "reports"
VALIDATED = BASE / "validated"

PAIRS = TABLES / "49_d2_cross_partition_semantic_review.csv"
MANIFEST = VALIDATED / "44_d2_frozen_split_manifest.csv"

for directory in (TABLES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

pairs = pd.read_csv(
    PAIRS, dtype=str, keep_default_na=False,
    encoding="utf-8-sig"
)

manifest = pd.read_csv(
    MANIFEST, dtype=str, keep_default_na=False,
    encoding="utf-8-sig"
)

if len(pairs) != 3:
    raise ValueError("Expected exactly three review pairs")

if manifest["Requirement ID"].duplicated().any():
    raise ValueError("Duplicate manifest Requirement IDs")

# Decisions are preliminary recommendations,
# not independent human annotation approvals.
decisions = {
    frozenset(["RQP0494680", "RQP0494674"]): {
        "assessment": "SEMANTICALLY_EQUIVALENT",
        "label_issue": "LIKELY_NFR_MISLABEL",
        "proposed_action": "QUARANTINE_BOTH",
        "reason": (
            "Both describe email-password login behavior. "
            "The NFR annotation requires independent review."
        )
    },
    frozenset(["RQP0482022", "RQP0481194"]): {
        "assessment": "SEMANTICALLY_EQUIVALENT",
        "label_issue": "NONE_IDENTIFIED",
        "proposed_action": "CONNECT_SIMILARITY_GROUP",
        "reason": (
            "Customer and consumer are used interchangeably "
            "for the same seller-notification behavior."
        )
    },
    frozenset(["RQP0495693", "RQP0495076"]): {
        "assessment": "STRONGLY_RELATED",
        "label_issue": "NONE_IDENTIFIED",
        "proposed_action": "CONNECT_SIMILARITY_GROUP",
        "reason": (
            "Both describe successful-validation redirection. "
            "The target pages differ, but conservative grouping "
            "reduces contamination risk."
        )
    }
}

lookup = manifest.set_index("Requirement ID")

records = []

for _, row in pairs.iterrows():
    id_a = row["requirement_id_a"]
    id_b = row["requirement_id_b"]

    pair_key = frozenset([id_a, id_b])

    if pair_key not in decisions:
        raise ValueError(
            f"Unrecognized pair: {id_a}, {id_b}"
        )

    if id_a not in lookup.index or id_b not in lookup.index:
        raise ValueError("Pair contains missing requirement ID")

    decision = decisions[pair_key]

    records.append({
        "pair_id": row["pair_id"],
        "requirement_id_a": id_a,
        "requirement_id_b": id_b,
        "split_a_v1": lookup.loc[id_a, "split"],
        "split_b_v1": lookup.loc[id_b, "split"],
        "label_a_v1": lookup.loc[id_a, "label"],
        "label_b_v1": lookup.loc[id_b, "label"],
        "similarity": float(row["similarity"]),
        "assessment": decision["assessment"],
        "label_issue": decision["label_issue"],
        "proposed_action": decision["proposed_action"],
        "reason": decision["reason"],
        "reviewer_confirmation": "PENDING",
        "decision_status": "PROVISIONAL"
    })

plan = pd.DataFrame(records)

if len(plan) != len(decisions):
    raise ValueError("Review plan is incomplete")

quarantine_ids = set()
connect_pairs = []

for _, row in plan.iterrows():
    if row["proposed_action"] == "QUARANTINE_BOTH":
        quarantine_ids.update([
            row["requirement_id_a"],
            row["requirement_id_b"]
        ])

    elif row["proposed_action"] == "CONNECT_SIMILARITY_GROUP":
        connect_pairs.append((
            row["requirement_id_a"],
            row["requirement_id_b"]
        ))

affected_ids = set()
for _, row in plan.iterrows():
    affected_ids.update([
        row["requirement_id_a"],
        row["requirement_id_b"]
    ])

affected = manifest[
    manifest["Requirement ID"].isin(affected_ids)
].copy()

affected["proposed_quarantine"] = (
    affected["Requirement ID"].isin(quarantine_ids)
)

if len(affected) != 6:
    raise ValueError("Expected six distinct affected rows")

plan_path = TABLES / "50_d2_provisional_review_decisions.csv"
impact_path = TABLES / "50_d2_revision_impact.csv"

plan.to_csv(
    plan_path, index=False, encoding="utf-8-sig"
)

affected.to_csv(
    impact_path, index=False, encoding="utf-8-sig"
)

print("=" * 76)
print("SCRIPT 50: D2 PROTOCOL REVISION PLAN")
print("=" * 76)

print("\nPROVISIONAL DECISIONS")
print(
    plan[[
        "pair_id",
        "assessment",
        "label_issue",
        "proposed_action",
        "reviewer_confirmation"
    ]].to_string(index=False)
)

print("\nREVISION IMPACT")
print("Affected requirement IDs:", len(affected_ids))
print("Proposed quarantined requirements:", len(quarantine_ids))
print("Proposed group connections:", len(connect_pairs))
print(
    "Affected original partitions:",
    sorted(affected["split"].unique())
)

print("\nQUARANTINE BY ORIGINAL PARTITION")
print(
    affected[affected["proposed_quarantine"]]
    ["split"].value_counts().to_string()
)

print("\nPROTOCOL STATUS")
print("Version 1.0: PRESERVED")
print("Version 1.1: PROPOSED")
print("Human annotation confirmation: PENDING")
print("Dataset or split modified: NO")
print("Model training: NOT STARTED")

report = [
    "D2 PROTOCOL VERSION 1.1 REVISION PLAN",
    "=" * 65,
    "",
    f"Reviewed pairs: {len(plan)}",
    f"Affected IDs: {len(affected_ids)}",
    f"Proposed quarantines: {len(quarantine_ids)}",
    f"Proposed group connections: {len(connect_pairs)}",
    "",
    plan.to_string(index=False),
    "",
    "Status: PROVISIONAL",
    "Human annotation confirmation: PENDING",
    "Original split modified: NO",
    "",
    "Next actions:",
    "1. Confirm semantic and label decisions.",
    "2. Rebuild candidate membership if approved.",
    "3. Recompute similarity-connected components.",
    "4. Create deterministic v1.1 partitions.",
    "5. Repeat contamination and integrity audits.",
    "6. Document final approval before training."
]

report_path = REPORTS / "50_d2_protocol_revision_plan.txt"

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")
print(plan_path)
print(impact_path)
print(report_path)

print("\nScript 50 completed.")
