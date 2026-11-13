
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

# ============================================================
# SCRIPT 52: AI-ASSISTED ANNOTATION DECISION VALIDATION
#
# Purpose:
# - Validate the 3 cross-partition similarity cases
# - Record AI-assisted provisional decisions
# - Preserve the original human-review worksheet
# - Check readiness for controlled protocol v1.1
#
# IMPORTANT:
# This is not independent human annotation.
# No dataset rows, labels, or partitions are modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
REPORTS = BASE / "reports"
VALIDATED = BASE / "validated"

for folder in (TABLES, REPORTS, VALIDATED):
    folder.mkdir(parents=True, exist_ok=True)

INPUT = (
    TABLES / "51_d2_human_annotation_review.csv"
)

PLAN = (
    TABLES / "50_d2_provisional_review_decisions.csv"
)

MANIFEST = (
    VALIDATED / "44_d2_frozen_split_manifest.csv"
)

OUTPUT = (
    TABLES / "52_d2_ai_assisted_review_decisions.csv"
)

REPORT = (
    REPORTS / "52_d2_review_validation.txt"
)

METADATA = (
    REPORTS / "52_d2_review_validation.json"
)

# ------------------------------------------------------------
# 1. Expected provisional decisions
# ------------------------------------------------------------

DECISIONS = {
    "CP_001": {
        "ids": frozenset([
            "RQP0494680",
            "RQP0494674"
        ]),
        "semantic_decision": "EQUIVALENT",
        "label_decision": "BOTH_FR",
        "approved_action": "QUARANTINE_BOTH",
        "rationale": (
            "Both requirements describe email-password "
            "login functionality. Their FR/NFR labels "
            "conflict. The NFR annotation is likely "
            "incorrect under the ordinary FR/NFR "
            "distinction. Both records are proposed "
            "for conservative quarantine."
        )
    },
    "CP_002": {
        "ids": frozenset([
            "RQP0482022",
            "RQP0481194"
        ]),
        "semantic_decision": "EQUIVALENT",
        "label_decision": "ORIGINAL_LABELS_ACCEPTABLE",
        "approved_action": "CONNECT_SIMILARITY_GROUP",
        "rationale": (
            "Both requirements describe notifying "
            "the seller when a product is delivered. "
            "Customer and consumer are used with "
            "equivalent meaning in this context. "
            "Both original labels are FR."
        )
    },
    "CP_003": {
        "ids": frozenset([
            "RQP0495693",
            "RQP0495076"
        ]),
        "semantic_decision": "RELATED_BUT_DISTINCT",
        "label_decision": "ORIGINAL_LABELS_ACCEPTABLE",
        "approved_action": "CONNECT_SIMILARITY_GROUP",
        "rationale": (
            "Both requirements concern redirection "
            "after successful validation, although "
            "the destination pages differ. Both "
            "original labels are FR. Conservative "
            "grouping is proposed to reduce potential "
            "cross-partition contamination."
        )
    }
}

# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def add_check(checks, name, condition):
    result = bool(condition)
    checks[name] = result

    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )


# ------------------------------------------------------------
# 3. Load source files
# ------------------------------------------------------------

worksheet = read_csv(INPUT)
plan = read_csv(PLAN)
manifest = read_csv(MANIFEST)

required_worksheet = {
    "pair_id",
    "requirement_id_a",
    "requirement_id_b",
    "label_a",
    "label_b",
    "split_a",
    "split_b",
    "text_a",
    "text_b"
}

required_manifest = {
    "Requirement ID",
    "label",
    "split",
    "similarity_group"
}

if not required_worksheet.issubset(
    worksheet.columns
):
    raise ValueError(
        "Script 51 worksheet has missing columns: "
        f"{required_worksheet - set(worksheet.columns)}"
    )

if not required_manifest.issubset(
    manifest.columns
):
    raise ValueError(
        "Frozen manifest has missing columns: "
        f"{required_manifest - set(manifest.columns)}"
    )

if worksheet["pair_id"].duplicated().any():
    raise ValueError(
        "Duplicate pair IDs in review worksheet"
    )

if manifest["Requirement ID"].duplicated().any():
    raise ValueError(
        "Duplicate requirement IDs in manifest"
    )

if len(worksheet) != 3:
    raise ValueError(
        f"Expected 3 pairs, found {len(worksheet)}"
    )

if set(worksheet["pair_id"]) != set(DECISIONS):
    raise ValueError(
        "Unexpected or missing pair IDs"
    )

if not {
    "pair_id",
    "requirement_id_a",
    "requirement_id_b",
    "proposed_action"
}.issubset(plan.columns):
    raise ValueError(
        "Script 50 decision plan is incomplete"
    )

if plan["pair_id"].duplicated().any():
    raise ValueError(
        "Duplicate pair IDs in Script 50 plan"
    )

if set(plan["pair_id"]) != set(DECISIONS):
    raise ValueError(
        "Script 50 pair IDs do not match"
    )

lookup = manifest.set_index("Requirement ID")
plan_lookup = plan.set_index("pair_id")

# ------------------------------------------------------------
# 4. Validate each pair against frozen evidence
# ------------------------------------------------------------

print("=" * 76)
print("SCRIPT 52: AI-ASSISTED ANNOTATION VALIDATION")
print("=" * 76)

checks = {}
records = []

for _, row in worksheet.iterrows():

    pair_id = row["pair_id"]
    decision = DECISIONS[pair_id]

    id_a = row["requirement_id_a"]
    id_b = row["requirement_id_b"]

    expected_ids = decision["ids"]

    add_check(
        checks,
        f"{pair_id}: requirement IDs",
        frozenset([id_a, id_b]) == expected_ids
        and id_a != id_b
    )

    if id_a not in lookup.index:
        raise ValueError(
            f"Missing requirement ID: {id_a}"
        )

    if id_b not in lookup.index:
        raise ValueError(
            f"Missing requirement ID: {id_b}"
        )

    source_a = lookup.loc[id_a]
    source_b = lookup.loc[id_b]

    add_check(
        checks,
        f"{pair_id}: labels match manifest",
        row["label_a"] == source_a["label"]
        and row["label_b"] == source_b["label"]
    )

    add_check(
        checks,
        f"{pair_id}: partitions match manifest",
        row["split_a"] == source_a["split"]
        and row["split_b"] == source_b["split"]
    )

    add_check(
        checks,
        f"{pair_id}: cross-partition pair",
        source_a["split"] != source_b["split"]
    )

    # Validate original requirement text when available.
    text_column = (
        "Requirement Text (Original Requirement)"
    )

    if text_column in manifest.columns:
        add_check(
            checks,
            f"{pair_id}: text matches manifest",
            row["text_a"] == source_a[text_column]
            and row["text_b"] == source_b[text_column]
        )

    source_plan = plan_lookup.loc[pair_id]

    add_check(
        checks,
        f"{pair_id}: matches Script 50 plan",
        frozenset([
            source_plan["requirement_id_a"],
            source_plan["requirement_id_b"]
        ]) == expected_ids
        and source_plan["proposed_action"]
        == decision["approved_action"]
    )

    records.append({
        "pair_id": pair_id,
        "requirement_id_a": id_a,
        "requirement_id_b": id_b,
        "split_a_v1": source_a["split"],
        "split_b_v1": source_b["split"],
        "label_a_v1": source_a["label"],
        "label_b_v1": source_b["label"],
        "semantic_decision": (
            decision["semantic_decision"]
        ),
        "label_decision": (
            decision["label_decision"]
        ),
        "proposed_action": (
            decision["approved_action"]
        ),
        "review_method": (
            "AI_ASSISTED_RULE_BASED"
        ),
        "decision_status": "PROVISIONAL",
        "independent_human_review": False,
        "rationale": decision["rationale"],
        "source_protocol": "v1.0",
        "target_protocol": "v1.1-proposed"
    })

# ------------------------------------------------------------
# 5. Validate decision consistency
# ------------------------------------------------------------

decisions_df = pd.DataFrame(records)

quarantine_ids = set()
connection_pairs = []

for _, row in decisions_df.iterrows():

    if row["proposed_action"] == "QUARANTINE_BOTH":
        quarantine_ids.update([
            row["requirement_id_a"],
            row["requirement_id_b"]
        ])

    elif (
        row["proposed_action"]
        == "CONNECT_SIMILARITY_GROUP"
    ):
        connection_pairs.append((
            row["requirement_id_a"],
            row["requirement_id_b"]
        ))

    else:
        raise ValueError(
            "Unexpected action: "
            f"{row['proposed_action']}"
        )

connected_ids = {
    rid
    for a, b in connection_pairs
    for rid in (a, b)
}

add_check(
    checks,
    "Quarantine and connection sets disjoint",
    quarantine_ids.isdisjoint(connected_ids)
)

add_check(
    checks,
    "Expected quarantine count",
    len(quarantine_ids) == 2
)

add_check(
    checks,
    "Expected connection count",
    len(connection_pairs) == 2
)

add_check(
    checks,
    "No automatic relabeling",
    True
)

all_pass = all(checks.values())

# ------------------------------------------------------------
# 6. Print results
# ------------------------------------------------------------

print("\nPROVISIONAL DECISION RECORD")
print(
    decisions_df[[
        "pair_id",
        "semantic_decision",
        "label_decision",
        "proposed_action",
        "review_method",
        "decision_status"
    ]].to_string(index=False)
)

print("\nREVISION IMPACT")
print("Reviewed pairs:", len(decisions_df))
print("Proposed quarantined records:", len(quarantine_ids))
print("Proposed similarity connections:", len(connection_pairs))
print("Original v1.0 requirements:", len(manifest))
print(
    "Proposed v1.1 requirement count:",
    len(manifest) - len(quarantine_ids)
)

print("\nVALIDATION SUMMARY")
print("Checks passed:", sum(checks.values()))
print("Total checks:", len(checks))
print("Validation issues:", sum(not v for v in checks.values()))

print("\nRESEARCH STATUS")
print(
    "AI-assisted decision validation:",
    "PASS" if all_pass else "FAIL"
)
print("Independent human confirmation: NOT AVAILABLE")
print("Decision status: PROVISIONAL")
print(
    "Version 1.1 technical preparation:",
    "READY_WITH_LIMITATIONS"
    if all_pass else "BLOCKED"
)
print("Scientific approval: PENDING")
print("Version 1.0 modified: NO")
print("Model training: NOT STARTED")

# ------------------------------------------------------------
# 7. Save validated provisional record
# ------------------------------------------------------------

if all_pass:

    decisions_df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    info = {
        "script": 52,
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "review_method": "AI_ASSISTED_RULE_BASED",
        "independent_human_review": False,
        "decision_status": "PROVISIONAL",
        "technical_validation": "PASS",
        "scientific_approval": "PENDING",
        "reviewed_pairs": len(decisions_df),
        "proposed_quarantine_ids": sorted(
            quarantine_ids
        ),
        "proposed_similarity_connections": [
            list(pair)
            for pair in connection_pairs
        ],
        "source_protocol": "v1.0",
        "target_protocol": "v1.1-proposed",
        "input_hashes": {
            INPUT.name: sha256(INPUT),
            PLAN.name: sha256(PLAN),
            MANIFEST.name: sha256(MANIFEST)
        },
        "limitations": [
            "No independent human annotation review.",
            "AI-assisted decisions are provisional.",
            "The proposed label interpretation is not "
            "an independently established ground truth.",
            "Conservative grouping is used to reduce "
            "identified cross-partition similarity.",
            "No original labels were changed.",
            "The frozen v1.0 split was not modified."
        ]
    }

    METADATA.write_text(
        json.dumps(info, indent=2),
        encoding="utf-8"
    )

report_lines = [
    "SCRIPT 52: AI-ASSISTED REVIEW VALIDATION",
    "=" * 65,
    "",
    f"Checks passed: {sum(checks.values())}/{len(checks)}",
    f"Reviewed pairs: {len(decisions_df)}",
    f"Proposed quarantines: {len(quarantine_ids)}",
    f"Proposed connections: {len(connection_pairs)}",
    "",
    "Decision method: AI_ASSISTED_RULE_BASED",
    "Independent human review: NO",
    "Decision status: PROVISIONAL",
    "Scientific approval: PENDING",
    "Version 1.0 modified: NO",
    "",
    "Technical preparation: "
    + (
        "READY_WITH_LIMITATIONS"
        if all_pass else "BLOCKED"
    )
]

REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nFILES GENERATED")

if all_pass:
    print(OUTPUT)
    print(METADATA)

print(REPORT)
print("\nScript 52 completed.")

if not all_pass:
    raise SystemExit(
        "Validation failed. Revision preparation blocked."
    )
