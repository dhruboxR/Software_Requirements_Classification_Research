
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

# ==========================================================
# SCRIPT 56: D2 v1.1 RESEARCH READINESS AUDIT
#
# Read-only verification of frozen research inputs.
# Creates a final status report.
# Does not train models or change partitions.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

VALIDATED = BASE / "validated"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

CANDIDATES = VALIDATED / "53_d2_v11_candidates.csv"
GROUPS = TABLES / "53_d2_v11_group_membership.csv"
QUARANTINE = VALIDATED / "53_d2_v11_quarantine_audit.csv"

PREP_METADATA = (
    REPORTS / "53_d2_v11_preparation_metadata.json"
)

MANIFEST = (
    VALIDATED / "54_d2_v11_frozen_split_manifest.csv"
)

SPLIT_METADATA = (
    VALIDATED / "54_d2_v11_split_metadata.json"
)

AUDIT_METADATA = (
    REPORTS / "55_d2_v11_similarity_audit.json"
)

AUDIT_SUMMARY = (
    TABLES / "55_d2_v11_similarity_summary.csv"
)

OUTPUT_JSON = (
    REPORTS / "56_d2_v11_readiness.json"
)

OUTPUT_TXT = (
    REPORTS / "56_d2_v11_readiness.txt"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def load_csv(path):
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


# ----------------------------------------------------------
# 1. Load files
# ----------------------------------------------------------

candidate = load_csv(CANDIDATES)
groups = load_csv(GROUPS)
quarantine = load_csv(QUARANTINE)
manifest = load_csv(MANIFEST)
audit_summary = load_csv(AUDIT_SUMMARY)

prep = json.loads(PREP_METADATA.read_text("utf-8"))
split = json.loads(SPLIT_METADATA.read_text("utf-8"))
audit = json.loads(AUDIT_METADATA.read_text("utf-8"))

check(
    prep["protocol"] == split["protocol"] == audit["protocol"]
    == "D2-v1.1",
    "Protocol versions disagree"
)

# ----------------------------------------------------------
# 2. Verify hashes
# ----------------------------------------------------------

for path in (CANDIDATES, GROUPS, QUARANTINE):
    expected = prep["output_sha256"].get(path.name)
    check(
        expected == sha256(path),
        f"Preparation file changed: {path.name}"
    )

for path in (MANIFEST,):
    expected = split["output_sha256"].get(path.name)
    check(
        expected == sha256(path),
        f"Frozen manifest changed: {path.name}"
    )

check(
    audit["frozen_manifest_sha256"] == sha256(MANIFEST),
    "Audit and frozen manifest hashes disagree"
)

check(
    audit["output_sha256"][AUDIT_SUMMARY.name]
    == sha256(AUDIT_SUMMARY),
    "Similarity summary changed after audit"
)

# ----------------------------------------------------------
# 3. Candidate and grouping integrity
# ----------------------------------------------------------

check(len(candidate) == 4833, "Wrong candidate count")
check(len(manifest) == 4833, "Wrong manifest count")
check(len(groups) == 4833, "Wrong membership count")

for name, frame in [
    ("Candidate", candidate),
    ("Membership", groups),
    ("Manifest", manifest)
]:
    check(
        frame["key"].is_unique,
        f"{name} contains duplicate text keys"
    )

check(
    set(candidate["key"])
    == set(groups["key"])
    == set(manifest["key"]),
    "Candidate, group, and manifest keys differ"
)

check(
    set(manifest["Requirement ID"]).isdisjoint(
        set(quarantine["Requirement ID"])
    ),
    "Quarantined IDs found in manifest"
)

joined = manifest[
    ["key", "Requirement ID", "label", "similarity_group"]
].merge(
    groups[
        ["key", "Requirement ID", "label", "similarity_group"]
    ],
    on="key",
    suffixes=("_manifest", "_groups"),
    validate="one_to_one"
)

check(len(joined) == 4833, "Membership merge incomplete")

for column in [
    "Requirement ID",
    "label",
    "similarity_group"
]:
    check(
        (
            joined[f"{column}_manifest"]
            == joined[f"{column}_groups"]
        ).all(),
        f"Membership differs: {column}"
    )

candidate_check = manifest[
    ["key", "Requirement ID", "label"]
].merge(
    candidate[["key", "Requirement ID", "label"]],
    on="key",
    suffixes=("_manifest", "_candidate"),
    validate="one_to_one"
)

check(
    len(candidate_check) == 4833,
    "Candidate merge incomplete"
)

for column in ["Requirement ID", "label"]:
    check(
        (
            candidate_check[f"{column}_manifest"]
            == candidate_check[f"{column}_candidate"]
        ).all(),
        f"Candidate differs: {column}"
    )

# ----------------------------------------------------------
# 4. Frozen split verification
# ----------------------------------------------------------

check(
    set(manifest["split"])
    == {"train", "validation", "test"},
    "Unexpected split labels"
)

group_crossings = int(
    (
        manifest.groupby("similarity_group")["split"]
        .nunique() > 1
    ).sum()
)

check(group_crossings == 0, "Cross-partition group found")

counts = (
    manifest.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(["train", "validation", "test"])
    .reindex(columns=["FR", "NFR"], fill_value=0)
)

expected = {
    "train": {"FR": 2224, "NFR": 1154},
    "validation": {"FR": 489, "NFR": 239},
    "test": {"FR": 488, "NFR": 239}
}

for partition, labels in expected.items():
    for label, count in labels.items():
        check(
            counts.loc[partition, label] == count,
            f"Unexpected count for {partition}/{label}"
        )
        check(
            split["class_counts"][partition][label] == count,
            f"Metadata mismatch: {partition}/{label}"
        )

# ----------------------------------------------------------
# 5. Exhaustive similarity audit confirmation
# ----------------------------------------------------------

for column in [
    "pairs_ge_0.80",
    "pairs_ge_0.85"
]:
    audit_summary[column] = pd.to_numeric(
        audit_summary[column],
        errors="raise"
    )

pairs_080 = int(audit_summary["pairs_ge_0.80"].sum())
pairs_085 = int(audit_summary["pairs_ge_0.85"].sum())

check(
    pairs_080 == audit["pairs_ge_080"],
    "0.80 audit count mismatch"
)

check(
    pairs_085 == audit["pairs_ge_085"],
    "0.85 audit count mismatch"
)

check(pairs_085 == 0, "Similarity review threshold failed")

# ----------------------------------------------------------
# 6. Research readiness record
# ----------------------------------------------------------

record = {
    "protocol": "D2-v1.1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": {
        "requirements": len(manifest),
        "fr": int((manifest["label"] == "FR").sum()),
        "nfr": int((manifest["label"] == "NFR").sum()),
        "similarity_groups": int(
            manifest["similarity_group"].nunique()
        ),
        "quarantined_v11_records": len(quarantine)
    },
    "split_counts": {
        partition: {
            label: int(counts.loc[partition, label])
            for label in ["FR", "NFR"]
        }
        for partition in counts.index
    },
    "technical_checks": {
        "file_hash_integrity": "PASS",
        "candidate_consistency": "PASS",
        "group_isolation": "PASS",
        "frozen_split_integrity": "PASS",
        "cross_partition_similarity_ge_085": 0,
        "cross_partition_similarity_ge_080": pairs_080
    },
    "research_status": {
        "technical_readiness": "PASS",
        "experimental_protocol": (
            "READY_WITH_DOCUMENTED_LIMITATIONS"
        ),
        "independent_annotation_validation": (
            "NOT_PERFORMED"
        ),
        "project_independence": "NOT_ESTABLISHED",
        "scientific_approval": "CONDITIONAL",
        "model_training": "NOT_STARTED",
        "test_evaluation": "NOT_STARTED"
    },
    "limitations": [
        "Source-label agreement is not independent "
        "semantic verification.",
        "AI-assisted review decisions remain provisional.",
        "Original-project independence is unverified.",
        "Lexical similarity threshold does not establish "
        "semantic independence.",
        "17 cross-partition lexical pairs scored at "
        "least 0.80 but below 0.85 in Script 55.",
        "The frozen test set must not be used for "
        "model or threshold selection."
    ],
    "frozen_manifest_sha256": sha256(MANIFEST)
}

REPORTS.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON.write_text(
    json.dumps(record, indent=2),
    encoding="utf-8"
)

lines = [
    "SCRIPT 56: FINAL D2 v1.1 RESEARCH READINESS",
    "=" * 65,
    "",
    f"Requirements: {len(manifest)}",
    f"FR: {record['dataset']['fr']}",
    f"NFR: {record['dataset']['nfr']}",
    f"Similarity groups: {record['dataset']['similarity_groups']}",
    "",
    "CLASS DISTRIBUTION",
    counts.to_string(),
    "",
    "TECHNICAL VALIDATION",
    "File hash integrity: PASS",
    "Dataset consistency: PASS",
    "Group isolation: PASS",
    "Frozen split integrity: PASS",
    f"Cross-partition pairs >= 0.85: {pairs_085}",
    f"Cross-partition pairs >= 0.80: {pairs_080}",
    "",
    "RESEARCH STATUS",
    "Technical readiness: PASS",
    "Experimental protocol: READY_WITH_DOCUMENTED_LIMITATIONS",
    "Independent human label verification: NOT PERFORMED",
    "Project independence: NOT ESTABLISHED",
    "Scientific approval: CONDITIONAL",
    "Model training: NOT STARTED",
    "Test evaluation: NOT STARTED"
]

OUTPUT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("=" * 76)
print("SCRIPT 56: FINAL D2 v1.1 RESEARCH READINESS")
print("=" * 76)

print("\nCLASS DISTRIBUTION")
print(counts.to_string())

print("\nTECHNICAL VALIDATION")
print("File hash integrity: PASS")
print("Dataset consistency: PASS")
print("Group isolation: PASS")
print("Frozen split integrity: PASS")
print("Cross-partition pairs >= 0.85:", pairs_085)
print("Cross-partition pairs >= 0.80:", pairs_080)

print("\nRESEARCH STATUS")
for name, value in record["research_status"].items():
    print(f"{name}: {value}")

print("\nFILES GENERATED")
print(OUTPUT_JSON)
print(OUTPUT_TXT)

print("\nScript 56 completed.")
