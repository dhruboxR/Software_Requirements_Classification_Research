
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

# ============================================================
# SCRIPT 53: PREPARE D2 PROTOCOL v1.1 CANDIDATES
#
# 1. Preserve v1.0
# 2. Quarantine two conflicting requirements
# 3. Connect two additional similarity pairs
# 4. Rebuild deterministic group identifiers
# 5. Save new v1.1 candidate files
#
# No train/validation/test partition is created.
# AI-assisted decisions remain provisional.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

TABLES = BASE / "tables"
VALIDATED = BASE / "validated"
REPORTS = BASE / "reports"

for folder in (TABLES, VALIDATED, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

CANDIDATES_V10 = (
    VALIDATED / "42_d2_source_consistent_candidates.csv"
)

GROUPS_V10 = (
    TABLES / "43_d2_candidate_group_membership.csv"
)

MANIFEST_V10 = (
    VALIDATED / "44_d2_frozen_split_manifest.csv"
)

DECISIONS = (
    TABLES / "52_d2_ai_assisted_review_decisions.csv"
)

VALIDATION_METADATA = (
    REPORTS / "52_d2_review_validation.json"
)

CANDIDATES_V11 = (
    VALIDATED / "53_d2_v11_candidates.csv"
)

GROUPS_V11 = (
    TABLES / "53_d2_v11_group_membership.csv"
)

QUARANTINE_V11 = (
    VALIDATED / "53_d2_v11_quarantine_audit.csv"
)

METADATA_V11 = (
    REPORTS / "53_d2_v11_preparation_metadata.json"
)

REPORT_V11 = (
    REPORTS / "53_d2_v11_preparation_report.txt"
)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


print("=" * 76)
print("SCRIPT 53: D2 PROTOCOL v1.1 CANDIDATE PREPARATION")
print("=" * 76)

# ------------------------------------------------------------
# 1. Prevent accidental overwrite
# ------------------------------------------------------------

outputs = [
    CANDIDATES_V11,
    GROUPS_V11,
    QUARANTINE_V11,
    METADATA_V11,
    REPORT_V11
]

existing = [str(p) for p in outputs if p.exists()]

if existing:
    raise FileExistsError(
        "Version 1.1 output already exists. "
        "Existing files will not be overwritten:\n"
        + "\n".join(existing)
    )

# ------------------------------------------------------------
# 2. Load and verify source evidence
# ------------------------------------------------------------

candidate = read_csv(CANDIDATES_V10)
membership = read_csv(GROUPS_V10)
manifest = read_csv(MANIFEST_V10)
decisions = read_csv(DECISIONS)

validation = json.loads(
    VALIDATION_METADATA.read_text(encoding="utf-8")
)

if validation.get("technical_validation") != "PASS":
    raise RuntimeError(
        "Script 52 technical validation has not passed"
    )

if validation.get("review_method") != (
    "AI_ASSISTED_RULE_BASED"
):
    raise RuntimeError("Unexpected decision method")

expected_hashes = validation.get("input_hashes", {})

if sha256(MANIFEST_V10) != expected_hashes.get(
    MANIFEST_V10.name
):
    raise RuntimeError(
        "Version 1.0 manifest differs from Script 52 input"
    )

for path in [CANDIDATES_V10, GROUPS_V10]:
    if not path.exists():
        raise FileNotFoundError(path)

required_candidate = {
    "Requirement ID", "key", "label"
}

required_groups = {
    "Requirement ID",
    "key",
    "similarity_group"
}

required_decisions = {
    "pair_id",
    "requirement_id_a",
    "requirement_id_b",
    "proposed_action",
    "decision_status",
    "review_method"
}

for name, frame, columns in [
    ("Candidate", candidate, required_candidate),
    ("Membership", membership, required_groups),
    ("Decisions", decisions, required_decisions)
]:
    missing = columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )

if len(candidate) != 4835:
    raise ValueError("Unexpected v1.0 candidate count")

if not candidate["Requirement ID"].is_unique:
    raise ValueError("Duplicate candidate Requirement IDs")

if not candidate["key"].is_unique:
    raise ValueError("Duplicate candidate text keys")

if not membership["key"].is_unique:
    raise ValueError("Duplicate membership text keys")

if set(candidate["key"]) != set(membership["key"]):
    raise ValueError("v1.0 group membership differs")

if set(candidate["key"]) != set(manifest["key"]):
    raise ValueError("v1.0 manifest differs")

if len(decisions) != 3:
    raise ValueError("Expected three decisions")

if set(decisions["pair_id"]) != {
    "CP_001", "CP_002", "CP_003"
}:
    raise ValueError("Unexpected review pair IDs")

if not (
    decisions["decision_status"] == "PROVISIONAL"
).all():
    raise ValueError("Unexpected decision status")

if not (
    decisions["review_method"]
    == "AI_ASSISTED_RULE_BASED"
).all():
    raise ValueError("Unexpected review method")

expected_actions = {
    "CP_001": "QUARANTINE_BOTH",
    "CP_002": "CONNECT_SIMILARITY_GROUP",
    "CP_003": "CONNECT_SIMILARITY_GROUP"
}

for _, row in decisions.iterrows():
    if row["proposed_action"] != expected_actions[
        row["pair_id"]
    ]:
        raise ValueError(
            f"Unexpected action for {row['pair_id']}"
        )

# Check decisions refer to the exact reviewed IDs.
expected_ids = {
    "CP_001": {"RQP0494680", "RQP0494674"},
    "CP_002": {"RQP0482022", "RQP0481194"},
    "CP_003": {"RQP0495693", "RQP0495076"}
}

for _, row in decisions.iterrows():
    actual = {
        row["requirement_id_a"],
        row["requirement_id_b"]
    }

    if actual != expected_ids[row["pair_id"]]:
        raise ValueError(
            f"Unexpected IDs in {row['pair_id']}"
        )

# Verify original membership assignments.
merged = candidate.merge(
    membership[["key", "similarity_group"]],
    on="key",
    how="left",
    validate="one_to_one",
    indicator=True
)

if not (merged["_merge"] == "both").all():
    raise RuntimeError("Unmatched v1.0 groups")

merged = merged.drop(columns=["_merge"])

original_groups = (
    merged.groupby("similarity_group")["key"].count()
)

original_group_count = len(original_groups)

# ------------------------------------------------------------
# 3. Apply provisional quarantine
# ------------------------------------------------------------

quarantine_ids = set()
connect_pairs = []

for _, row in decisions.iterrows():
    a = row["requirement_id_a"]
    b = row["requirement_id_b"]

    if row["proposed_action"] == "QUARANTINE_BOTH":
        quarantine_ids.update([a, b])

    elif row["proposed_action"] == (
        "CONNECT_SIMILARITY_GROUP"
    ):
        connect_pairs.append((a, b))

if len(quarantine_ids) != 2:
    raise RuntimeError("Unexpected quarantine count")

quarantined = merged[
    merged["Requirement ID"].isin(quarantine_ids)
].copy()

if len(quarantined) != 2:
    raise RuntimeError(
        "Could not locate both quarantine records"
    )

quarantined["quarantine_reason"] = (
    "AI_ASSISTED_PROVISIONAL_LABEL_CONFLICT"
)

quarantined["quarantine_protocol"] = "v1.1"

v11 = merged[
    ~merged["Requirement ID"].isin(quarantine_ids)
].copy()

v11 = v11.sort_values("key").reset_index(drop=True)

if len(v11) != 4833:
    raise RuntimeError("Unexpected v1.1 candidate count")

if set(v11["Requirement ID"]) & quarantine_ids:
    raise RuntimeError("Quarantine failed")

# ------------------------------------------------------------
# 4. Rebuild similarity groups
# ------------------------------------------------------------

uf = UnionFind(len(v11))

# Preserve every original v1.0 similarity connection
# among the remaining records.
group_first = {}

for i, group in enumerate(v11["similarity_group"]):
    if group in group_first:
        uf.union(i, group_first[group])
    else:
        group_first[group] = i

id_to_index = {
    rid: i
    for i, rid in enumerate(v11["Requirement ID"])
}

connections_applied = 0

for a, b in connect_pairs:
    if a not in id_to_index or b not in id_to_index:
        raise RuntimeError(
            f"Missing connection endpoint: {a}, {b}"
        )

    uf.union(id_to_index[a], id_to_index[b])
    connections_applied += 1

# Deterministic component identifiers.
# Stable for identical v1.1 inputs.
roots = [uf.find(i) for i in range(len(v11))]

component_members = {}

for i, root in enumerate(roots):
    component_members.setdefault(root, []).append(i)

ordered_components = sorted(
    component_members.values(),
    key=lambda indices: min(
        v11.loc[i, "key"] for i in indices
    )
)

new_group_ids = {}

for group_number, indices in enumerate(
    ordered_components, start=1
):
    group_id = f"D2V11_G{group_number:05d}"

    for i in indices:
        new_group_ids[i] = group_id

v11["similarity_group_v11"] = [
    new_group_ids[i] for i in range(len(v11))
]

group_count = v11["similarity_group_v11"].nunique()

largest_group = int(
    v11.groupby("similarity_group_v11").size().max()
)

mixed_label_groups = int(
    (
        v11.groupby("similarity_group_v11")[
            "label"
        ].nunique() > 1
    ).sum()
)

# Check original groups were never split.
old_group_violations = int(
    (
        v11.groupby("similarity_group")[
            "similarity_group_v11"
        ].nunique() > 1
    ).sum()
)

if old_group_violations:
    raise RuntimeError(
        "An original similarity group was split"
    )

# Check both new connections.
for a, b in connect_pairs:
    group_a = v11.loc[
        id_to_index[a], "similarity_group_v11"
    ]

    group_b = v11.loc[
        id_to_index[b], "similarity_group_v11"
    ]

    if group_a != group_b:
        raise RuntimeError(
            f"Connection not preserved: {a}, {b}"
        )

if mixed_label_groups:
    raise RuntimeError(
        "Mixed-label similarity groups detected; "
        "manual investigation needed before saving"
    )

# ------------------------------------------------------------
# 5. Prepare separate output files
# ------------------------------------------------------------

candidate_output = v11.drop(
    columns=[
        "similarity_group",
        "similarity_group_v11",
        "split"
    ],
    errors="ignore"
)

group_output = v11[[
    "Requirement ID",
    "key",
    "label",
    "similarity_group_v11"
]].copy()

group_output = group_output.rename(
    columns={
        "similarity_group_v11": "similarity_group"
    }
)

# Never retain old partition assignments.
if "split" in candidate_output.columns:
    raise RuntimeError(
        "Old split assignments unexpectedly present"
    )

if "split" in group_output.columns:
    raise RuntimeError(
        "Old split assignments unexpectedly present"
    )

# ------------------------------------------------------------
# 6. Print verification
# ------------------------------------------------------------

print("\nDATASET REVISION")
print("v1.0 requirements:", len(candidate))
print("Quarantined requirements:", len(quarantined))
print("v1.1 requirements:", len(candidate_output))

print("\nCLASS DISTRIBUTION")
print(
    candidate_output["label"]
    .value_counts()
    .to_string()
)

print("\nSIMILARITY GROUPS")
print("Original v1.0 groups:", original_group_count)
print("New v1.1 groups:", group_count)
print("Largest group:", largest_group)
print("Additional connections applied:", connections_applied)
print("Mixed-label groups:", mixed_label_groups)
print("Original groups broken:", old_group_violations)

print("\nINTEGRITY")
print(
    "Unique v1.1 texts:",
    candidate_output["key"].is_unique
)
print(
    "Unique requirement IDs:",
    candidate_output["Requirement ID"].is_unique
)
print(
    "Quarantined IDs absent:",
    not bool(
        set(candidate_output["Requirement ID"])
        & quarantine_ids
    )
)
print(
    "New split generated:",
    "NO"
)

# ------------------------------------------------------------
# 7. Save v1.1 artifacts
# ------------------------------------------------------------

candidate_output.to_csv(
    CANDIDATES_V11,
    index=False,
    encoding="utf-8-sig"
)

group_output.to_csv(
    GROUPS_V11,
    index=False,
    encoding="utf-8-sig"
)

quarantined.to_csv(
    QUARANTINE_V11,
    index=False,
    encoding="utf-8-sig"
)

metadata = {
    "protocol": "D2-v1.1",
    "status": "CANDIDATE_PREPARED_NOT_APPROVED",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "review_method": "AI_ASSISTED_RULE_BASED",
    "decision_status": "PROVISIONAL",
    "independent_human_review": False,
    "source_protocol": "v1.0",
    "source_requirements": len(candidate),
    "quarantined_requirements": len(quarantined),
    "candidate_requirements": len(candidate_output),
    "class_counts": {
        label: int(count)
        for label, count in candidate_output[
            "label"
        ].value_counts().items()
    },
    "original_groups": original_group_count,
    "v11_groups": group_count,
    "largest_group": largest_group,
    "mixed_label_groups": mixed_label_groups,
    "additional_connections": connections_applied,
    "new_partition_generated": False,
    "input_sha256": {
        path.name: sha256(path)
        for path in [
            CANDIDATES_V10,
            GROUPS_V10,
            MANIFEST_V10,
            DECISIONS,
            VALIDATION_METADATA
        ]
    },
    "output_sha256": {
        path.name: sha256(path)
        for path in [
            CANDIDATES_V11,
            GROUPS_V11,
            QUARANTINE_V11
        ]
    },
    "limitations": [
        "AI-assisted decisions are provisional.",
        "No independent human annotations were obtained.",
        "No classifier has been trained.",
        "No new train/validation/test split exists yet.",
        "Existing similarity detection may be incomplete.",
        "Original-project independence is unverified."
    ]
}

METADATA_V11.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

report_lines = [
    "D2 PROTOCOL v1.1 CANDIDATE PREPARATION",
    "=" * 65,
    f"v1.0 requirements: {len(candidate)}",
    f"Quarantined: {len(quarantined)}",
    f"v1.1 requirements: {len(candidate_output)}",
    "",
    candidate_output["label"].value_counts().to_string(),
    "",
    f"v1.0 groups: {original_group_count}",
    f"v1.1 groups: {group_count}",
    f"Largest group: {largest_group}",
    f"Mixed-label groups: {mixed_label_groups}",
    f"Additional connections: {connections_applied}",
    "",
    "Review method: AI_ASSISTED_RULE_BASED",
    "Independent review: NO",
    "Scientific approval: PENDING",
    "New split generated: NO",
    "Original v1.0 modified: NO"
]

REPORT_V11.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Protocol: v1.1")
print("Candidate preparation: PASS")
print("Scientific approval: PENDING")
print("New split: NOT GENERATED")
print("Version 1.0 modified: NO")
print("Model training: NOT STARTED")

print("\nFILES GENERATED")
print(CANDIDATES_V11)
print(GROUPS_V11)
print(QUARANTINE_V11)
print(METADATA_V11)
print(REPORT_V11)

print("\nScript 53 completed.")
