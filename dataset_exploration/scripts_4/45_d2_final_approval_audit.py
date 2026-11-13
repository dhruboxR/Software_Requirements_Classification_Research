
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"

VALIDATED = BASE / "validated"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

CANDIDATES = VALIDATED / "42_d2_source_consistent_candidates.csv"
MEMBERSHIP = TABLES / "43_d2_candidate_group_membership.csv"
MANIFEST = VALIDATED / "44_d2_frozen_split_manifest.csv"
METADATA = VALIDATED / "44_d2_split_metadata.json"
WITHHELD = VALIDATED / "42_d2_withheld_label_cases.csv"

REPORTS.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


candidate = pd.read_csv(
    CANDIDATES, dtype=str, keep_default_na=False
)
membership = pd.read_csv(
    MEMBERSHIP, dtype=str, keep_default_na=False
)
manifest = pd.read_csv(
    MANIFEST, dtype=str, keep_default_na=False
)
withheld = pd.read_csv(
    WITHHELD, dtype=str, keep_default_na=False
)

metadata = json.loads(METADATA.read_text(encoding="utf-8"))

checks = {}


def check(name, condition):
    checks[name] = bool(condition)
    print(f"{name}: {'PASS' if condition else 'FAIL'}")


print("=" * 75)
print("SCRIPT 45: FINAL D2 DATASET APPROVAL AUDIT")
print("=" * 75)

check(
    "Candidate SHA256 unchanged",
    sha256(CANDIDATES) == metadata["candidate_sha256"]
)

check(
    "Membership SHA256 unchanged",
    sha256(MEMBERSHIP) == metadata["membership_sha256"]
)

check(
    "Manifest SHA256 unchanged",
    sha256(MANIFEST) == metadata["manifest_sha256"]
)

check("Candidate count", len(candidate) == 4835)
check("Manifest count", len(manifest) == 4835)
check("Unique candidate texts", candidate["key"].is_unique)
check("Unique manifest texts", manifest["key"].is_unique)

check(
    "Same candidate and manifest keys",
    set(candidate["key"]) == set(manifest["key"])
)

check(
    "No withheld-text overlap",
    set(candidate["key"]).isdisjoint(set(withheld["key"]))
)

check(
    "Only FR/NFR labels",
    set(manifest["label"]) == {"FR", "NFR"}
)

check(
    "Expected partition names",
    set(manifest["split"]) == {
        "train", "validation", "test"
    }
)

group_counts = (
    manifest.groupby("similarity_group")["split"].nunique()
)

check(
    "No similarity group crosses partitions",
    bool((group_counts == 1).all())
)

counts = (
    manifest.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(["train", "validation", "test"])
)

expected = {
    "train": {"FR": 2256, "NFR": 1136},
    "validation": {"FR": 472, "NFR": 251},
    "test": {"FR": 474, "NFR": 246},
}

check(
    "Expected class counts",
    all(
        int(counts.loc[split, label]) == count
        for split, labels in expected.items()
        for label, count in labels.items()
    )
)

check(
    "Recorded metadata class counts",
    all(
        int(counts.loc[split, label])
        == metadata["class_counts"][split][label]
        for split in expected
        for label in ["FR", "NFR"]
    )
)

# Verify each frozen label and group against
# the corresponding original candidate records.
joined = manifest[[
    "key", "label", "similarity_group"
]].merge(
    candidate[["key", "label"]],
    on="key",
    how="left",
    suffixes=("_manifest", "_candidate"),
    validate="one_to_one"
)

check(
    "Candidate labels unchanged",
    bool(
        (joined["label_manifest"]
         == joined["label_candidate"]).all()
    )
)

joined_groups = manifest[[
    "key", "similarity_group"
]].merge(
    membership[["key", "similarity_group"]],
    on="key",
    how="left",
    suffixes=("_manifest", "_original"),
    validate="one_to_one"
)

check(
    "Similarity group assignments unchanged",
    bool(
        (joined_groups["similarity_group_manifest"]
         == joined_groups["similarity_group_original"]).all()
    )
)

passed = sum(checks.values())
total = len(checks)
all_pass = passed == total

print("\n" + "=" * 75)
print("AUDIT SUMMARY")
print("=" * 75)

print(f"Checks passed: {passed}/{total}")
print("Total requirements:", len(manifest))
print("Similarity groups:", len(group_counts))
print("\nCLASS DISTRIBUTION")
print(counts.to_string())

print(
    "\nTechnical audit:",
    "PASS" if all_pass else "FAIL"
)

print(
    "Research approval:",
    "PENDING DOCUMENTED APPROVAL"
    if all_pass else "BLOCKED"
)

report = [
    "FINAL D2 DATASET TECHNICAL AUDIT",
    "=" * 65,
    "",
    *[
        f"{name}: {'PASS' if result else 'FAIL'}"
        for name, result in checks.items()
    ],
    "",
    f"Checks passed: {passed}/{total}",
    "",
    counts.to_string(),
    "",
    "Remaining scientific limitations:",
    "- Original project independence is unverified.",
    "- Semantic label correctness is not fully established.",
    "- Undetected paraphrase leakage remains possible.",
    "- Test set must remain sealed during development.",
    "- Technical PASS does not imply scientific approval.",
]

path = REPORTS / "45_d2_final_approval_audit.txt"
path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSaved:", path)

if not all_pass:
    raise SystemExit("Audit failed. Do not approve the dataset.")
