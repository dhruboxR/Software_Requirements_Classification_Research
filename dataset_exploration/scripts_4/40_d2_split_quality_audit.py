
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset_exploration/outputs"
TABLES = BASE / "tables"
REPORTS = BASE / "reports"

TRIALS_FILE = TABLES / "39_d2_split_trial_counts.csv"
GROUPS_FILE = TABLES / "39_d2_group_structure.csv"

TRIALS = pd.read_csv(TRIALS_FILE)
GROUPS = pd.read_csv(GROUPS_FILE)

EXPECTED_TOTAL = 4871

TARGETS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

TOLERANCE = 0.05

print("=" * 78)
print("SCRIPT 40: D2 SPLIT QUALITY AND STABILITY AUDIT")
print("=" * 78)

records = []

for scenario, part in TRIALS.groupby("scenario"):
    part = part.copy()

    for split, target in TARGETS.items():
        part[f"{split}_fraction"] = (
            part[f"{split}_size"] / EXPECTED_TOTAL
        )

        part[f"{split}_ratio_ok"] = (
            (part[f"{split}_fraction"] - target).abs()
            <= TOLERANCE
        )

    part["total_rows"] = (
        part["train_size"]
        + part["val_size"]
        + part["test_size"]
    )

    if not (part["total_rows"] == EXPECTED_TOTAL).all():
        raise ValueError(
            f"{scenario}: split sizes do not total "
            f"{EXPECTED_TOTAL}"
        )

    for split in TARGETS:
        class_sum = (
            part[f"{split}_FR"]
            + part[f"{split}_NFR"]
        )
        if not (class_sum == part[f"{split}_size"]).all():
            raise ValueError(
                f"{scenario}: class counts do not "
                f"match {split} sizes"
            )

    part["all_ratios_ok"] = (
        part["train_ratio_ok"]
        & part["val_ratio_ok"]
        & part["test_ratio_ok"]
    )

    part["all_classes_present"] = (
        (part["train_FR"] > 0)
        & (part["train_NFR"] > 0)
        & (part["val_FR"] > 0)
        & (part["val_NFR"] > 0)
        & (part["test_FR"] > 0)
        & (part["test_NFR"] > 0)
    )

    part["nfr_test_at_least_100"] = (
        part["test_NFR"] >= 100
    )

    part["quality_screen_pass"] = (
        part["all_ratios_ok"]
        & part["all_classes_present"]
        & part["nfr_test_at_least_100"]
    )

    records.append({
        "scenario": scenario,
        "trials": len(part),
        "train_fraction_min": round(
            part["train_fraction"].min(), 4
        ),
        "train_fraction_median": round(
            part["train_fraction"].median(), 4
        ),
        "train_fraction_max": round(
            part["train_fraction"].max(), 4
        ),
        "test_fraction_min": round(
            part["test_fraction"].min(), 4
        ),
        "test_fraction_median": round(
            part["test_fraction"].median(), 4
        ),
        "test_fraction_max": round(
            part["test_fraction"].max(), 4
        ),
        "all_ratios_ok_trials": int(
            part["all_ratios_ok"].sum()
        ),
        "all_classes_present_trials": int(
            part["all_classes_present"].sum()
        ),
        "nfr_test_at_least_100_trials": int(
            part["nfr_test_at_least_100"].sum()
        ),
        "quality_screen_pass_trials": int(
            part["quality_screen_pass"].sum()
        ),
    })

summary = pd.DataFrame(records)

out = TABLES / "40_d2_split_quality_summary.csv"
summary.to_csv(out, index=False)

print("\nSPLIT QUALITY SUMMARY")
print(summary.to_string(index=False))

print("\nGROUP STRUCTURE")
print(GROUPS.to_string(index=False))

print("\nINTERPRETATION")
print(
    "Ratio tolerance: +/- 5 percentage points "
    "around 70/15/15."
)
print(
    "Quality screen also requires both classes "
    "in every partition and at least 100 NFR "
    "test examples."
)
print(
    "This is a preliminary feasibility screen, "
    "not a final split-selection procedure."
)

report = REPORTS / "40_d2_split_quality_audit.txt"
report.write_text(
    "D2 SPLIT QUALITY AUDIT\n"
    + "=" * 65 + "\n\n"
    + summary.to_string(index=False)
    + "\n\nGROUP STRUCTURE\n"
    + GROUPS.to_string(index=False)
    + "\n\nLimitations:\n"
    + "- Project independence is not verified.\n"
    + "- Near-duplicate detection may be incomplete.\n"
    + "- Preliminary annotation decisions remain pending.\n"
    + "- Successful ratio screening does not prove generalization.\n"
    + "- No final split or model was created.\n",
    encoding="utf-8"
)

print("\nSaved:", out)
print("Saved:", report)
print("Script 40 completed.")
