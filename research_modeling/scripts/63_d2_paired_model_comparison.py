
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score

# ==========================================================
# SCRIPT 63: PAIRED BASELINE MODEL COMPARISON
#
# Compares B02, S01 and S02 on existing validation
# predictions. Uses paired group-level bootstrapping
# and exact McNemar tests.
#
# No training, threshold search or test-set evaluation.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset_exploration/outputs/validated"
OUT = ROOT / "research_modeling/outputs"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
META = DATA / "54_d2_v11_split_metadata.json"

LR_FILE = TABLES / "57_d2_validation_predictions.csv"
SVM_FILE = TABLES / "62_d2_svm_validation_predictions.csv"

B02 = "B02_BalancedLogisticRegression"
S01 = "S01_LinearSVM"
S02 = "S02_BalancedLinearSVM"

SEED = 42
BOOTSTRAPS = 2000


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def scores(y, p):
    return {
        "nfr_f1": f1_score(y, p, zero_division=0),
        "nfr_recall": recall_score(y, p, zero_division=0),
        "macro_f1": f1_score(
            y, p, average="macro", zero_division=0
        ),
    }


def load_predictions(path, model, prefix):
    df = pd.read_csv(
        path, dtype={"requirement_id": str}
    )
    df = df[df["model"] == model].copy()

    if len(df) != 728 or not df["requirement_id"].is_unique:
        raise RuntimeError(f"Invalid predictions: {model}")

    if set(df["true_label"]) != {"FR", "NFR"}:
        raise RuntimeError(f"Invalid labels: {model}")

    return df[
        ["requirement_id", "true_label", "predicted_label"]
    ].rename(columns={
        "true_label": f"true_{prefix}",
        "predicted_label": f"pred_{prefix}",
    })


# ----------------------------------------------------------
# 1. Integrity and validation-only manifest
# ----------------------------------------------------------

meta = json.loads(META.read_text(encoding="utf-8"))
expected = meta["output_sha256"][MANIFEST.name]

if meta["protocol"] != "D2-v1.1" or sha256(MANIFEST) != expected:
    raise RuntimeError("Frozen manifest integrity failed")

parts = []
for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    parts.append(
        chunk.loc[
            chunk["split"] == "validation",
            ["Requirement ID", "label", "similarity_group"]
        ].copy()
    )

val = pd.concat(parts, ignore_index=True)
val = val.rename(columns={
    "Requirement ID": "requirement_id",
    "label": "actual_label",
})

if len(val) != 728 or not val["requirement_id"].is_unique:
    raise RuntimeError("Invalid validation manifest")

# ----------------------------------------------------------
# 2. Align the three models by requirement ID
# ----------------------------------------------------------

merged = val.copy()

for path, name, prefix in [
    (LR_FILE, B02, "b02"),
    (SVM_FILE, S01, "s01"),
    (SVM_FILE, S02, "s02"),
]:
    pred = load_predictions(path, name, prefix)

    merged = merged.merge(
        pred,
        on="requirement_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 728:
        raise RuntimeError(f"Incomplete merge: {name}")

    if not (
        merged[f"true_{prefix}"] == merged["actual_label"]
    ).all():
        raise RuntimeError(f"True-label mismatch: {name}")

    if not merged[f"pred_{prefix}"].isin(["FR", "NFR"]).all():
        raise RuntimeError(f"Prediction-label error: {name}")

merged["y"] = (merged["actual_label"] == "NFR").astype(int)

for prefix in ("b02", "s01", "s02"):
    merged[f"p_{prefix}"] = (
        merged[f"pred_{prefix}"] == "NFR"
    ).astype(int)

merged = merged.sort_values(
    "requirement_id"
).reset_index(drop=True)

y = merged["y"].to_numpy()
predictions = {
    k: merged[f"p_{k}"].to_numpy()
    for k in ("b02", "s01", "s02")
}

print("=" * 76)
print("SCRIPT 63: PAIRED BASELINE MODEL COMPARISON")
print("=" * 76)

print("\nDATASET")
print("Validation requirements:", len(merged))
print("Validation similarity groups:", merged["similarity_group"].nunique())
print("NFR requirements:", int(y.sum()))
print("Frozen manifest integrity: PASS")
print("Test data accessed: NO")

print("\nINDIVIDUAL MODEL RESULTS")

for name, pred in predictions.items():
    print(name, scores(y, pred))

# ----------------------------------------------------------
# 3. Bootstrap entire similarity groups
# ----------------------------------------------------------

rng = np.random.default_rng(SEED)

groups = merged["similarity_group"].to_numpy()
unique_groups = np.unique(groups)

group_indices = [
    np.flatnonzero(groups == group)
    for group in unique_groups
]

comparisons = [
    ("s02", "b02"),
    ("s01", "b02"),
    ("s02", "s01"),
]

records = []
bootstrap_records = []

for a, b in comparisons:
    result_a = scores(y, predictions[a])
    result_b = scores(y, predictions[b])

    correct_a = predictions[a] == y
    correct_b = predictions[b] == y

    a_only_correct = int(
        np.sum(correct_a & ~correct_b)
    )
    b_only_correct = int(
        np.sum(~correct_a & correct_b)
    )

    discordant = a_only_correct + b_only_correct

    # Two-sided exact McNemar test, via binomial test.
    pvalue = (
        binomtest(
            min(a_only_correct, b_only_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant else 1.0
    )

    differences = {
        key: []
        for key in ("nfr_f1", "nfr_recall", "macro_f1")
    }

    for iteration in range(BOOTSTRAPS):
        sampled_groups = rng.integers(
            0, len(unique_groups),
            len(unique_groups),
        )

        ix = np.concatenate(
            [group_indices[j] for j in sampled_groups]
        )

        # Skip invalid resamples with only one class.
        if len(np.unique(y[ix])) != 2:
            continue

        sa = scores(y[ix], predictions[a][ix])
        sb = scores(y[ix], predictions[b][ix])

        for metric in differences:
            differences[metric].append(
                sa[metric] - sb[metric]
            )

    record = {
        "model_a": a,
        "model_b": b,
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "mcnemar_exact_p": pvalue,
    }

    for metric in differences:
        values = np.asarray(differences[metric])

        if len(values) < 100:
            raise RuntimeError("Too few valid bootstrap samples")

        lo, hi = np.quantile(
            values, [0.025, 0.975]
        )

        record[f"{metric}_difference"] = (
            result_a[metric] - result_b[metric]
        )
        record[f"{metric}_ci_low"] = float(lo)
        record[f"{metric}_ci_high"] = float(hi)

    records.append(record)

    print(f"\n{a.upper()} VS {b.upper()}")
    print("A only correct:", a_only_correct)
    print("B only correct:", b_only_correct)
    print("McNemar exact p:", round(pvalue, 6))

    for metric in differences:
        print(
            metric,
            "difference:",
            round(record[f"{metric}_difference"], 4),
            "95% bootstrap CI:",
            (
                round(record[f"{metric}_ci_low"], 4),
                round(record[f"{metric}_ci_high"], 4),
            ),
        )

# ----------------------------------------------------------
# 4. Identify shared and corrected NFR errors
# ----------------------------------------------------------

nfr = merged[merged["y"] == 1].copy()

error_rows = []

for _, row in nfr.iterrows():
    error_rows.append({
        "requirement_id": row["requirement_id"],
        "similarity_group": row["similarity_group"],
        "b02_missed": row["p_b02"] == 0,
        "s01_missed": row["p_s01"] == 0,
        "s02_missed": row["p_s02"] == 0,
        "s02_fixes_b02": (
            row["p_b02"] == 0
            and row["p_s02"] == 1
        ),
        "s02_new_error_vs_b02": (
            row["p_b02"] == 1
            and row["p_s02"] == 0
        ),
    })

error_df = pd.DataFrame(error_rows)

print("\nNFR ERROR OVERLAP")
print("B02 NFR misses:", int(error_df["b02_missed"].sum()))
print("S01 NFR misses:", int(error_df["s01_missed"].sum()))
print("S02 NFR misses:", int(error_df["s02_missed"].sum()))
print("S02 fixes B02 misses:", int(error_df["s02_fixes_b02"].sum()))
print(
    "S02 introduces NFR misses vs B02:",
    int(error_df["s02_new_error_vs_b02"].sum())
)
print(
    "Missed by all three:",
    int(
        (
            error_df["b02_missed"]
            & error_df["s01_missed"]
            & error_df["s02_missed"]
        ).sum()
    )
)

# ----------------------------------------------------------
# 5. Save outputs
# ----------------------------------------------------------

comparison = pd.DataFrame(records)

comparison.to_csv(
    TABLES / "63_d2_paired_comparison.csv",
    index=False,
)

error_df.to_csv(
    TABLES / "63_d2_nfr_error_overlap.csv",
    index=False,
)

report = [
    "SCRIPT 63: PAIRED BASELINE COMPARISON",
    "=" * 65,
    "",
    "Partition: validation only",
    f"Requirements: {len(merged)}",
    f"Similarity groups: {len(unique_groups)}",
    f"Bootstrap resamples: {BOOTSTRAPS}",
    "",
    "PAIRED MODEL RESULTS",
    comparison.round(5).to_string(index=False),
    "",
    "NFR ERROR OVERLAP",
    f"S02 fixes B02 misses: {error_df['s02_fixes_b02'].sum()}",
    f"S02 new NFR misses: {error_df['s02_new_error_vs_b02'].sum()}",
    "",
    "LIMITATIONS",
    "- Exploratory comparison on a reused validation partition.",
    "- Bootstrap resampling uses similarity groups.",
    "- Bootstrap intervals do not adjust for model selection.",
    "- McNemar tests concern total classification error.",
    "- Multiple comparisons were made without adjustment.",
    "- No external or independent-project evaluation.",
    "- Frozen test partition remains sealed.",
]

report_path = REPORTS / "63_d2_paired_comparison.txt"
report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

metadata_path = REPORTS / "63_d2_paired_comparison.json"
metadata_path.write_text(
    json.dumps({
        "experiment": 63,
        "protocol": "D2-v1.1",
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "method": "similarity_group_bootstrap",
        "mcnemar": "exact_two_sided",
        "manifest_sha256": sha256(MANIFEST),
        "validation_previously_used_for_selection": True,
        "test_used": False,
        "training": False,
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Paired model comparison: COMPLETED")
print("Group-level bootstrap: COMPLETED")
print("Test set: SEALED")
print("Model selection: PENDING")

print("\nFILES GENERATED")
print(TABLES / "63_d2_paired_comparison.csv")
print(TABLES / "63_d2_nfr_error_overlap.csv")
print(report_path)
print(metadata_path)
print("\nScript 63 completed.")
