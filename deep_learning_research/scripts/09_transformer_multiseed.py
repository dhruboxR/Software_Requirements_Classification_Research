
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import pandas as pd

# ==========================================================
# SCRIPT 09: MULTI-SEED TRANSFORMER STABILITY
#
# Reuses existing experiment implementations.
# Historical seed: 42
# New seeds: 17, 29
#
# Generates isolated scripts and filenames.
# Preserves existing checkpoints.
# Never accesses test data.
# ==========================================================

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"
LOGS = PROJECT / "logs"

for folder in (TABLES, REPORTS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

MODELS = {
    "distilbert": {
        "source": "05_distilbert_baseline.py",
        "prefix": "05_distilbert_",
        "historical": "05_distilbert_validation_metrics.csv",
        "experiment": "D01_DistilBERT",
    },
    "bert": {
        "source": "06_bert_baseline.py",
        "prefix": "06_bert_",
        "historical": "06_bert_validation_metrics.csv",
        "experiment": "D02_BERTBase",
    },
    "roberta": {
        "source": "07_roberta_baseline.py",
        "prefix": "07_roberta_",
        "historical": "07_roberta_validation_metrics.csv",
        "experiment": "D03_RoBERTaBase",
    },
}

SEEDS = [42, 17, 29]
NEW_SEEDS = [17, 29]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prepare-only",
    action="store_true",
    help="Generate scripts without training",
)
parser.add_argument(
    "--models",
    nargs="+",
    choices=list(MODELS),
    default=list(MODELS),
)
args = parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1048576), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def prepare_script(name, seed):
    info = MODELS[name]
    source = SCRIPTS / info["source"]

    if not source.is_file():
        raise FileNotFoundError(source)

    code = source.read_text(encoding="utf-8")

    if code.count("SEED = 42") != 1:
        raise RuntimeError(
            f"Unexpected seed definition: {source}"
        )

    code = code.replace(
        "SEED = 42",
        f"SEED = {seed}",
    )

    old_prefix = info["prefix"]
    new_prefix = f"09_{name}_seed{seed}_"

    if old_prefix not in code:
        raise RuntimeError(
            f"Output prefix missing: {old_prefix}"
        )

    code = code.replace(old_prefix, new_prefix)

    old_experiment = (
        f'EXPERIMENT = "{info["experiment"]}"'
    )

    new_experiment = (
        f'EXPERIMENT = "{info["experiment"]}_seed{seed}"'
    )

    if code.count(old_experiment) != 1:
        raise RuntimeError(
            f"Experiment definition missing: {name}"
        )

    code = code.replace(
        old_experiment,
        new_experiment,
    )

    target = (
        SCRIPTS / f"09_{name}_seed{seed}.py"
    )

    compile(code, str(target), "exec")

    if target.exists():
        if target.read_text(encoding="utf-8") != code:
            raise RuntimeError(
                f"Generated script differs: {target}"
            )
    else:
        target.write_text(code, encoding="utf-8")

    metrics_path = (
        TABLES
        / f"09_{name}_seed{seed}_validation_metrics.csv"
    )

    return target, metrics_path


def run_training(name, seed, script, metrics_path):
    if metrics_path.exists():
        print(
            f"Existing metrics found: {name}, seed={seed}"
        )
        return

    log_path = (
        LOGS / f"09_{name}_seed{seed}.log"
    )

    print(
        f"\nRunning {name}, seed={seed}",
        flush=True,
    )
    print("Log:", log_path, flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"{name}, seed={seed} failed. "
            f"Inspect {log_path}"
        )

    if not metrics_path.is_file():
        raise RuntimeError(
            f"Training ended without metrics: {metrics_path}"
        )

    print(
        f"Completed {name}, seed={seed}",
        flush=True,
    )


print("=" * 76)
print("SCRIPT 09: TRANSFORMER MULTI-SEED STABILITY")
print("=" * 76)

print("\nCONFIGURATION")
print("Existing seed: 42")
print("Additional seeds:", NEW_SEEDS)
print("Selected models:", args.models)
print("Frozen test evaluation: NOT PERFORMED")

prepared = {}

for name in args.models:
    info = MODELS[name]

    source_path = SCRIPTS / info["source"]

    if not (
        TABLES / info["historical"]
    ).is_file():
        raise FileNotFoundError(
            "Historical result missing: "
            + info["historical"]
        )

    for seed in NEW_SEEDS:
        script, metrics_path = prepare_script(
            name, seed
        )

        prepared[(name, seed)] = (
            script, metrics_path
        )

        print(
            f"Prepared: {script.name}"
        )

if args.prepare_only:
    print("\nPreparation completed.")
    print("No training performed.")
    sys.exit(0)

# ----------------------------------------------------------
# Run each experiment in a separate Python process
# ----------------------------------------------------------

for (name, seed), (script, path) in prepared.items():
    run_training(name, seed, script, path)

# ----------------------------------------------------------
# Aggregate completed seeds
# ----------------------------------------------------------

rows = []

for name in args.models:
    info = MODELS[name]

    for seed in SEEDS:
        if seed == 42:
            path = TABLES / info["historical"]
        else:
            path = (
                TABLES
                / f"09_{name}_seed{seed}_validation_metrics.csv"
            )

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if len(frame) != 1:
            raise RuntimeError(
                f"Expected one metrics row: {path}"
            )

        row = frame.iloc[0].to_dict()

        rows.append({
            "architecture": name,
            "seed": seed,
            "model": row["model"],
            "best_epoch": row["best_epoch"],
            "accuracy": row["accuracy"],
            "balanced_accuracy": row["balanced_accuracy"],
            "macro_f1": row["macro_f1"],
            "nfr_precision": row["nfr_precision"],
            "nfr_recall": row["nfr_recall"],
            "nfr_f1": row["nfr_f1"],
            "fp": row["fp"],
            "fn": row["fn"],
            "training_seconds": row["training_seconds"],
        })

details = pd.DataFrame(rows)

metrics = [
    "nfr_f1",
    "nfr_recall",
    "nfr_precision",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
]

summary_rows = []

for name, subset in details.groupby("architecture"):
    record = {
        "architecture": name,
        "seed_count": len(subset),
    }

    for metric in metrics:
        values = subset[metric]

        record[f"{metric}_mean"] = values.mean()
        record[f"{metric}_std"] = values.std(ddof=1)
        record[f"{metric}_min"] = values.min()
        record[f"{metric}_max"] = values.max()

    summary_rows.append(record)

summary = pd.DataFrame(summary_rows).sort_values(
    "nfr_f1_mean",
    ascending=False,
)

details_path = (
    TABLES / "09_transformer_seed_results.csv"
)
summary_path = (
    TABLES / "09_transformer_seed_stability.csv"
)
report_path = (
    REPORTS / "09_transformer_multiseed_report.txt"
)

details.to_csv(details_path, index=False)
summary.to_csv(summary_path, index=False)

print("\nINDIVIDUAL SEED RESULTS")

print(
    details[
        [
            "architecture", "seed",
            "best_epoch", "nfr_f1",
            "nfr_recall", "macro_f1",
            "fp", "fn",
        ]
    ].round(4).to_string(index=False)
)

print("\nMULTI-SEED STABILITY")

print(
    summary[
        [
            "architecture",
            "seed_count",
            "nfr_f1_mean",
            "nfr_f1_std",
            "nfr_recall_mean",
            "nfr_recall_std",
            "macro_f1_mean",
            "macro_f1_std",
        ]
    ].round(4).to_string(index=False)
)

report_path.write_text(
    "\n".join([
        "SCRIPT 09: TRANSFORMER MULTI-SEED STABILITY",
        "=" * 76,
        "",
        "Dataset: D2-v1.1",
        "Train: 3378",
        "Validation: 728",
        "Test: SEALED",
        f"Seeds: {SEEDS}",
        "",
        "INDIVIDUAL RESULTS",
        details.round(4).to_string(index=False),
        "",
        "STABILITY SUMMARY",
        summary.round(4).to_string(index=False),
        "",
        "RESEARCH LIMITATIONS",
        "- Three seeds give limited variance estimates.",
        "- All results use the same validation partition.",
        "- Checkpoint selection uses validation NFR F1.",
        "- These results do not replace group-aware CV.",
        "- Validation was reused in earlier experiments.",
        "- Test evaluation remains sealed.",
    ]) + "\n",
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("Multi-seed stability: COMPLETED")
print("Original seed-42 checkpoints: PRESERVED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
print(details_path)
print(summary_path)
print(report_path)
print("\nScript 09 completed.")
