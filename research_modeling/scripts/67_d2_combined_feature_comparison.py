
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score
)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "research_modeling/outputs/tables"
REPORTS = ROOT / "research_modeling/outputs/reports"

WORD_FILE = TABLES / "62_d2_svm_validation_predictions.csv"
COMBINED_FILE = TABLES / "66_d2_validation_predictions.csv"

CONFIGS = [
    ("S01_LinearSVM", "F03_CombinedSVM"),
    ("S02_BalancedLinearSVM", "F04_BalancedCombinedSVM")
]

word = pd.read_csv(WORD_FILE, dtype={"requirement_id": str})
combined = pd.read_csv(
    COMBINED_FILE, dtype={"requirement_id": str}
)

print("=" * 72)
print("SCRIPT 67: WORD VS COMBINED SVM COMPARISON")
print("=" * 72)

results = []
changes = []

for word_name, combined_name in CONFIGS:

    a = word[word["model"] == word_name].copy()
    b = combined[
        combined["model"] == combined_name
    ].copy()

    a = a.merge(
        b,
        on="requirement_id",
        suffixes=("_word", "_combined"),
        validate="one_to_one"
    )

    if len(a) != 728:
        raise RuntimeError("Incomplete validation alignment")

    if not (
        a["true_label_word"]
        == a["true_label_combined"]
    ).all():
        raise RuntimeError("Label mismatch")

    y = (
        a["true_label_word"] == "NFR"
    ).astype(int).to_numpy()

    pred_word = (
        a["predicted_label_word"] == "NFR"
    ).astype(int).to_numpy()

    pred_combined = (
        a["predicted_label_combined"] == "NFR"
    ).astype(int).to_numpy()

    score_word = a[
        "decision_score_word"
    ].to_numpy(dtype=float)

    score_combined = a[
        "decision_score_combined"
    ].to_numpy(dtype=float)

    word_correct = pred_word == y
    combined_correct = pred_combined == y

    fixed = int(
        np.sum(~word_correct & combined_correct)
    )

    introduced = int(
        np.sum(word_correct & ~combined_correct)
    )

    changed = int(
        np.sum(pred_word != pred_combined)
    )

    word_auc = roc_auc_score(y, score_word)
    combined_auc = roc_auc_score(y, score_combined)

    word_ap = average_precision_score(
        y, score_word
    )
    combined_ap = average_precision_score(
        y, score_combined
    )

    word_f1 = f1_score(y, pred_word)
    combined_f1 = f1_score(y, pred_combined)

    results.append({
        "word_model": word_name,
        "combined_model": combined_name,
        "predictions_changed": changed,
        "errors_corrected": fixed,
        "errors_introduced": introduced,
        "word_nfr_f1": word_f1,
        "combined_nfr_f1": combined_f1,
        "word_roc_auc": word_auc,
        "combined_roc_auc": combined_auc,
        "roc_auc_change": combined_auc - word_auc,
        "word_average_precision": word_ap,
        "combined_average_precision": combined_ap,
        "ap_change": combined_ap - word_ap
    })

    changed_rows = a[
        pred_word != pred_combined
    ].copy()

    changed_rows["comparison"] = (
        f"{word_name}_vs_{combined_name}"
    )

    changes.append(changed_rows)

    print(f"\n{word_name} VS {combined_name}")
    print("Predictions changed:", changed)
    print("Errors corrected:", fixed)
    print("Errors introduced:", introduced)
    print("Word NFR F1:", round(word_f1, 4))
    print("Combined NFR F1:", round(combined_f1, 4))
    print(
        "ROC-AUC improvement:",
        round(combined_auc - word_auc, 4)
    )
    print(
        "Average precision improvement:",
        round(combined_ap - word_ap, 4)
    )

summary = pd.DataFrame(results)

summary_path = (
    TABLES / "67_d2_feature_comparison.csv"
)

changes_path = (
    TABLES / "67_d2_changed_predictions.csv"
)

summary.to_csv(summary_path, index=False)

pd.concat(
    changes, ignore_index=True
).to_csv(changes_path, index=False)

REPORTS.mkdir(parents=True, exist_ok=True)

report_path = (
    REPORTS / "67_d2_feature_comparison.txt"
)

report_path.write_text(
    "SCRIPT 67: WORD VS COMBINED FEATURE COMPARISON\n"
    + "=" * 65
    + "\n\n"
    + summary.round(5).to_string(index=False)
    + "\n\nValidation only. Test set not accessed.\n",
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("Prediction comparison: COMPLETED")
print("Validation only: YES")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nScript 67 completed.")
