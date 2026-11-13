
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs/predictions/01_final_test_predictions.csv"
META = ROOT / "outputs/reports/01_final_test_metrics.json"
FIGURES = ROOT / "outputs/figures"
REPORT = ROOT / "outputs/reports/02_figure_generation_report.txt"

FIGURES.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


print("=" * 76)
print("FINAL MODEL - SCRIPT 02: EVALUATION FIGURES")
print("=" * 76)

require(PRED.is_file(), "Missing final predictions")
require(META.is_file(), "Missing final evaluation metadata")

metadata = json.loads(META.read_text(encoding="utf-8"))

require(
    metadata.get("status") == "COMPLETED",
    "Final test evaluation is not complete",
)

require(
    sha256(PRED) == metadata["prediction_file_sha256"],
    "Saved prediction SHA256 mismatch",
)

df = pd.read_csv(PRED)

require(len(df) == 727, "Unexpected prediction count")
require(df["requirement_id"].is_unique, "Duplicate IDs")

require(
    df["true_label"].isin(["FR", "NFR"]).all(),
    "Invalid true labels",
)
require(
    df["predicted_label"].isin(["FR", "NFR"]).all(),
    "Invalid predictions",
)

y = df["true_label"].map({"FR": 0, "NFR": 1}).to_numpy()
pred = df["predicted_label"].map(
    {"FR": 0, "NFR": 1}
).to_numpy()

prob = pd.to_numeric(
    df["nfr_probability"],
    errors="raise",
).to_numpy()

require(
    np.isfinite(prob).all()
    and ((prob >= 0) & (prob <= 1)).all(),
    "Invalid probabilities",
)

require(
    np.array_equal(pred, (prob >= 0.5).astype(int)),
    "Saved predictions disagree with frozen threshold",
)

cm = confusion_matrix(y, pred, labels=[0, 1])
require(
    cm.tolist() == [[441, 47], [31, 208]],
    "Confusion matrix differs from frozen results",
)

# Refuse to overwrite preexisting figure outputs.
names = [
    "02_confusion_matrix",
    "02_precision_recall_curve",
    "02_roc_curve",
    "02_per_class_metrics",
]
files = [
    FIGURES / f"{name}.{ext}"
    for name in names
    for ext in ("png", "pdf")
]
require(
    not any(path.exists() for path in files),
    "Figure outputs already exist; refusing to overwrite",
)
require(
    not REPORT.exists(),
    "Figure report already exists",
)


def save_figure(fig, name):
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# Figure 1: Confusion matrix
fig, ax = plt.subplots(figsize=(6.5, 5.5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FR", "NFR"],
)
display.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("DistilBERT: Final Test Confusion Matrix")
fig.tight_layout()
save_figure(fig, "02_confusion_matrix")


# Figure 2: Precision-recall curve
precision, recall, _ = precision_recall_curve(y, prob)
ap = average_precision_score(y, prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, linewidth=2)
ax.axhline(y.mean(), linestyle="--", alpha=0.6)
ax.set_xlabel("NFR Recall")
ax.set_ylabel("NFR Precision")
ax.set_title(f"Final Test Precision-Recall Curve (AP={ap:.4f})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "02_precision_recall_curve")


# Figure 3: ROC curve
fpr, tpr, _ = roc_curve(y, prob)
auc = roc_auc_score(y, prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Final Test ROC Curve (AUC={auc:.4f})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "02_roc_curve")


# Figure 4: Per-class metrics
p, r, f1, support = precision_recall_fscore_support(
    y, pred, labels=[0, 1], zero_division=0
)

x = np.arange(2)
width = 0.25

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.bar(x - width, p, width, label="Precision")
ax.bar(x, r, width, label="Recall")
ax.bar(x + width, f1, width, label="F1")
ax.set_xticks(x, ["FR", "NFR"])
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Final Test Per-Class Performance")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
save_figure(fig, "02_per_class_metrics")

report_lines = [
    "FINAL MODEL: TEST EVALUATION FIGURES",
    "=" * 76,
    "",
    "Model: Frozen DistilBERT, seed 42",
    "Test examples: 727",
    f"Prediction SHA256: {sha256(PRED)}",
    "",
    f"Average precision: {ap:.6f}",
    f"ROC AUC: {auc:.6f}",
    "",
    "CONFUSION MATRIX",
    str(cm),
    "",
    "FIGURES",
    *[str(path) for path in files],
    "",
    "No retraining or repeated model inference performed.",
    "No threshold optimization performed.",
    "Figures use saved final test predictions.",
]

REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\nFINAL TEST FIGURES")
for path in files:
    print(path)

print("\nRESEARCH STATUS")
print("Figure generation: COMPLETED")
print("Saved test predictions: VERIFIED")
print("New inference: NO")
print("Training: NO")
print("Threshold tuning: NO")
print("Final evaluation: LOCKED")
print("\nScript 02 completed.")
