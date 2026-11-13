
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# ==========================================================
# SCRIPT 05: D2 v1.1 DISTILBERT BASELINE
#
# Fully fine-tuned pretrained DistilBERT
# Single seed, standard cross-entropy
#
# Train: 3378
# Validation: 728
# Test: SEALED
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deep_learning_research"
DATA = ROOT / "dataset_exploration/outputs/validated"

MANIFEST = DATA / "54_d2_v11_frozen_split_manifest.csv"
META = DATA / "54_d2_v11_split_metadata.json"

TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"
PREDS = PROJECT / "outputs/predictions"
CHECKPOINTS = PROJECT / "checkpoints"

for folder in [TABLES, REPORTS, PREDS, CHECKPOINTS]:
    folder.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Requirement Text (Original Requirement)"

MODEL_NAME = "distilbert/distilbert-base-uncased"
EXPERIMENT = "D01_DistilBERT_seed17"

SEED = 17
MAX_LENGTH = 96
BATCH_SIZE = 8
ACCUM_STEPS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 4
PATIENCE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
USE_FP16 = DEVICE.type == "cuda"

LABEL_MAP = {"FR": 0, "NFR": 1}

CHECKPOINT_DIR = CHECKPOINTS / "09_distilbert_seed17_best"

if CHECKPOINT_DIR.exists():
    raise FileExistsError(
        f"Checkpoint directory already exists: {CHECKPOINT_DIR}"
    )


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred, probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "nfr_precision": precision_score(
            y_true, y_pred,
            zero_division=0,
        ),
        "nfr_recall": recall_score(
            y_true, y_pred,
            zero_division=0,
        ),
        "nfr_f1": f1_score(
            y_true, y_pred,
            zero_division=0,
        ),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ----------------------------------------------------------
# 1. Validate frozen dataset
# ----------------------------------------------------------

set_seed(SEED)

meta = json.loads(
    META.read_text(encoding="utf-8")
)

assert meta["protocol"] == "D2-v1.1"

manifest_hash = sha256(MANIFEST)

assert manifest_hash == (
    meta["output_sha256"][MANIFEST.name]
)

parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    selected = chunk[
        chunk["split"].isin(["train", "validation"])
    ].copy()

    if len(selected):
        parts.append(selected)

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL,
    "Requirement ID",
    "key",
    "label",
    "split",
    "similarity_group",
}

assert required.issubset(df.columns)

train = (
    df[df["split"] == "train"]
    .sort_values("key")
    .reset_index(drop=True)
)

val = (
    df[df["split"] == "validation"]
    .sort_values("key")
    .reset_index(drop=True)
)

assert len(train) == 3378
assert len(val) == 728

assert train["Requirement ID"].is_unique
assert val["Requirement ID"].is_unique

assert not (
    set(train["similarity_group"])
    & set(val["similarity_group"])
)

assert set(df["label"]) == {"FR", "NFR"}
assert df[TEXT_COL].str.strip().ne("").all()

# ----------------------------------------------------------
# 2. Tokenizer and model
# ----------------------------------------------------------

print("=" * 76)
print("SCRIPT 05: D2 v1.1 DISTILBERT BASELINE")
print("=" * 76)

print("\nDATASET")
print("Training:", len(train))
print("Validation:", len(val))
print("Device:", DEVICE)
print("Model:", MODEL_NAME)
print("Max length:", MAX_LENGTH)
print("Manifest integrity: PASS")
print("Test accessed: NO")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "FR", 1: "NFR"},
    label2id=LABEL_MAP,
)

# The classification head is newly initialized.
# The underlying encoder uses pretrained weights.

model.to(DEVICE)

print("\nMODEL")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)
print("Mixed precision FP16:", USE_FP16)


class RequirementDataset(Dataset):
    def __init__(self, frame):
        self.ids = frame["Requirement ID"].tolist()
        self.labels = frame["label"].map(
            LABEL_MAP
        ).tolist()

        self.encodings = tokenizer(
            frame[TEXT_COL].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


train_dataset = RequirementDataset(train)
val_dataset = RequirementDataset(val)

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    generator=generator,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
)

# ----------------------------------------------------------
# 3. Training configuration
# ----------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_FP16,
)


@torch.no_grad()
def evaluate():
    model.eval()

    y_true = []
    y_pred = []
    probs = []

    for batch in val_loader:
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16 if USE_FP16 else torch.bfloat16,
            enabled=USE_FP16,
        ):
            outputs = model(**batch)

        logits = outputs.logits.float()

        batch_probs = torch.softmax(
            logits, dim=1
        )[:, 1]

        y_true.extend(
            batch["labels"].cpu().tolist()
        )

        y_pred.extend(
            logits.argmax(dim=1).cpu().tolist()
        )

        probs.extend(
            batch_probs.cpu().tolist()
        )

    return metrics(y_true, y_pred, probs), y_pred, probs


history = []

best_f1 = -1.0
best_epoch = None
no_improvement = 0

start_time = time.perf_counter()

# ----------------------------------------------------------
# 4. Training loop
# ----------------------------------------------------------

print("\nTRAINING")

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16 if USE_FP16 else torch.bfloat16,
            enabled=USE_FP16,
        ):
            outputs = model(**batch)
            loss = outputs.loss

        total_loss += (
            loss.item() * len(batch["labels"])
        )

        loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (
            step % ACCUM_STEPS == 0
            or step == len(train_loader)
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

    train_loss = total_loss / len(train_dataset)

    val_metrics, _, _ = evaluate()

    current_f1 = val_metrics["nfr_f1"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        **val_metrics,
    })

    print(
        f"Epoch {epoch:02d} | "
        f"loss={train_loss:.4f} | "
        f"NFR F1={current_f1:.4f} | "
        f"recall={val_metrics['nfr_recall']:.4f} | "
        f"macro F1={val_metrics['macro_f1']:.4f}"
    )

    if current_f1 > best_f1 + 1e-8:
        best_f1 = current_f1
        best_epoch = epoch
        no_improvement = 0

        model.save_pretrained(
            CHECKPOINT_DIR,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(
            CHECKPOINT_DIR
        )

    else:
        no_improvement += 1

    if no_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

training_seconds = time.perf_counter() - start_time

# ----------------------------------------------------------
# 5. Reload best checkpoint and evaluate
# ----------------------------------------------------------

del model

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_DIR
).to(DEVICE)

final_metrics, predictions, probabilities = evaluate()

# ----------------------------------------------------------
# 6. Save results
# ----------------------------------------------------------

history_path = TABLES / "09_distilbert_seed17_epoch_history.csv"
metrics_path = TABLES / "09_distilbert_seed17_validation_metrics.csv"
prediction_path = PREDS / "09_distilbert_seed17_validation_predictions.csv"

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
)

result = {
    "model": EXPERIMENT,
    "best_epoch": best_epoch,
    "training_seconds": round(training_seconds, 2),
    **final_metrics,
}

pd.DataFrame([result]).to_csv(
    metrics_path,
    index=False,
)

prediction_rows = []

for idx, (pred, prob) in enumerate(
    zip(predictions, probabilities)
):
    prediction_rows.append({
        "requirement_id": val.loc[
            idx, "Requirement ID"
        ],
        "true_label": val.loc[idx, "label"],
        "predicted_label": (
            "NFR" if pred == 1 else "FR"
        ),
        "nfr_probability": prob,
    })

pd.DataFrame(prediction_rows).to_csv(
    prediction_path,
    index=False,
)

print("\n" + "=" * 76)
print("VALIDATION RESULTS")
print("=" * 76)

display_columns = [
    "model",
    "best_epoch",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "nfr_precision",
    "nfr_recall",
    "nfr_f1",
    "average_precision",
    "roc_auc",
    "fp",
    "fn",
    "training_seconds",
]

print(
    pd.DataFrame([result])[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nCONFUSION MATRIX")
print("Rows: actual FR, NFR")
print("Columns: predicted FR, NFR")

print([
    [final_metrics["tn"], final_metrics["fp"]],
    [final_metrics["fn"], final_metrics["tp"]],
])

report_path = REPORTS / "09_distilbert_seed17_baseline_report.txt"
metadata_path = REPORTS / "09_distilbert_seed17_baseline_metadata.json"

report_lines = [
    "D2 v1.1 DISTILBERT BASELINE",
    "=" * 76,
    "",
    f"Pretrained model: {MODEL_NAME}",
    f"Seed: {SEED}",
    f"Best epoch: {best_epoch}",
    "Train: 3378",
    "Validation: 728",
    "Test: SEALED",
    "",
    "EXPERIMENT CONFIGURATION",
    f"Max sequence length: {MAX_LENGTH}",
    f"Batch size: {BATCH_SIZE}",
    f"Gradient accumulation: {ACCUM_STEPS}",
    f"Learning rate: {LR}",
    f"Weight decay: {WEIGHT_DECAY}",
    f"Max epochs: {MAX_EPOCHS}",
    f"Patience: {PATIENCE}",
    f"FP16: {USE_FP16}",
    "",
    "EPOCH HISTORY",
    pd.DataFrame(history).round(4).to_string(index=False),
    "",
    "FINAL VALIDATION METRICS",
    pd.DataFrame([result])[display_columns]
    .round(4).to_string(index=False),
    "",
    "LIMITATIONS",
    "- One training seed.",
    "- Validation selects the checkpoint.",
    "- No neural group-aware cross-validation.",
    "- No class weighting in this configuration.",
    "- Validation previously reused for ML development.",
    "- Dataset annotation review remains conditional.",
    "- Project-level independence is unverified.",
    "- Test partition remains sealed.",
]

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

metadata_path.write_text(
    json.dumps({
        "experiment": "DL-05",
        "protocol": "D2-v1.1",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pretrained_model": MODEL_NAME,
        "manifest_sha256": manifest_hash,
        "seed": SEED,
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": ACCUM_STEPS,
        "effective_batch_size": (
            BATCH_SIZE * ACCUM_STEPS
        ),
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "mixed_precision_fp16": USE_FP16,
        "best_epoch": best_epoch,
        "checkpoint": str(CHECKPOINT_DIR),
        "test_accessed": False,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8",
)

print("\nRESEARCH STATUS")
print("DistilBERT baseline: COMPLETED")
print("Best checkpoint: SAVED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    history_path,
    metrics_path,
    prediction_path,
    report_path,
    metadata_path,
    CHECKPOINT_DIR,
]:
    print(path)

print("\nScript 05 completed.")
