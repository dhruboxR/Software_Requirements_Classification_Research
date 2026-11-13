
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score,
    roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# ==========================================================
# SCRIPT 06: D2 v1.1 BERT-BASE BASELINE
#
# Fully fine-tuned BERT-base
# Train / validation only
# Test partition remains sealed
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

for folder in (TABLES, REPORTS, PREDS, CHECKPOINTS):
    folder.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Requirement Text (Original Requirement)"
MODEL_NAME = "google-bert/bert-base-uncased"
EXPERIMENT = "D02_BERTBase"

SEED = 42
MAX_LENGTH = 96
BATCH_SIZE = 4
ACCUM_STEPS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 4
PATIENCE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
USE_FP16 = DEVICE.type == "cuda"
LABEL_MAP = {"FR": 0, "NFR": 1}

CHECKPOINT = CHECKPOINTS / "06_bert_base_best"

if CHECKPOINT.exists():
    raise FileExistsError(
        f"Checkpoint already exists: {CHECKPOINT}"
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(true, pred, probs):
    tn, fp, fn, tp = confusion_matrix(
        true, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(
            true, pred
        ),
        "macro_f1": f1_score(
            true, pred, average="macro",
            zero_division=0
        ),
        "nfr_precision": precision_score(
            true, pred, zero_division=0
        ),
        "nfr_recall": recall_score(
            true, pred, zero_division=0
        ),
        "nfr_f1": f1_score(
            true, pred, zero_division=0
        ),
        "average_precision": average_precision_score(
            true, probs
        ),
        "roc_auc": roc_auc_score(true, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ----------------------------------------------------------
# 1. Frozen data verification
# ----------------------------------------------------------

set_seed(SEED)

meta = json.loads(
    META.read_text(encoding="utf-8")
)

if meta.get("protocol") != "D2-v1.1":
    raise RuntimeError("Unexpected dataset protocol")

manifest_hash = sha256(MANIFEST)

if manifest_hash != meta["output_sha256"][MANIFEST.name]:
    raise RuntimeError("Manifest hash mismatch")

parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    selected = chunk.loc[
        chunk["split"].isin(["train", "validation"])
    ].copy()
    if not selected.empty:
        parts.append(selected)

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL, "Requirement ID", "key",
    "label", "split", "similarity_group"
}

if not required.issubset(df.columns):
    raise RuntimeError("Missing required columns")

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

if len(train) != 3378 or len(val) != 728:
    raise RuntimeError("Unexpected partition sizes")

if not train["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate training IDs")

if not val["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate validation IDs")

if set(train["similarity_group"]) & set(
    val["similarity_group"]
):
    raise RuntimeError("Similarity-group leakage")

if set(df["label"]) != {"FR", "NFR"}:
    raise RuntimeError("Unexpected labels")

if df[TEXT_COL].str.strip().eq("").any():
    raise RuntimeError("Blank requirement text")

print("=" * 76)
print("SCRIPT 06: D2 v1.1 BERT-BASE BASELINE")
print("=" * 76)

print("\nDATASET")
print("Training:", len(train))
print("Validation:", len(val))
print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("Manifest integrity: PASS")
print("Test accessed: NO")

# ----------------------------------------------------------
# 2. Model and tokenizer
# ----------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "FR", 1: "NFR"},
    label2id=LABEL_MAP,
)

if not hasattr(model, "gradient_checkpointing_enable"):
    raise RuntimeError(
        "Gradient checkpointing unavailable"
    )

model.gradient_checkpointing_enable()
model.to(DEVICE)

print("\nMODEL CONFIGURATION")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)
print("Max length:", MAX_LENGTH)
print("Physical batch size:", BATCH_SIZE)
print("Effective batch size:", BATCH_SIZE * ACCUM_STEPS)
print("FP16:", USE_FP16)
print("Gradient checkpointing: True")


class RequirementDataset(Dataset):
    def __init__(self, frame):
        self.ids = frame["Requirement ID"].tolist()
        self.labels = frame["label"].map(
            LABEL_MAP
        ).tolist()

        self.tokens = tokenizer(
            frame[TEXT_COL].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        result = {
            key: value[idx]
            for key, value in self.tokens.items()
        }
        result["labels"] = self.labels[idx]
        return result


train_data = RequirementDataset(train)
val_data = RequirementDataset(val)

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    generator=generator,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_FP16
)


@torch.no_grad()
def evaluate(current_model):
    current_model.eval()

    true = []
    pred = []
    probs = []

    for batch in val_loader:
        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_FP16
        ):
            outputs = current_model(**batch)

        logits = outputs.logits.float()
        probabilities = torch.softmax(
            logits, dim=1
        )[:, 1]

        true.extend(
            batch["labels"].cpu().tolist()
        )
        pred.extend(
            logits.argmax(dim=1).cpu().tolist()
        )
        probs.extend(
            probabilities.cpu().tolist()
        )

    return (
        calculate_metrics(true, pred, probs),
        pred,
        probs
    )


# ----------------------------------------------------------
# 3. Training
# ----------------------------------------------------------

history = []
best_f1 = -1.0
best_epoch = None
no_improvement = 0

start = time.perf_counter()

print("\nTRAINING")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_total = 0.0
    accumulated_samples = 0

    for step, batch in enumerate(train_loader, start=1):
        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        batch_size = len(batch["labels"])
        loss_scale = batch_size / (
            BATCH_SIZE * ACCUM_STEPS
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_FP16
        ):
            outputs = model(**batch)
            loss = outputs.loss

        loss_total += loss.item() * batch_size
        accumulated_samples += batch_size

        scaler.scale(loss * loss_scale).backward()

        if (
            step % ACCUM_STEPS == 0
            or step == len(train_loader)
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    training_loss = loss_total / accumulated_samples

    val_metrics, _, _ = evaluate(model)
    current_f1 = val_metrics["nfr_f1"]

    history.append({
        "epoch": epoch,
        "train_loss": training_loss,
        **val_metrics
    })

    print(
        f"Epoch {epoch:02d} | "
        f"loss={training_loss:.4f} | "
        f"NFR F1={current_f1:.4f} | "
        f"recall={val_metrics['nfr_recall']:.4f} | "
        f"macro F1={val_metrics['macro_f1']:.4f}"
    )

    if current_f1 > best_f1 + 1e-8:
        best_f1 = current_f1
        best_epoch = epoch
        no_improvement = 0

        model.save_pretrained(
            CHECKPOINT,
            safe_serialization=True
        )
        tokenizer.save_pretrained(CHECKPOINT)
    else:
        no_improvement += 1

    if no_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

elapsed = time.perf_counter() - start

# ----------------------------------------------------------
# 4. Reload best checkpoint
# ----------------------------------------------------------

del model

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT
).to(DEVICE)

final_metrics, predictions, probabilities = evaluate(model)

result = {
    "model": EXPERIMENT,
    "best_epoch": best_epoch,
    "training_seconds": round(elapsed, 2),
    **final_metrics
}

# ----------------------------------------------------------
# 5. Save research outputs
# ----------------------------------------------------------

history_path = TABLES / "06_bert_epoch_history.csv"
metrics_path = TABLES / "06_bert_validation_metrics.csv"
pred_path = PREDS / "06_bert_validation_predictions.csv"

pd.DataFrame(history).to_csv(
    history_path, index=False
)
pd.DataFrame([result]).to_csv(
    metrics_path, index=False
)

pd.DataFrame({
    "requirement_id": val["Requirement ID"],
    "true_label": val["label"],
    "predicted_label": [
        "NFR" if p == 1 else "FR"
        for p in predictions
    ],
    "nfr_probability": probabilities
}).to_csv(pred_path, index=False)

columns = [
    "model", "best_epoch",
    "accuracy", "balanced_accuracy",
    "macro_f1", "nfr_precision",
    "nfr_recall", "nfr_f1",
    "average_precision", "roc_auc",
    "fp", "fn", "training_seconds"
]

print("\n" + "=" * 76)
print("VALIDATION RESULTS")
print("=" * 76)
print(
    pd.DataFrame([result])[columns]
    .round(4).to_string(index=False)
)

print("\nCONFUSION MATRIX")
print("Rows: actual FR, NFR")
print("Columns: predicted FR, NFR")
print([
    [final_metrics["tn"], final_metrics["fp"]],
    [final_metrics["fn"], final_metrics["tp"]]
])

report_path = REPORTS / "06_bert_baseline_report.txt"
metadata_path = REPORTS / "06_bert_baseline_metadata.json"

report_path.write_text(
    "\n".join([
        "D2 v1.1 BERT-BASE BASELINE",
        "=" * 76,
        "",
        f"Model: {MODEL_NAME}",
        f"Seed: {SEED}",
        f"Best epoch: {best_epoch}",
        "Training: 3378",
        "Validation: 728",
        "Test accessed: NO",
        "",
        "CONFIGURATION",
        f"Max length: {MAX_LENGTH}",
        f"Physical batch size: {BATCH_SIZE}",
        f"Gradient accumulation: {ACCUM_STEPS}",
        f"Learning rate: {LR}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"FP16: {USE_FP16}",
        "Gradient checkpointing: True",
        "",
        "EPOCH HISTORY",
        pd.DataFrame(history).round(4).to_string(
            index=False
        ),
        "",
        "FINAL VALIDATION RESULTS",
        pd.DataFrame([result])[columns]
        .round(4).to_string(index=False),
        "",
        "LIMITATIONS",
        "- Single training seed.",
        "- Validation used for checkpoint selection.",
        "- No neural group-aware cross-validation.",
        "- Pretrained model differs from DistilBERT.",
        "- Physical batch sizes differ from DistilBERT.",
        "- Validation has been reused.",
        "- Test partition remains sealed."
    ]) + "\n",
    encoding="utf-8"
)

metadata_path.write_text(
    json.dumps({
        "experiment": "DL-06",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": "D2-v1.1",
        "manifest_sha256": manifest_hash,
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * ACCUM_STEPS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "fp16": USE_FP16,
        "gradient_checkpointing": True,
        "best_epoch": best_epoch,
        "checkpoint": str(CHECKPOINT),
        "test_accessed": False,
        "model_selection": "PENDING"
    }, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("BERT-base baseline: COMPLETED")
print("Best checkpoint: SAVED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in (
    history_path, metrics_path,
    pred_path, report_path,
    metadata_path, CHECKPOINT
):
    print(path)

print("\nScript 06 completed.")
