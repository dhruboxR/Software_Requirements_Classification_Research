
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# SCRIPT 03: D2 v1.1 TEXTCNN BASELINES
#
# C01: Standard TextCNN
# C02: Class-weighted TextCNN
#
# Training vocabulary: TRAIN ONLY
# Early stopping: VALIDATION NFR F1
# Test partition: NEVER LOADED
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deep_learning_research"
DATA_DIR = ROOT / "dataset_exploration/outputs/validated"

MANIFEST = DATA_DIR / "54_d2_v11_frozen_split_manifest.csv"
SPLIT_META = DATA_DIR / "54_d2_v11_split_metadata.json"

TABLES = PROJECT / "outputs/tables"
REPORTS = PROJECT / "outputs/reports"
PREDS = PROJECT / "outputs/predictions"
CHECKPOINTS = PROJECT / "checkpoints"

for directory in (TABLES, REPORTS, PREDS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Requirement Text (Original Requirement)"

SEED = 42
MAX_LENGTH = 64
MIN_FREQ = 2
EMBED_DIM = 128
FILTERS = 64
KERNELS = (3, 4, 5)
DROPOUT = 0.5
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 15
PATIENCE = 4

LABELS = {"FR": 0, "NFR": 1}
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONFIGS = {
    "C01_TextCNN": False,
    "C02_BalancedTextCNN": True,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return re.findall(r"\S+", text.lower())


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for token, freq in sorted(counter.items()):
        if freq >= MIN_FREQ:
            vocab[token] = len(vocab)

    return vocab


def encode(text, vocab):
    tokens = tokenize(text)[:MAX_LENGTH]
    length = len(tokens)
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (MAX_LENGTH - len(ids))
    return ids, length


class RequirementDataset(Dataset):
    def __init__(self, frame, vocab):
        encoded = [
            encode(text, vocab)
            for text in frame[TEXT_COL].tolist()
        ]
        self.x = torch.tensor(
            [item[0] for item in encoded],
            dtype=torch.long,
        )
        self.lengths = torch.tensor(
            [item[1] for item in encoded],
            dtype=torch.long,
        )
        self.y = torch.tensor(
            frame["label"].map(LABELS).tolist(),
            dtype=torch.long,
        )
        self.ids = frame["Requirement ID"].tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.lengths[idx], self.y[idx]


class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, EMBED_DIM, padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                EMBED_DIM, FILTERS, kernel_size=k
            )
            for k in KERNELS
        ])

        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(
            FILTERS * len(KERNELS), 2
        )

    def forward(self, x, lengths):
        embedded = self.embedding(x).transpose(1, 2)
        pooled = []

        for conv, kernel in zip(self.convs, KERNELS):
            features = torch.relu(conv(embedded))

            # Do not pool windows containing padding
            valid_windows = (
                lengths - kernel + 1
            ).clamp(min=1)

            positions = torch.arange(
                features.shape[2],
                device=features.device
            )[None, :]

            mask = positions >= valid_windows[:, None]
            features = features.masked_fill(
                mask[:, None, :],
                torch.finfo(features.dtype).min,
            )

            pooled.append(features.max(dim=2).values)

        combined = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(combined))


def calculate_metrics(y_true, y_pred):
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
            average="macro", zero_division=0
        ),
        "nfr_precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "nfr_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "nfr_f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    true_labels = []
    predictions = []
    nfr_probs = []

    for x, lengths, y in loader:
        logits = model(
            x.to(DEVICE), lengths.to(DEVICE)
        )
        probs = torch.softmax(
            logits.float(), dim=1
        )[:, 1]

        true_labels.extend(y.tolist())
        predictions.extend(
            logits.argmax(dim=1).cpu().tolist()
        )
        nfr_probs.extend(probs.cpu().tolist())

    metrics = calculate_metrics(
        true_labels, predictions
    )

    return (
        metrics,
        predictions,
        nfr_probs,
    )


# ------------------------------------------------------------
# 1. Read and verify frozen inputs
# ------------------------------------------------------------

set_seed(SEED)

metadata = json.loads(
    SPLIT_META.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError("Wrong dataset protocol")

if sha256(MANIFEST) != metadata[
    "output_sha256"
][MANIFEST.name]:
    raise RuntimeError("Frozen manifest SHA256 mismatch")

parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000,
):
    part = chunk[
        chunk["split"].isin(["train", "validation"])
    ].copy()
    if len(part):
        parts.append(part)

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL, "Requirement ID", "label",
    "split", "similarity_group", "key",
}
if not required.issubset(df.columns):
    raise RuntimeError(
        f"Missing columns: {required - set(df.columns)}"
    )

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
    raise RuntimeError("Unexpected split sizes")

if (
    train["Requirement ID"].duplicated().any()
    or val["Requirement ID"].duplicated().any()
):
    raise RuntimeError("Duplicate requirement IDs")

if set(train["similarity_group"]) & set(
    val["similarity_group"]
):
    raise RuntimeError("Similarity-group overlap")

if set(df["label"]) != {"FR", "NFR"}:
    raise RuntimeError("Unexpected class labels")

if df[TEXT_COL].str.strip().eq("").any():
    raise RuntimeError("Blank requirement text")

vocab = build_vocab(train[TEXT_COL].tolist())

train_data = RequirementDataset(train, vocab)
val_data = RequirementDataset(val, vocab)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

vocab_path = PROJECT / "models/03_textcnn_vocabulary.json"
vocab_path.write_text(
    json.dumps(vocab, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 76)
print("SCRIPT 03: D2 v1.1 TEXTCNN BASELINE")
print("=" * 76)

print("\nDATASET")
print("Training:", len(train))
print("Validation:", len(val))
print("Training FR:", int((train["label"] == "FR").sum()))
print("Training NFR:", int((train["label"] == "NFR").sum()))
print("Vocabulary size:", len(vocab))
print("Maximum word-token length:", MAX_LENGTH)
print("Device:", DEVICE)
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

# ------------------------------------------------------------
# 2. Train separate fixed baseline configurations
# ------------------------------------------------------------

history_rows = []
result_rows = []
prediction_rows = []

for name, balanced in CONFIGS.items():

    # Restart from the same random seed so both models
    # begin from comparable initial conditions.
    set_seed(SEED)

    model = TextCNN(len(vocab)).to(DEVICE)

    y_train = train_data.y.numpy()
    class_counts = np.bincount(
        y_train, minlength=2
    )

    if balanced:
        weights = len(y_train) / (
            2 * class_counts
        )
        weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE
        )
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINTS / f"03_{name}_best.pt"
    )

    start = time.perf_counter()

    print(f"\nTRAINING: {name}")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for x, lengths, y in train_loader:
            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x, lengths)
            loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            total_loss += (
                loss.item() * len(y)
            )

        train_loss = total_loss / len(train_data)

        val_metrics, _, _ = evaluate_model(
            model, val_loader
        )

        current_f1 = val_metrics["nfr_f1"]

        history_rows.append({
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"NFR F1={current_f1:.4f} | "
            f"recall={val_metrics['nfr_recall']:.4f}"
        )

        if current_f1 > best_f1 + 1e-8:
            best_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "max_length": MAX_LENGTH,
                "embedding_dim": EMBED_DIM,
                "filters": FILTERS,
                "kernels": list(KERNELS),
                "dropout": DROPOUT,
                "balanced": balanced,
                "seed": SEED,
                "best_epoch": best_epoch,
                "dataset_protocol": "D2-v1.1",
                "manifest_sha256": sha256(MANIFEST),
            }, checkpoint_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    training_seconds = time.perf_counter() - start

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics, pred, probs = evaluate_model(
        model, val_loader
    )

    result_rows.append({
        "model": name,
        "balanced": balanced,
        "best_epoch": best_epoch,
        "training_seconds": round(
            training_seconds, 2
        ),
        **final_metrics,
    })

    for i, (prediction, prob) in enumerate(
        zip(pred, probs)
    ):
        prediction_rows.append({
            "model": name,
            "requirement_id": val_data.ids[i],
            "true_label": val.loc[i, "label"],
            "predicted_label": (
                "NFR" if prediction == 1 else "FR"
            ),
            "nfr_probability": prob,
        })

    print(
        f"BEST {name}: epoch={best_epoch}, "
        f"NFR F1={final_metrics['nfr_f1']:.4f}, "
        f"recall={final_metrics['nfr_recall']:.4f}"
    )

# ------------------------------------------------------------
# 3. Save complete experiment records
# ------------------------------------------------------------

history = pd.DataFrame(history_rows)
results = pd.DataFrame(result_rows)

history_path = TABLES / "03_textcnn_epoch_history.csv"
metrics_path = TABLES / "03_textcnn_validation_metrics.csv"
predictions_path = PREDS / "03_textcnn_validation_predictions.csv"

history.to_csv(history_path, index=False)
results.to_csv(metrics_path, index=False)

pd.DataFrame(prediction_rows).to_csv(
    predictions_path, index=False
)

display_cols = [
    "model", "best_epoch", "accuracy",
    "balanced_accuracy", "macro_f1",
    "nfr_precision", "nfr_recall", "nfr_f1",
    "fp", "fn", "training_seconds"
]

print("\n" + "=" * 76)
print("VALIDATION COMPARISON")
print("=" * 76)
print(
    results[display_cols]
    .round(4)
    .to_string(index=False)
)

print("\nVALIDATION CONFUSION MATRICES")
print("Rows: actual FR, NFR")
print("Columns: predicted FR, NFR")

for _, row in results.iterrows():
    print(f"\n{row['model']}")
    print([
        [int(row["tn"]), int(row["fp"])],
        [int(row["fn"]), int(row["tp"])],
    ])

report_path = REPORTS / "03_textcnn_baseline_report.txt"
metadata_path = REPORTS / "03_textcnn_baseline_metadata.json"

report_lines = [
    "D2 v1.1 TEXTCNN BASELINE",
    "=" * 76,
    "",
    "Train: 3378",
    "Validation: 728",
    "Test accessed: NO",
    "Preprocessing vocabulary: TRAIN ONLY",
    "Model initialization seed: 42",
    "Early stopping: validation NFR F1",
    "No hyperparameter search",
    "",
    "VALIDATION COMPARISON",
    results[display_cols].round(4).to_string(index=False),
    "",
    "IMPORTANT LIMITATIONS",
    "- Neural results are from one random seed.",
    "- Validation selects the best checkpoint.",
    "- No grouped cross-validation in this experiment.",
    "- No pretrained word embeddings were used.",
    "- Validation was used during prior ML development.",
    "- Project independence is not established.",
    "- The frozen test partition remains sealed.",
]

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

metadata_path.write_text(
    json.dumps({
        "experiment": "DL-03",
        "protocol": "D2-v1.1",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": SEED,
        "manifest_sha256": sha256(MANIFEST),
        "device": str(DEVICE),
        "train_count": len(train),
        "validation_count": len(val),
        "test_accessed": False,
        "vocab_size": len(vocab),
        "max_length": MAX_LENGTH,
        "min_frequency": MIN_FREQ,
        "embedding_dim": EMBED_DIM,
        "filters_per_kernel": FILTERS,
        "kernel_sizes": list(KERNELS),
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "early_stopping_metric": "validation_nfr_f1",
        "model_configs": CONFIGS,
        "model_selection": "PENDING",
    }, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("TextCNN baseline: COMPLETED")
print("Best checkpoints: SAVED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in [
    vocab_path,
    history_path,
    metrics_path,
    predictions_path,
    report_path,
    metadata_path,
]:
    print(path)

print("\nScript 03 completed.")
