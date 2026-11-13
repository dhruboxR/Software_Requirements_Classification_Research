
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
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==========================================================
# SCRIPT 04: D2 v1.1 BiLSTM BASELINES
#
# L01: Standard BiLSTM
# L02: Balanced BiLSTM
#
# Vocabulary built from training text only.
# Best checkpoint selected by validation NFR F1.
# Test requirements are never loaded.
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
MODELS = PROJECT / "models"

for folder in (TABLES, REPORTS, PREDS, CHECKPOINTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Requirement Text (Original Requirement)"

SEED = 42
MAX_LENGTH = 64
MIN_FREQ = 2
EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.5
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 15
PATIENCE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABEL_MAP = {"FR": 0, "NFR": 1}

CONFIGS = {
    "L01_BiLSTM": False,
    "L02_BalancedBiLSTM": True,
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

    for token, frequency in sorted(counter.items()):
        if frequency >= MIN_FREQ:
            vocab[token] = len(vocab)

    return vocab


def encode(text, vocab):
    tokens = tokenize(text)[:MAX_LENGTH]
    length = len(tokens)

    if length == 0:
        raise RuntimeError("Empty token sequence")

    ids = [vocab.get(token, 1) for token in tokens]
    ids += [0] * (MAX_LENGTH - length)

    return ids, length


class RequirementDataset(Dataset):
    def __init__(self, frame, vocab):
        encoded = [
            encode(text, vocab)
            for text in frame[TEXT_COL].tolist()
        ]

        self.x = torch.tensor(
            [item[0] for item in encoded],
            dtype=torch.long
        )

        self.lengths = torch.tensor(
            [item[1] for item in encoded],
            dtype=torch.long
        )

        self.y = torch.tensor(
            frame["label"].map(LABEL_MAP).tolist(),
            dtype=torch.long
        )

        self.ids = frame["Requirement ID"].tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            self.x[index],
            self.lengths[index],
            self.y[index],
        )


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            2 * HIDDEN_DIM,
            2
        )

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Last forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        representation = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        return self.classifier(
            self.dropout(representation)
        )


def calculate_metrics(true, predicted):
    tn, fp, fn, tp = confusion_matrix(
        true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            true, predicted
        ),
        "macro_f1": f1_score(
            true, predicted,
            average="macro",
            zero_division=0
        ),
        "nfr_precision": precision_score(
            true, predicted,
            zero_division=0
        ),
        "nfr_recall": recall_score(
            true, predicted,
            zero_division=0
        ),
        "nfr_f1": f1_score(
            true, predicted,
            zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    y_true = []
    y_pred = []
    probabilities = []

    for x, lengths, y in loader:
        logits = model(
            x.to(DEVICE),
            lengths
        )

        prob = torch.softmax(
            logits.float(),
            dim=1
        )[:, 1]

        y_true.extend(y.tolist())

        y_pred.extend(
            logits.argmax(dim=1).cpu().tolist()
        )

        probabilities.extend(
            prob.cpu().tolist()
        )

    return (
        calculate_metrics(y_true, y_pred),
        y_pred,
        probabilities
    )


# ----------------------------------------------------------
# 1. Frozen dataset verification
# ----------------------------------------------------------

metadata = json.loads(
    META.read_text(encoding="utf-8")
)

if metadata.get("protocol") != "D2-v1.1":
    raise RuntimeError("Unexpected dataset protocol")

manifest_hash = sha256(MANIFEST)

if manifest_hash != metadata[
    "output_sha256"
][MANIFEST.name]:
    raise RuntimeError("Frozen manifest hash mismatch")

parts = []

for chunk in pd.read_csv(
    MANIFEST,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
    chunksize=1000
):
    selected = chunk[
        chunk["split"].isin(["train", "validation"])
    ].copy()

    if not selected.empty:
        parts.append(selected)

df = pd.concat(parts, ignore_index=True)

required = {
    TEXT_COL,
    "Requirement ID",
    "key",
    "label",
    "split",
    "similarity_group"
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
    raise RuntimeError("Unexpected partition counts")

if not train["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate training IDs")

if not val["Requirement ID"].is_unique:
    raise RuntimeError("Duplicate validation IDs")

if set(train["similarity_group"]) & set(
    val["similarity_group"]
):
    raise RuntimeError("Similarity group overlap")

if set(df["label"]) != {"FR", "NFR"}:
    raise RuntimeError("Unexpected labels")

if df[TEXT_COL].str.strip().eq("").any():
    raise RuntimeError("Blank requirement text")

# ----------------------------------------------------------
# 2. Vocabulary and datasets
# ----------------------------------------------------------

vocab = build_vocab(train[TEXT_COL].tolist())

vocab_path = MODELS / "04_bilstm_vocabulary.json"

if vocab_path.exists():
    existing = json.loads(
        vocab_path.read_text(encoding="utf-8")
    )
    if existing != vocab:
        raise RuntimeError(
            "Existing BiLSTM vocabulary differs"
        )
else:
    vocab_path.write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

train_data = RequirementDataset(train, vocab)
val_data = RequirementDataset(val, vocab)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("=" * 76)
print("SCRIPT 04: D2 v1.1 BiLSTM BASELINE")
print("=" * 76)

print("\nDATASET")
print("Training:", len(train))
print("Validation:", len(val))
print("Training FR:", int((train["label"] == "FR").sum()))
print("Training NFR:", int((train["label"] == "NFR").sum()))
print("Vocabulary size:", len(vocab))
print("Max word-token length:", MAX_LENGTH)
print("Device:", DEVICE)
print("Frozen manifest integrity: PASS")
print("Test accessed: NO")

# ----------------------------------------------------------
# 3. Train two fixed configurations
# ----------------------------------------------------------

history_rows = []
results_rows = []
prediction_rows = []

for name, balanced in CONFIGS.items():
    set_seed(SEED)

    checkpoint_path = CHECKPOINTS / f"04_{name}_best.pt"

    if checkpoint_path.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}"
        )

    model = BiLSTMClassifier(
        vocab_size=len(vocab)
    ).to(DEVICE)

    train_labels = train_data.y.numpy()

    counts = np.bincount(
        train_labels,
        minlength=2
    )

    weights = None

    if balanced:
        weights = torch.tensor(
            len(train_labels) / (2 * counts),
            dtype=torch.float32,
            device=DEVICE
        )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
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
    best_epoch = None
    no_improvement = 0

    start_time = time.perf_counter()

    print(f"\nTRAINING: {name}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for x, lengths, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x, lengths)
            loss = criterion(logits, y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_data)

        validation_metrics, _, _ = evaluate(
            model, val_loader
        )

        current_f1 = validation_metrics["nfr_f1"]

        history_rows.append({
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            **validation_metrics
        })

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"NFR F1={current_f1:.4f} | "
            f"recall={validation_metrics['nfr_recall']:.4f}"
        )

        if current_f1 > best_f1 + 1e-8:
            best_f1 = current_f1
            best_epoch = epoch
            no_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "dataset_protocol": "D2-v1.1",
                "manifest_sha256": manifest_hash,
                "vocab_size": len(vocab),
                "max_length": MAX_LENGTH,
                "embedding_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "bidirectional": True,
                "balanced": balanced,
                "seed": SEED,
                "best_epoch": best_epoch
            }, checkpoint_path)

        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    elapsed = time.perf_counter() - start_time

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics, predictions, probs = evaluate(
        model, val_loader
    )

    results_rows.append({
        "model": name,
        "balanced": balanced,
        "best_epoch": best_epoch,
        "training_seconds": round(elapsed, 2),
        **final_metrics
    })

    for idx, (prediction, probability) in enumerate(
        zip(predictions, probs)
    ):
        prediction_rows.append({
            "model": name,
            "requirement_id": val_data.ids[idx],
            "true_label": val.loc[idx, "label"],
            "predicted_label": (
                "NFR" if prediction == 1 else "FR"
            ),
            "nfr_probability": probability
        })

    print(
        f"BEST {name}: "
        f"epoch={best_epoch}, "
        f"NFR F1={final_metrics['nfr_f1']:.4f}, "
        f"recall={final_metrics['nfr_recall']:.4f}"
    )

# ----------------------------------------------------------
# 4. Save results
# ----------------------------------------------------------

history = pd.DataFrame(history_rows)
results = pd.DataFrame(results_rows)

history_path = TABLES / "04_bilstm_epoch_history.csv"
metrics_path = TABLES / "04_bilstm_validation_metrics.csv"
pred_path = PREDS / "04_bilstm_validation_predictions.csv"

history.to_csv(history_path, index=False)
results.to_csv(metrics_path, index=False)
pd.DataFrame(prediction_rows).to_csv(
    pred_path,
    index=False
)

display_columns = [
    "model",
    "best_epoch",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "nfr_precision",
    "nfr_recall",
    "nfr_f1",
    "fp",
    "fn",
    "training_seconds"
]

print("\n" + "=" * 76)
print("VALIDATION COMPARISON")
print("=" * 76)

print(
    results[display_columns]
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
        [int(row["fn"]), int(row["tp"])]
    ])

report_path = REPORTS / "04_bilstm_baseline_report.txt"
metadata_path = REPORTS / "04_bilstm_baseline_metadata.json"

report_path.write_text(
    "\n".join([
        "D2 v1.1 BiLSTM BASELINE",
        "=" * 70,
        "",
        "Training: 3378",
        "Validation: 728",
        "Test accessed: NO",
        "Vocabulary fitted on training only",
        "Early stopping metric: validation NFR F1",
        "Seed: 42",
        "",
        "VALIDATION COMPARISON",
        results[display_columns]
        .round(4).to_string(index=False),
        "",
        "LIMITATIONS",
        "- Single random seed.",
        "- No grouped CV for neural models.",
        "- No pretrained embeddings.",
        "- Validation used for checkpoint selection.",
        "- Test partition remains sealed.",
    ]) + "\n",
    encoding="utf-8"
)

metadata_path.write_text(
    json.dumps({
        "experiment": "DL-04",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": "D2-v1.1",
        "manifest_sha256": manifest_hash,
        "seed": SEED,
        "device": str(DEVICE),
        "vocab_size": len(vocab),
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "configurations": CONFIGS,
        "test_accessed": False,
        "model_selection": "PENDING"
    }, indent=2),
    encoding="utf-8"
)

print("\nRESEARCH STATUS")
print("BiLSTM baseline: COMPLETED")
print("Best checkpoints: SAVED")
print("Validation evaluation: COMPLETED")
print("Test set: SEALED")
print("Final model selection: PENDING")

print("\nFILES GENERATED")
for path in (
    vocab_path,
    history_path,
    metrics_path,
    pred_path,
    report_path,
    metadata_path
):
    print(path)

print("\nScript 04 completed.")
