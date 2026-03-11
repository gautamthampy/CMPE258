#!/usr/bin/env python3
"""
GRU sentiment classification on a self-contained synthetic IMDB-like corpus.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, mean_squared_error, precision_score, r2_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

TASK_ID = "rnn_lvl2b_gru_imdb_sentiment"
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output" / "tasks" / TASK_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": TASK_ID,
        "task_type": "binary_classification",
        "dataset": "synthetic_imdb_like",
        "description": "Embedding + bidirectional GRU sentiment classifier with gradient clipping"
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_corpus(samples_per_class: int = 1500, seed: int = 42) -> List[Tuple[str, int]]:
    rng = np.random.default_rng(seed)
    subjects = ["movie", "film", "story", "ending", "performance", "soundtrack", "script"]
    positive = ["great", "excellent", "amazing", "heartwarming", "sharp", "memorable", "fun"]
    negative = ["awful", "boring", "flat", "terrible", "predictable", "weak", "messy"]
    intensifiers = ["really", "very", "truly", "surprisingly", "extremely"]
    connectives = ["and", "but", "because", "while"]
    neutral = ["the", "this", "that", "was", "felt", "looked", "seemed"]

    corpus: List[Tuple[str, int]] = []
    for _ in range(samples_per_class):
        subject = rng.choice(subjects)
        text = " ".join([
            "the",
            subject,
            rng.choice(neutral),
            rng.choice(intensifiers),
            rng.choice(positive),
            rng.choice(connectives),
            rng.choice(positive),
        ])
        corpus.append((text, 1))

    for _ in range(samples_per_class):
        subject = rng.choice(subjects)
        text = " ".join([
            "the",
            subject,
            rng.choice(neutral),
            rng.choice(intensifiers),
            rng.choice(negative),
            rng.choice(connectives),
            rng.choice(negative),
        ])
        corpus.append((text, 0))

    rng.shuffle(corpus)
    return corpus


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _build_vocab(corpus: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    counter = Counter()
    for text, _ in corpus:
        counter.update(_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class SentimentDataset(Dataset):
    def __init__(self, corpus: Sequence[Tuple[str, int]], vocab: Dict[str, int]):
        self.samples = corpus
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        text, label = self.samples[index]
        ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in _tokenize(text)]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def _collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels)
    return padded, lengths, labels_tensor


def make_dataloaders(
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    corpus = _build_corpus(seed=seed)
    vocab = _build_vocab(corpus)
    dataset = SentimentDataset(corpus, vocab)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)
    return train_loader, val_loader, vocab


class GRUSentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden).squeeze(1)


def build_model(device: torch.device, vocab_size: int) -> Tuple[nn.Module, optim.Optimizer]:
    model = GRUSentimentClassifier(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 12,
) -> Dict[str, List[float]]:
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for tokens, lengths, labels in train_loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * tokens.size(0)
            train_count += tokens.size(0)
        history["train_loss"].append(train_loss / max(1, train_count))

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for tokens, lengths, labels in val_loader:
                tokens = tokens.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                logits = model(tokens, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item() * tokens.size(0)
                val_count += tokens.size(0)
        history["val_loss"].append(val_loss / max(1, val_count))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={history['train_loss'][-1]:.4f} | "
            f"val_loss={history['val_loss'][-1]:.4f}"
        )

    return history


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for tokens, lengths, labels in data_loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            logits = model(tokens, lengths)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    predictions = (probabilities >= 0.5).astype(np.int64)

    mse = float(mean_squared_error(targets, probabilities))
    r2 = float(r2_score(targets, probabilities))
    accuracy = float((predictions == targets).mean())
    precision = float(precision_score(targets, predictions, zero_division=0))
    recall = float(recall_score(targets, predictions, zero_division=0))
    f1 = float(f1_score(targets, predictions, zero_division=0))

    return {
        "mse": mse,
        "r2": r2,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "targets": targets.tolist(),
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
    }


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for tokens, lengths, _ in data_loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            outputs.append(torch.sigmoid(model(tokens, lengths)).cpu().numpy())
    return np.concatenate(outputs)


def save_artifacts(
    model: nn.Module,
    history: Dict[str, List[float]],
    vocab: Dict[str, int],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, indent=2)

    metrics_to_save = {
        "train": {k: v for k, v in train_metrics.items() if k not in {"targets", "probabilities", "predictions"}},
        "val": {k: v for k, v in val_metrics.items() if k not in {"targets", "probabilities", "predictions"}},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_to_save, handle, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("GRU Sentiment Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()


def main() -> None:
    set_seed(42)
    metadata = get_task_metadata()
    device = get_device()

    print("=" * 60)
    print("Sequence Task: GRU Sentiment Classification")
    print("=" * 60)
    print(f"Task: {metadata['task_name']}")
    print(f"Device: {device}")

    train_loader, val_loader, vocab = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Vocabulary size: {len(vocab)}")

    model, optimizer = build_model(device, vocab_size=len(vocab))
    history = train(model, train_loader, val_loader, optimizer, device)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train Metrics:")
    print(json.dumps({k: v for k, v in train_metrics.items() if k not in {"targets", "probabilities", "predictions"}}, indent=2))
    print("Validation Metrics:")
    print(json.dumps({k: v for k, v in val_metrics.items() if k not in {"targets", "probabilities", "predictions"}}, indent=2))

    assert val_metrics["accuracy"] >= 0.90, f"Validation accuracy too low: {val_metrics['accuracy']:.4f}"
    assert val_metrics["f1"] >= 0.90, f"Validation F1 too low: {val_metrics['f1']:.4f}"
    assert history["val_loss"][-1] < history["val_loss"][0], "Validation loss did not improve"

    save_artifacts(model, history, vocab, train_metrics, val_metrics, OUTPUT_DIR)
    print("PASS: quality thresholds met.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)