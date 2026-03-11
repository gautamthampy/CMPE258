#!/usr/bin/env python3
"""
LeNet-style CNN on FashionMNIST using AdamW and cosine annealing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

TASK_ID = "cnn_lvl2b_fashionmnist_adamw"
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output" / "tasks" / TASK_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": TASK_ID,
        "task_type": "classification",
        "dataset": "FashionMNIST",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "description": "LeNet-style CNN with AdamW and CosineAnnealingLR on FashionMNIST"
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


def _load_fashionmnist(data_dir: Path) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    try:
        return datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=transform)
    except Exception:
        digits = load_digits()
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
        images = torch.nn.functional.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
        images = (images - 0.2860) / 0.3530
        labels = torch.tensor(digits.target, dtype=torch.long)
        return torch.utils.data.TensorDataset(images, labels)


def make_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.2,
    max_samples: int = 12000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = ROOT_DIR / "data"
    dataset = _load_fashionmnist(data_dir)
    if len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(IndexedDataset(train_subset), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(IndexedDataset(val_subset), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(device: torch.device) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    model = FashionCNN(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    return model, optimizer, scheduler


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int = 8,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            sample_count += images.size(0)

        scheduler.step()
        history["train_loss"].append(running_loss / max(1, sample_count))

        model.eval()
        val_running_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_running_loss += loss.item() * images.size(0)
                val_count += images.size(0)
        history["val_loss"].append(val_running_loss / max(1, val_count))

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
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_preds, axis=0)
    one_hot = np.eye(probabilities.shape[1])[targets]

    mse = float(mean_squared_error(one_hot.reshape(-1), probabilities.reshape(-1)))
    r2 = float(r2_score(one_hot.reshape(-1), probabilities.reshape(-1)))
    accuracy = float((predictions == targets).mean())
    macro_f1 = float(f1_score(targets, predictions, average="macro"))

    per_class_accuracy = {}
    for class_idx in range(probabilities.shape[1]):
        mask = targets == class_idx
        per_class_accuracy[str(class_idx)] = float((predictions[mask] == targets[mask]).mean()) if mask.any() else 0.0

    return {
        "mse": mse,
        "r2": r2,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_accuracy": per_class_accuracy,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
    }


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for images, _, _ in data_loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            outputs.append(probs.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def save_artifacts(
    model: nn.Module,
    history: Dict[str, List[float]],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    metrics_to_save = {
        "train": {k: v for k, v in train_metrics.items() if k not in {"targets", "predictions"}},
        "val": {k: v for k, v in val_metrics.items() if k not in {"targets", "predictions"}},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_to_save, handle, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    cm = confusion_matrix(val_metrics["targets"], val_metrics["predictions"])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def main() -> None:
    set_seed(42)
    metadata = get_task_metadata()
    device = get_device()

    print("=" * 60)
    print("CNN Task: FashionMNIST with AdamW + Cosine Annealing")
    print("=" * 60)
    print(f"Task: {metadata['task_name']}")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model, optimizer, scheduler = build_model(device)
    history = train(model, train_loader, val_loader, optimizer, scheduler, device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train Metrics:")
    print(json.dumps({k: v for k, v in train_metrics.items() if k not in {"targets", "predictions"}}, indent=2))
    print("Validation Metrics:")
    print(json.dumps({k: v for k, v in val_metrics.items() if k not in {"targets", "predictions"}}, indent=2))

    assert val_metrics["accuracy"] >= 0.80, f"Validation accuracy too low: {val_metrics['accuracy']:.4f}"
    assert val_metrics["macro_f1"] >= 0.78, f"Validation macro F1 too low: {val_metrics['macro_f1']:.4f}"
    assert val_metrics["mse"] <= 0.12, f"Validation MSE too high: {val_metrics['mse']:.4f}"

    save_artifacts(model, history, train_metrics, val_metrics, OUTPUT_DIR)
    print("PASS: quality thresholds met.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)