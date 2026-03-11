#!/usr/bin/env python3
"""
MLP on MNIST with mixup augmentation, label smoothing, and warmup scheduling.
"""

import json
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
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

TASK_ID = "mlp_lvl2b_mnist_mixup_labelsmooth"
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output" / "tasks" / TASK_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": TASK_ID,
        "task_type": "classification",
        "dataset": "MNIST",
        "input_dim": 784,
        "num_classes": 10,
        "description": "MLP classifier with mixup, label smoothing, and warmup"
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlattenedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image.view(-1), label


def _digits_fallback() -> Dataset:
    digits = load_digits()
    images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
    images = torch.nn.functional.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
    images = (images - 0.1307) / 0.3081
    labels = torch.tensor(digits.target, dtype=torch.long)
    return TensorDataset(images, labels)


def _load_mnist(data_dir: Path) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        return datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    except Exception:
        return _digits_fallback()


def make_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.2,
    max_samples: int = 20000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = ROOT_DIR / "data"
    dataset = _load_mnist(data_dir)
    if len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(FlattenedDataset(train_subset), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FlattenedDataset(val_subset), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    device: torch.device,
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    model = MLPClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3, total_iters=2)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])
    return model, optimizer, scheduler


def _smoothed_one_hot(targets: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    result = torch.full((targets.size(0), num_classes), off_value, device=targets.device)
    result.scatter_(1, targets.unsqueeze(1), on_value)
    return result


def _mixup_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    return mixed_inputs, targets, targets[permutation], float(lam)


def _mixup_loss(logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float, smoothing: float = 0.1) -> torch.Tensor:
    num_classes = logits.size(1)
    probs_a = _smoothed_one_hot(targets_a, num_classes, smoothing)
    probs_b = _smoothed_one_hot(targets_b, num_classes, smoothing)
    mixed_targets = lam * probs_a + (1.0 - lam) * probs_b
    log_probs = torch.log_softmax(logits, dim=1)
    return -(mixed_targets * log_probs).sum(dim=1).mean()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int = 10,
) -> Dict[str, List[float]]:
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            features, labels_a, labels_b, lam = _mixup_batch(features, labels)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = _mixup_loss(logits, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_count += features.size(0)

        scheduler.step()
        history["train_loss"].append(train_loss / max(1, train_count))

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                smoothed = _smoothed_one_hot(labels, logits.size(1), smoothing=0.1)
                loss = -(smoothed * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                val_loss += loss.item() * features.size(0)
                val_count += features.size(0)
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
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            probs = torch.softmax(model(features), dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_preds, axis=0)
    one_hot = np.eye(probabilities.shape[1])[targets]

    mse = float(mean_squared_error(one_hot.reshape(-1), probabilities.reshape(-1)))
    r2 = float(r2_score(one_hot.reshape(-1), probabilities.reshape(-1)))
    accuracy = float((predictions == targets).mean())
    macro_f1 = float(f1_score(targets, predictions, average="macro"))

    return {
        "mse": mse,
        "r2": r2,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
    }


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            outputs.append(torch.softmax(model(features), dim=1).cpu().numpy())
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
    plt.title("MLP Mixup + Label Smoothing Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()


def main() -> None:
    set_seed(42)
    metadata = get_task_metadata()
    device = get_device()

    print("=" * 60)
    print("MLP Task: MNIST with Mixup + Label Smoothing + Warmup")
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

    assert val_metrics["accuracy"] >= 0.92, f"Validation accuracy too low: {val_metrics['accuracy']:.4f}"
    assert val_metrics["macro_f1"] >= 0.92, f"Validation macro F1 too low: {val_metrics['macro_f1']:.4f}"
    assert history["val_loss"][-1] < history["val_loss"][0], "Validation loss did not improve"

    save_artifacts(model, history, train_metrics, val_metrics, OUTPUT_DIR)
    print("PASS: quality thresholds met.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)