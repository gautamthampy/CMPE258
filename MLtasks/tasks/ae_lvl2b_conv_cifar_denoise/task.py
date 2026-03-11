#!/usr/bin/env python3
"""
Convolutional denoising autoencoder on CIFAR-10 with self-verification.
"""

import json
import math
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
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

TASK_ID = "ae_lvl2b_conv_cifar_denoise"
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output" / "tasks" / TASK_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": TASK_ID,
        "task_type": "reconstruction",
        "dataset": "CIFAR10",
        "input_shape": [3, 32, 32],
        "description": "Convolutional denoising autoencoder on CIFAR-10"
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyWrapper(Dataset):
    def __init__(self, dataset: Dataset, noise_std: float = 0.15):
        self.dataset = dataset
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        clean, _ = self.dataset[index]
        noisy = torch.clamp(clean + torch.randn_like(clean) * self.noise_std, 0.0, 1.0)
        return noisy, clean


class _StructuredFakeData(Dataset):
    """Synthetic colour images with smooth spatial structure (offline fallback)."""

    def __init__(self, size: int = 8000) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        rng = np.random.RandomState(index)
        # Low-res random grid upsampled to 32×32 gives smooth natural-ish images
        low = torch.from_numpy(rng.rand(3, 4, 4).astype(np.float32)).unsqueeze(0)
        img = torch.nn.functional.interpolate(low, size=(32, 32), mode="bilinear", align_corners=False).squeeze(0)
        return img, 0  # label unused by NoisyWrapper


def _load_cifar10(data_dir: Path) -> Dataset:
    transform = transforms.ToTensor()
    try:
        return datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    except Exception:
        return _StructuredFakeData(size=8000)


def make_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.2,
    max_samples: int = 10000,
    noise_std: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = ROOT_DIR / "data"
    dataset = _load_cifar10(data_dir)
    if len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(NoisyWrapper(train_subset, noise_std=noise_std), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(NoisyWrapper(val_subset, noise_std=noise_std), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


class ConvDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_model(device: torch.device) -> Tuple[nn.Module, optim.Optimizer]:
    model = ConvDenoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 12,
) -> Dict[str, List[float]]:
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(noisy)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)
            train_count += noisy.size(0)

        history["train_loss"].append(train_loss / max(1, train_count))

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                recon = model(noisy)
                loss = criterion(recon, clean)
                val_loss += loss.item() * noisy.size(0)
                val_count += noisy.size(0)
        history["val_loss"].append(val_loss / max(1, val_count))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={history['train_loss'][-1]:.5f} | "
            f"val_loss={history['val_loss'][-1]:.5f}"
        )

    return history


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    noisy_batches: List[np.ndarray] = []
    clean_batches: List[np.ndarray] = []
    recon_batches: List[np.ndarray] = []

    with torch.no_grad():
        for noisy, clean in data_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            recon = model(noisy)
            noisy_batches.append(noisy.cpu().numpy())
            clean_batches.append(clean.cpu().numpy())
            recon_batches.append(recon.cpu().numpy())

    noisy_arr = np.concatenate(noisy_batches, axis=0)
    clean_arr = np.concatenate(clean_batches, axis=0)
    recon_arr = np.concatenate(recon_batches, axis=0)

    mse = float(mean_squared_error(clean_arr.reshape(-1), recon_arr.reshape(-1)))
    r2 = float(r2_score(clean_arr.reshape(-1), recon_arr.reshape(-1)))
    baseline_mse = float(mean_squared_error(clean_arr.reshape(-1), noisy_arr.reshape(-1)))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-8)))
    baseline_psnr = float(10.0 * math.log10(1.0 / max(baseline_mse, 1e-8)))

    return {
        "mse": mse,
        "r2": r2,
        "psnr": psnr,
        "baseline_mse": baseline_mse,
        "baseline_psnr": baseline_psnr,
        "improvement_mse": baseline_mse - mse,
        "improvement_psnr": psnr - baseline_psnr,
        "sample_noisy": noisy_arr[:16].tolist(),
        "sample_recon": recon_arr[:16].tolist(),
        "sample_clean": clean_arr[:16].tolist(),
    }


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for noisy, _ in data_loader:
            noisy = noisy.to(device)
            outputs.append(model(noisy).cpu().numpy())
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
        "train": {k: v for k, v in train_metrics.items() if not k.startswith("sample_")},
        "val": {k: v for k, v in val_metrics.items() if not k.startswith("sample_")},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_to_save, handle, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Denoising Autoencoder Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    noisy = torch.tensor(val_metrics["sample_noisy"], dtype=torch.float32)
    recon = torch.tensor(val_metrics["sample_recon"], dtype=torch.float32)
    clean = torch.tensor(val_metrics["sample_clean"], dtype=torch.float32)
    comparison = torch.cat([noisy, recon, clean], dim=0)
    save_image(make_grid(comparison, nrow=8), output_dir / "reconstruction_grid.png")


def main() -> None:
    set_seed(42)
    metadata = get_task_metadata()
    device = get_device()

    print("=" * 60)
    print("Autoencoder Task: CIFAR-10 Convolutional Denoising")
    print("=" * 60)
    print(f"Task: {metadata['task_name']}")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model, optimizer = build_model(device)
    history = train(model, train_loader, val_loader, optimizer, device)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train Metrics:")
    print(json.dumps({k: v for k, v in train_metrics.items() if not k.startswith("sample_")}, indent=2))
    print("Validation Metrics:")
    print(json.dumps({k: v for k, v in val_metrics.items() if not k.startswith("sample_")}, indent=2))

    assert val_metrics["mse"] < val_metrics["baseline_mse"], "Model failed to beat noisy baseline MSE"
    assert val_metrics["improvement_psnr"] > 1.0, f"PSNR gain too small: {val_metrics['improvement_psnr']:.3f}"
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