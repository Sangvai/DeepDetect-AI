"""Training loop. Run from the project root:

    python -m src.train --data-dir data/processed --epochs 15 --batch-size 32
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_model, save_checkpoint

RESULTS_METRICS_DIR = "results/metrics"
RESULTS_PLOTS_DIR = "results/plots"
MODELS_DIR = "models"


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4, help="epochs with no val improvement before early stopping")
    args = parser.parse_args()

    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, tqdm(train_loader, desc=f"Epoch {epoch} [train]"), criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, os.path.join(MODELS_DIR, "best_model.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val_loss improvement for {args.patience} epochs.")
                break

    with open(os.path.join(RESULTS_METRICS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    epochs_ran = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs_ran, history["train_loss"], label="Train Loss")
    plt.plot(epochs_ran, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_ran, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs_ran, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, "accuracy_curve.png"))
    plt.close()

    print(f"Done. Best model: {MODELS_DIR}/best_model.pt | History + plots saved.")


if __name__ == "__main__":
    main()
