"""Evaluates the best checkpoint on the held-out test set, and derives the
UNCERTAIN confidence band from the validation set. Run from the project root:

    python -m src.evaluate --data-dir data/processed
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from src.dataset import get_dataloaders
from src.model import load_model

RESULTS_METRICS_DIR = "results/metrics"
RESULTS_PLOTS_DIR = "results/plots"


def get_probs_and_labels(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def find_confidence_threshold(val_probs, val_labels, target_accuracy=0.90):
    """Empirically derived UNCERTAIN cutoff (not an arbitrary constant):

    For each candidate confidence level t, keep only validation predictions whose
    confidence (distance from the decision boundary) is >= t, and measure accuracy
    on that subset. We pick the smallest t at which accuracy reaches target_accuracy.
    Below that confidence, predictions get labeled UNCERTAIN in the UI instead of a
    hard REAL/MANIPULATED call.
    """
    preds = (val_probs >= 0.5).astype(float)
    confidence = np.where(preds == 1, val_probs, 1 - val_probs)

    thresholds = np.arange(0.5, 1.0, 0.01)
    for t in thresholds:
        mask = confidence >= t
        if mask.sum() == 0:
            continue
        acc = accuracy_score(val_labels[mask], preds[mask])
        if acc >= target_accuracy:
            return float(t)
    return 0.5  # target accuracy never reached even at full confidence -- band disabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-path", default="models/best_model.pt")
    args = parser.parse_args()

    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    _, val_loader, test_loader = get_dataloaders(args.data_dir)

    val_probs, val_labels = get_probs_and_labels(model, val_loader, device)
    threshold = find_confidence_threshold(val_probs, val_labels)
    with open(os.path.join(RESULTS_METRICS_DIR, "uncertainty_threshold.json"), "w") as f:
        json.dump({"confidence_threshold": threshold, "derived_from": "validation set", "target_accuracy": 0.90}, f, indent=2)
    print(f"UNCERTAIN confidence threshold (derived from validation set): {threshold:.2f}")

    test_probs, test_labels = get_probs_and_labels(model, test_loader, device)
    test_preds = (test_probs >= 0.5).astype(float)

    metrics = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "precision": precision_score(test_labels, test_preds),
        "recall": recall_score(test_labels, test_preds),
        "f1_score": f1_score(test_labels, test_preds),
        "test_samples": len(test_labels),
    }
    with open(os.path.join(RESULTS_METRICS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "MANIPULATED"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(RESULTS_PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    print("Saved: test_metrics.json, uncertainty_threshold.json, confusion_matrix.png")


if __name__ == "__main__":
    main()
