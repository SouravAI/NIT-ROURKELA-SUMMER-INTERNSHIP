# src/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from dataset import PCGDataset, CLASS_NAMES
from model import get_model

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["batch_size"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "models/best_model.pt"

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate_model():
    # Load dataset
    full_dataset = PCGDataset()
    total_len = len(full_dataset)
    train_len = int(config["train_split"] * total_len)
    val_len = int(config["val_split"] * total_len)
    test_len = total_len - train_len - val_len

    _, _, test_ds = random_split(full_dataset, [train_len, val_len, test_len])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Load model
    model = get_model("mobilenetv3_small", num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            x = x.repeat(1, 3, 1, 1)  # convert to 3-channel
            y = y.to(DEVICE)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Classification report
    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, save_path="outputs/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
