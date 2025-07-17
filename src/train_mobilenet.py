# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import yaml

from dataset import PCGDataset
from model import get_model

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["batch_size"]
EPOCHS = config["num_epochs"]
LR = config["learning_rate"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "models/best_model.pt"

def train():
    # Dataset
    full_dataset = PCGDataset()
    total_len = len(full_dataset)
    train_len = int(config["train_split"] * total_len)
    val_len = int(config["val_split"] * total_len)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = get_model("mobilenetv3_small", num_classes=5, pretrained=False)
    model.to(DEVICE)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.repeat(1, 3, 1, 1)  # repeat channel to RGB

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(model, val_loader)

        print(f"📈 Epoch {epoch}: Train Loss = {avg_loss:.4f} | Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            best_val_acc = val_acc
            print(f"✅ Saved best model @ epoch {epoch}")

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.repeat(1, 3, 1, 1)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

if __name__ == "__main__":
    train()
