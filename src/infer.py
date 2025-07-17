# src/infer.py

import torch
import librosa
import numpy as np
from model import get_model
from dataset import CLASS_NAMES
import sys
import argparse

# Same params as during training
SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 128
FMAX = 8000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "models/best_model.pt"

def preprocess_audio(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    expected_len = int(SAMPLE_RATE * DURATION)

    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    elif len(y) > expected_len:
        y = y[:expected_len]

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize + convert to tensor
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    tensor = torch.tensor(mel_db).unsqueeze(0)         # shape: [1, H, W]
    tensor = tensor.repeat(3, 1, 1).unsqueeze(0).to(DEVICE)  # shape: [1, 3, H, W]
    return tensor

def predict(file_path):
    # Load model
    model = get_model("mobilenetv3_small", num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Preprocess
    x = preprocess_audio(file_path)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    pred_class = CLASS_NAMES[pred_idx]
    print(f"🎧 File: {file_path}")
    print(f"🔍 Predicted class: {pred_class}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to WAV file")
    args = parser.parse_args()

    predict(args.path)
