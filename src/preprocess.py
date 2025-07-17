import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

# CONFIG
RAW_DATA_DIR = "data"
PROCESSED_DATA_DIR = "processed_data"
SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 128
FMAX = 8000
SAVE_AS = "npy"  # or "png"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_mel(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, fmax=FMAX):
    y, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    elif len(y) > expected_len:
        y = y[:expected_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)  # <-- FIXED!
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def save_mel(mel, save_path):
    if SAVE_AS == "npy":
        np.save(save_path, mel)
    elif SAVE_AS == "png":
        plt.imsave(save_path, mel, cmap='inferno')

def process_all():
    print(f"🎧 Starting preprocessing from {RAW_DATA_DIR}")
    class_folders = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

    for label in class_folders:
        input_folder = os.path.join(RAW_DATA_DIR, label)
        output_folder = os.path.join(PROCESSED_DATA_DIR, label)
        ensure_dir(output_folder)

        print(f"\n⚙️ Processing class: {label}")
        for fname in tqdm(os.listdir(input_folder)):
            if fname.endswith(".wav"):
                file_path = os.path.join(input_folder, fname)
                mel = extract_mel(file_path)

                out_name = fname.replace(".wav", f".{SAVE_AS}")
                save_path = os.path.join(output_folder, out_name)
                save_mel(mel, save_path)

    print("\n✅ All files processed and saved in 'processed_data/'.")

if __name__ == "__main__":
    process_all()
