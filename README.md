# 🫀 Lightweight Deep Learning Model for Resource-Constrained Devices to Classify Heart Sound Signals

This project is a full pipeline to classify **heart sound signals (PCG)** into five medical conditions using lightweight deep learning models optimized for **deployment on edge or resource-constrained devices**. Models are trained on mel-spectrograms derived from `.wav` PCG data and exported to ONNX with quantization and benchmarking support.

---

## 📂 Project Structure

```
pcg_classification_project/
├── data/                         # Contains GDrive link to dataset
│   └── gdrive_link.txt
├── models/                       # Saved ONNX + quantized models (auto-generated)
│   └── .temp.txt                 # Placeholder for GitHub (delete after clone)
├── outputs/                      # Evaluation reports (auto-generated)
│   └── .temp.txt                 # Placeholder (delete after clone)
├── processed_data/               # Mel spectrogram tensors (.pt files)
│   └── .temp.txt                 # Placeholder (delete after clone)
├── src/                          # All source code
│   ├── preprocess.py             # Preprocessing WAV to mel
│   ├── dataset.py                # PyTorch Dataset and Dataloader
│   ├── train_mobilenet.py        # Train MobileNetV3
│   ├── train_squeezenet.py       # Train SqueezeNet
│   ├── train_efficientnet.py     # Train EfficientNetV2-S
│   ├── evaluate_mobilenet.py     # Evaluation script (MobileNet)
│   ├── evaluate_squeezenet.py    # Evaluation script (SqueezeNet)
│   ├── evaluate_efficientnet.py  # Evaluation script (EfficientNet)
│   ├── export_onnx.py            # Export models to ONNX
│   ├── quantize_onnx.py          # Quantize ONNX models
│   ├── benchmark_onnx.py         # Benchmark ONNX inference speed
│   └── infer.py                  # Predict from a new .wav file
├── requirements.txt              # Python dependencies
└── README.md                     # You are here!
```

---

## 📥 Dataset Info

- **Classes**:
  - AS: Aortic Stenosis
  - MR: Mitral Regurgitation
  - MS: Mitral Stenosis
  - MVP: Mitral Valve Prolapse
  - N: Normal

📦 **Download Dataset**:  
Dataset is hosted externally (not included in repo).  
📁 Navigate to: `data/gdrive_link.txt` and open the GDrive link to download the PCG WAV files.

---

## ⚙️ Setup Instructions

### 1. 🔧 Environment Setup

```bash
pip install -r requirements.txt
```

### 2. 🎧 Preprocess WAV Data

Converts `.wav` files into mel-spectrogram `.pt` tensors:

```bash
python src/preprocess.py
```

---

## 🏋️ Model Training

Train models on your local machine:

```bash
python src/train_mobilenet.py
python src/train_squeezenet.py
python src/train_efficientnet.py
```

Each will auto-save the best model checkpoint in `models/`.

---

## 📊 Evaluation

Generate classification reports:

```bash
python src/evaluate_mobilenet.py
python src/evaluate_squeezenet.py
python src/evaluate_efficientnet.py
```

---

## 📦 ONNX Conversion + Optimization

### ➤ Export to ONNX:

```bash
python src/export_onnx.py
```

### ➤ Quantize for edge devices:

```bash
python src/quantize_onnx.py
```

---

## ⚡ Benchmark ONNX Models

Tests inference speed on CPU (or GPU if configured):

```bash
python src/benchmark_onnx.py
```

---

## 🧠 Inference Demo

Run live prediction on any new `.wav` file:

```bash
python src/infer.py --path data/AS/New_AS_001.wav
```

---

## 📈 Model Performance Summary

| Model            | Accuracy | Size (ONNX) | Quantized Size | Inference Time |
|------------------|----------|-------------|----------------|----------------|
| MobileNetV3      | 100%     | 5.82 MB     | 1.62 MB        | ~0.82 ms       |
| SqueezeNet       | 95%      | 2.83 MB     | 0.77 MB        | ~0.63 ms       |
| EfficientNetV2-S | 100%     | 76.8 MB     | 20.06 MB       | ~7.45 ms       |

> ✅ Inference time recorded on CPU using ONNX Runtime.  
> ✅ Accuracy based on evaluation over test split (20%).

---

## 🧼 Post-Clone Cleanup

GitHub doesn’t allow empty folders, so these `.temp.txt` placeholders exist inside:
- `models/`
- `outputs/`
- `processed_data/`

🔹 You can safely delete them after running the project once.

---

## 👨‍💻 Author

**Sourav Mahapatra**  
Internship ID: `CSINTERN/25/079`  
Supervisor: `Dr. Suchismita Chinara`

---

## 📃 License

MIT License — free for personal and academic use.  
Just drop a star ⭐ if it helped you!

---

## 💬 Final Words

This repo was built with passion, late-night grinding, and a mission to **bring healthcare intelligence to every corner**, even where resources are limited. If you learned something, consider sharing it forward 💖

---

🗓️ Generated on: July 17, 2025
