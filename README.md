# 🫀 Lightweight Deep Learning Model for Resource-Constrained Devices to Classify Heart Sound Signals

This project is a full pipeline to classify **heart sound signals (PCG)** into five medical conditions using lightweight deep learning models optimized for **deployment on edge or resource-constrained devices**. Models are trained on mel-spectrograms derived from `.wav` PCG data and exported to ONNX with quantization and benchmarking support.

---

## 📂 Project Structure

```
pcg_classification_project/
│
├── data/ # ⚠️ Not included — place your dataset here
│ └── dataset_link.txt # 🔗 Link to Google Drive for downloading WAV files
│
├── models/ # Trained & exported ONNX/quantized models (auto-created)
│ └── .temp # (To be deleted — just a placeholder for GitHub)
│
├── outputs/ # Evaluation reports, prediction logs, confusion matrix, etc.
│ └── .temp # (To be deleted)
│
├── processed_data/ # Preprocessed mel spectrograms (auto-generated)
│ └── .temp # (To be deleted)
│
├── src/ # All source code
│ ├── preprocess.py
│ ├── dataset.py
│ ├── train_mobilenet.py
│ ├── train_squeezenet.py
│ ├── train_efficientnet.py
│ ├── evaluate_mobilenet.py
│ ├── evaluate_squeezenet.py
│ ├── evaluate_efficientnet.py
│ ├── infer.py
│ ├── export_onnx.py
│ ├── quantize.py
│ └── benchmark_onnx.py
│
├── config.yaml # 🧠 Configuration file for hyperparameters and preprocessing
├── README.md # 📄 You're reading this
└── requirements.txt # Python dependencies
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
⚠️ Create your own `data/` folder if cloning this repo; not included due to size.

## 🔧 Configuration (config.yaml)

All the important training parameters and preprocessing constants are centralized in `config.yaml`. Example:

```yaml
# Training Config
batch_size: 32
num_epochs: 30
learning_rate: 0.001
sample_rate: 16000
n_mels: 128
fmax: 8000
segment_duration: 2.0
train_split: 0.8
val_split: 0.1
test_split: 0.1
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
