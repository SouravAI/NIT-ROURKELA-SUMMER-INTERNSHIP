# src/export_to_onnx.py

import torch
import numpy as np
import onnx
import onnxruntime as ort
import os

from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "mobilenetv3_small": {
        "ckpt": "models/best_mobilenetv3.pt",
        "onnx": "models/mobilenetv3_small.onnx"
    },
    "squeezenet": {
        "ckpt": "models/best_squeezenet.pt",
        "onnx": "models/squeezenet.onnx"
    },
    "efficientnetv2_s": {
        "ckpt": "models/best_efficientnet.pt",
        "onnx": "models/efficientnetv2_s.onnx"
    }
}

def export_model(model_name):
    print(f"\n🚀 Exporting: {model_name}")
    config = MODEL_CONFIGS[model_name]
    ckpt_path = config["ckpt"]
    onnx_path = config["onnx"]

    model = get_model(model_name, num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 63)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["mel_spectrogram"],
        output_names=["predictions"],
        dynamic_axes={"mel_spectrogram": {0: "batch_size"}, "predictions": {0: "batch_size"}},
        opset_version=16,
        export_params=True
    )

    print(f"✅ Exported to {onnx_path}")

    # Check + test inference
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {"mel_spectrogram": dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pred_class = np.argmax(ort_outs[0], axis=1)[0]
    print(f"🎯 Test prediction: class {pred_class}")

    # Show size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📦 File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    for model_name in MODEL_CONFIGS.keys():
        export_model(model_name)
