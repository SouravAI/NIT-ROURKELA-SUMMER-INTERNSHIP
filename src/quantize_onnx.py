from onnxruntime.quantization import quantize_dynamic, QuantType
import os

MODEL_PATHS = {
    "mobilenetv3_small": "models/mobilenetv3_small.onnx",
    "squeezenet": "models/squeezenet.onnx",
    "efficientnetv2_s": "models/efficientnetv2_s.onnx"
}

def quantize(model_name):
    input_path = MODEL_PATHS[model_name]
    output_path = input_path.replace(".onnx", "_quantized.onnx")

    print(f"⚙️ Quantizing {model_name}...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Quantized model saved as: {output_path}")
    print(f"📦 File size: {size_mb:.2f} MB\n")

if __name__ == "__main__":
    for name in MODEL_PATHS:
        quantize(name)
