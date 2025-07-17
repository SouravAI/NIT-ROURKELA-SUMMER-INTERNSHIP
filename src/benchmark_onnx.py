import time
import numpy as np
import onnxruntime as ort
import os

MODELS = {
    "MobileNetV3": "models/mobilenetv3_small.onnx",
    "SqueezeNet": "models/squeezenet.onnx",
    "EfficientNet": "models/efficientnetv2_s.onnx",
    "MobileNetV3-Quant": "models/mobilenetv3_small_quantized.onnx",
    "SqueezeNet-Quant": "models/squeezenet_quantized.onnx",
    "EfficientNet-Quant": "models/efficientnetv2_s_quantized.onnx"
}

INPUT_SHAPE = (1, 3, 128, 63)  # match your mel input shape

def benchmark(model_path, model_name, device="CPU", n_runs=100):
    print(f"\n🚀 Benchmarking {model_name} on {device}")
    
    providers = ["CPUExecutionProvider"] if device == "CPU" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})

    start = time.time()
    for _ in range(n_runs):
        _ = session.run(None, {input_name: dummy_input})
    end = time.time()

    avg_ms = ((end - start) / n_runs) * 1000
    print(f"🕒 Avg inference time: {avg_ms:.2f} ms")

if __name__ == "__main__":
    for model_name, path in MODELS.items():
        if os.path.exists(path):
            benchmark(path, model_name, device="CPU")
