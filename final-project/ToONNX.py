import os
import torch
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from SoundClassifier import AudioCNN # Ensure this is imported

# Load model
# Note: Ensure the path is correct relative to where you run the script
model_path = os.path.join("final-project", "models", "cnn_model")
model = torch.load(model_path, weights_only=False) 
model.eval()

# Create export directory
save_dir = "final-project/models/onnx_cnn_model"
os.makedirs(save_dir, exist_ok=True)

# Dummy input matching your expected input dimensions
dummy = torch.randn(1, 1, 62, 32) 

# Export ONNX with Opset 13
onnx_path = os.path.join(save_dir, "cnn_model.onnx")

torch.onnx.export(
    model,
    (dummy,),
    onnx_path,
    input_names=["melspec"],
    output_names=["logits"],
    opset_version=13,
    dynamo=False,  # ← key fix: use legacy exporter
    dynamic_axes={
        'melspec': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)

print("Saved:", onnx_path)

# Quantize
quant_path = os.path.join(save_dir, "cnn_model_int8.onnx")
quantize_dynamic(
    model_input=onnx_path,
    model_output=quant_path,
    weight_type=QuantType.QInt8,
)
print("Saved quantized:", quant_path)