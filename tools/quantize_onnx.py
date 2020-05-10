import onnx
from onnxruntime.quantize import quantize, QuantizationMode

# Load the onnx model
model = onnx.load('/home/lh/pretrain-models/pose_higher_hrnet_256_sim.onnx')
# Quantize
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
# Save the quantized model
onnx.save(quantized_model, '/home/lh/pretrain-models/pose_higher_hrnet_256_sim_int8.onnx')