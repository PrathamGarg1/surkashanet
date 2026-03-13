#!/bin/bash

# Ensure prerequisites are installed
# pip install torch transformers optimum[onnxruntime]

echo "==========================================================="
echo "  SurakshaNet: Tiny MACD ONNX Quantization Script"
echo "==========================================================="
echo ""
echo "This script converts the fine-tuned MiniLM PyTorch model into a highly compressed int8 ONNX format."
echo "The resulting model should be between 40MB - 60MB."
echo ""

# Point this to the output directory of your train_macd.py script
LOCAL_MODEL_DIR="./results_macd_minilm/final_model"
OUTPUT_ONNX_PATH="./dist_onnx_model_minilm"

if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    echo "❌ Error: '$LOCAL_MODEL_DIR' directory not found."
    echo "Please run 'python train_macd.py' to fine-tune your tiny model first."
    exit 1
fi

echo "🚀 Starting ONNX conversion and Int8 Quantization..."

# Use Optimum CLI to export the local model and apply INT8 quantization
optimum-cli export onnx \
  --model $LOCAL_MODEL_DIR \
  --task text-classification \
  --optimize 1 \
  --quantize int8 \
  --opset 14 \
  $OUTPUT_ONNX_PATH

echo "✅ Conversion complete! Checking file sizes:"
ls -lh $OUTPUT_ONNX_PATH

echo ""
echo "📦 Next step: Copy the entire contents of the '$OUTPUT_ONNX_PATH' folder into your extension at 'assets/models/custom-macd-model/'"
