with open("modal_train.py", "r") as f:
    content = f.read()

# I realize the error was likely due to a bug in optimum or simply because we need to pass `--task text-classification` instead of it failing.
# Actually, the problem was that `optimum-cli` tries to do sub-process calls and sometimes fails with 2 if the flags are malformed. 
# Wait, look at the error from modal_train.py:
# Command '['optimum-cli', 'export', 'onnx', '--model', '/tmp/results_macd_minilm/final_model', '--task', 'text-classification', '--optimize', '1', '--quantize', 'int8', '--opset', '14', '/tmp/dist_onnx_model_minilm']' returned non-zero exit status 2.
# Exit status 2 in `optimum-cli` usually means unrecognized arguments. 
# Let's check `optimum-cli export onnx --help`.
# Wait, I don't need `--optimize 1` and `--quantize int8` both via CLI in older versions, maybe they changed it?
# Actually, the argument `--optimize` is usually `-O1` or `-O2` or `--optimize O1`.
# Let's fix the optimize flag to `-O1` instead of `--optimize 1`.
# Let's also set `--monolith` or something if needed. 
# Or we can just do `--optimize O1`

content = content.replace('"--optimize", "1"', '"-O1"')
# Note: actually, the flag is usually `-O1` or `-O2` for optimization in ONNX.
# Or we can just remove optimization/quantization from CLI and do it manually, or just use `-O1`.
# The flag in Optimum for optimization is `-O1`, `-O2`, `-O3`, `-O4`. Let's use `-O1` instead of `--optimize 1`.

# Let's also check `--quantize`. The flag is often `--trust-remote-code` etc.
# Actually let's just use Python's optimum library directly instead of CLI to be safe and avoid argparse errors!

new_export_code = """
    print(f"☁️ [Modal GPU] Training complete! Converting to INT8 Quantized ONNX (<50MB)...")
    
    # Export using Python API directly to avoid CLI argparse issues
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    import os
    
    # Export and Optimize
    ort_model = ORTModelForSequenceClassification.from_pretrained(f"{OUTPUT_DIR}/final_model", export=True)
    tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/final_model")
    
    # Save the base ONNX model
    ort_model.save_pretrained(ONNX_OUTPUT_DIR)
    tokenizer.save_pretrained(ONNX_OUTPUT_DIR)
    
    # Now Quantize it to int8
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=ONNX_OUTPUT_DIR, quantization_config=dqconfig)
    
    # We now have int8 ONNX models in ONNX_OUTPUT_DIR!
"""

# Replace the subprocess call with pure python
content = content.replace("""
    print(f"☁️ [Modal GPU] Training complete! Converting to INT8 Quantized ONNX (<50MB)...")
    
    # Run Optimum CLI via subprocess to do the conversion on the cloud machine
    export_command = [
        "optimum-cli", "export", "onnx",
        "--model", f"{OUTPUT_DIR}/final_model",
        "--task", "text-classification",
        "--optimize", "1",
        "--quantize", "int8",
        "--opset", "14",
        ONNX_OUTPUT_DIR
    ]
    
    subprocess.run(export_command, check=True)
""", new_export_code)

with open("modal_train.py", "w") as f:
    f.write(content)

