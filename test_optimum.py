import modal

app = modal.App("test-optimum")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("optimum[onnxruntime]==1.16.1", "transformers==4.36.2")
)

@app.function(image=image)
def test_cli():
    import subprocess
    result = subprocess.run(["optimum-cli", "export", "onnx", "--help"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

@app.local_entrypoint()
def main():
    test_cli.remote()
