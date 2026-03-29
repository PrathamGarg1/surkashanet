import modal
import os

# Define the Modal App
app = modal.App("surakshannet-macd-trainer")

# Define an Image with all our required dependencies
# We install torch, transformers, datasets, optimum, and onnxruntime
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.2",
        "datasets==2.16.1",
        "optimum[onnxruntime]==1.16.1",
        "accelerate==0.25.0",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.2",
        "onnx==1.16.0"
    )
)

# Configuration for the tiny model
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
EPOCHS = 3
BATCH_SIZE = 16

@app.function(
    image=training_image,
    gpu="A10G", # Use a fast GPU on Modal
    timeout=7200 # 2 hour timeout limit just in case
)
def train_and_export_model():
    """
    This function runs entirely on Modal's cloud GPUs.
    It downloads the dataset, trains the model, quantizes it to ONNX, 
    and returns the raw bytes of the final directory so we can save it locally.
    """
    import torch
    import shutil
    import subprocess
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )

    OUTPUT_DIR = "/tmp/results_macd_minilm"
    ONNX_OUTPUT_DIR = "/tmp/dist_onnx_model_minilm"

    print(f"☁️ [Modal GPU] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # We explicitly define the label mappings
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "abusive", 1: "non-abusive"},
        label2id={"abusive": 0, "non-abusive": 1}
    )

    print(f"☁️ [Modal GPU] Loading dataset from GitHub CSVs...")
    
    # The dataset isn't on the HF Hub, it's on their GitHub repo.
    # We download the raw CSVs directly.
    import urllib.request
    import pandas as pd
    from datasets import Dataset, DatasetDict
    import os

    # GitHub Raw URLs for the Hindi dataset splits
    github_base = "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/"
    splits = ["hindi_train", "hindi_val", "hindi_test"]
    
    dataset_dict = {}
    
    for split in splits:
        url = f"{github_base}{split}.csv"
        local_path = f"/tmp/{split}.csv"
        print(f"☁️ [Modal GPU] Downloading {split} set from {url}...")
        try:
            # Add headers to avoid 403s on raw.githubusercontent
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
                out_file.write(response.read())
                
            # Read CSV
            df = pd.read_csv(local_path)
            print(f"☁️ [Modal GPU] {split} successfully loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
            
            # Map common column names to 'text' and 'label' just in case
            if "comment_text" in df.columns: df = df.rename(columns={"comment_text": "text"})
            if "tweet" in df.columns: df = df.rename(columns={"tweet": "text"})
            if "class" in df.columns: df = df.rename(columns={"class": "label"})
            
            # Filter to just text and label, drop NaNs
            if "text" in df.columns and "label" in df.columns:
                df = df[["text", "label"]].dropna()
                # Cast to correct types
                df["text"] = df["text"].astype(str)
                df["label"] = pd.to_numeric(df["label"], errors='coerce').fillna(0).astype(int)
            else:
                print(f"❌ Warning: 'text' or 'label' missing in {split}. Using generic mapping if possible.")
                # Fallback: assume column 0 is label, column 1 is text (or vice versa based on MACD typical format)
                if len(df.columns) >= 2:
                    df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'})
            
            # Normalize the split name for HuggingFace (e.g., 'hindi_train' -> 'train')
            split_clean = split.replace("hindi_", "")
            hf_split_name = "train" if split_clean == "train" else "validation" if split_clean in ["val", "validation"] else "test"
            dataset_dict[hf_split_name] = Dataset.from_pandas(df)
            
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error for {split} at {url}: {e}")
        except Exception as e:
            print(f"❌ Error processing {split}: {e}")

    # Fallback if no dataset was loaded at all (e.g. 404s)
    if not dataset_dict:
        raise ValueError("Failed to download any datasets from GitHub. URLs might be wrong.")
        
    if "train" not in dataset_dict:
        print("☁️ [Modal GPU] 'train' split missing! Trying to create one from whatever downloaded...")
        # Just use whatever downloaded as train if possible
        first_split = list(dataset_dict.keys())[0]
        dataset_dict["train"] = dataset_dict[first_split]

    # --- Add prestigious academic datasets for English and Hinglish ---
    from datasets import load_dataset, concatenate_datasets
    print("☁️ [Modal GPU] Augmenting Hindi dataset with academic English & Hinglish datasets...")
    
    extra_train = []
    
    # 1. Davidson et al. 2017 (English Hate Speech)
    try:
        print("☁️ [Modal GPU] Downloading Davidson English dataset (hate_speech_offensive)...")
        en_ds = load_dataset("hate_speech_offensive", split="train")
        en_df = en_ds.to_pandas()
        # Labels: 0=hate speech, 1=offensive language, 2=neither
        # MACD format: 0=abusive, 1=non-abusive
        en_df["label"] = en_df["class"].apply(lambda x: 0 if x in [0, 1] else 1)
        en_df = en_df.rename(columns={"tweet": "text"})[["text", "label"]]
        # Ensure text is string and label is int
        en_df["text"] = en_df["text"].astype(str)
        en_df["label"] = en_df["label"].astype(int)
        extra_train.append(Dataset.from_pandas(en_df))
    except Exception as e:
        print(f"❌ Failed to load Davidson dataset: {e}")

    # 2. L3Cube-Pune (Hinglish Hate Speech)
    try:
        print("☁️ [Modal GPU] Downloading L3Cube-Pune Hinglish dataset (l3cube-pune/hinglish-hate)...")
        hi_ds = load_dataset("l3cube-pune/hinglish-hate", split="train")
        hi_df = hi_ds.to_pandas()
        # Labels in l3cube: 0=Non-Hate, 1=Hate
        # MACD format: 0=abusive, 1=non-abusive
        if "label" in hi_df.columns:
            hi_df["label"] = hi_df["label"].apply(lambda x: 1 if x == 0 else 0)
        hi_df = hi_df.rename(columns={"tweet": "text"})[["text", "label"]]
        hi_df["text"] = hi_df["text"].astype(str)
        hi_df["label"] = hi_df["label"].astype(int)
        extra_train.append(Dataset.from_pandas(hi_df))
    except Exception as e:
        print(f"❌ Failed to load L3Cube dataset: {e}")

    if "train" in dataset_dict and extra_train:
        print("☁️ [Modal GPU] Merging Hindi, English, and Hinglish training sets!")
        combined = concatenate_datasets([dataset_dict["train"]] + extra_train)
        dataset_dict["train"] = combined.shuffle(seed=42)
        print(f"☁️ [Modal GPU] New combined train set size: {len(dataset_dict['train'])}")
    # ----------------------------------------------------------------

    dataset = DatasetDict(dataset_dict)

    def preprocess_function(examples):
        # We need to map labels. According to the MACD paper: 0 = abusive, 1 = non-abusive.
        # Ensure text is cast to string just in case
        return tokenizer([str(t) for t in examples["text"]], padding="max_length", truncation=True, max_length=128)

    print("☁️ [Modal GPU] Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Some datasets have a dict with 'train', 'test', 'validation'.
    # If validation is missing, we split the train set.
    if "validation" not in tokenized_datasets:
        print("☁️ [Modal GPU] Creating a validation split...")
        split = tokenized_datasets["train"].train_test_split(test_size=0.1)
        tokenized_datasets["train"] = split["train"]
        tokenized_datasets["validation"] = split["test"]

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    print("☁️ [Modal GPU] Starting lightning-fast cloud training...")
    trainer.train()

    print(f"☁️ [Modal GPU] Saving fine-tuned model to {OUTPUT_DIR}/final_model")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    print(f"☁️ [Modal GPU] Training complete! Converting to INT8 Quantized ONNX (<50MB)...")
    
    # Export using Python API directly to avoid CLI argparse issues
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    import os
    
    # Load model from PyTorch training output, and convert to ONNX!
    ort_model = ORTModelForSequenceClassification.from_pretrained(f"{OUTPUT_DIR}/final_model", export=True)
    tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/final_model")
    
    # Save the base fp32 ONNX model temporarily to disk
    tmp_onnx_dir = f"{OUTPUT_DIR}/tmp_onnx"
    ort_model.save_pretrained(tmp_onnx_dir)
    tokenizer.save_pretrained(tmp_onnx_dir)
    
    # Now Quantize it to int8 for browser inference
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    
    # Save final INT8 ONNX files to ONNX_OUTPUT_DIR
    quantizer.quantize(save_dir=ONNX_OUTPUT_DIR, quantization_config=dqconfig)
    tokenizer.save_pretrained(ONNX_OUTPUT_DIR)
    
    print("☁️ [Modal GPU] ONNX Conversion Successful! Packaging files to send back to your Mac...")
    
    # Package the ONNX directory into a zip file in memory/tmp
    shutil.make_archive("/tmp/custom-macd-model", 'zip', ONNX_OUTPUT_DIR)
    
    # Read the zip file as raw bytes to return it over the network to the user's laptop
    with open("/tmp/custom-macd-model.zip", "rb") as f:
        zip_data = f.read()
        
    return zip_data

@app.local_entrypoint()
def main():
    import zipfile
    import io
    import os
    
    print("🚀 Connecting to Modal servers...")
    print("🚀 Offloading MACD AI training to cloud GPUs (This will be much faster!)...")
    
    # Execute the training function remotely on Modal
    zip_bytes_from_cloud = train_and_export_model.remote()
    
    print("\n✅ Training and compression finished on Modal!")
    print("⬇️ Downloading the fine-tuned, quantized ONNX model to your Mac...")
    
    # Define where we want to place it locally
    # We put it exactly where the extension needs it
    extension_model_path = os.path.join(os.getcwd(), "assets", "models", "custom-macd-model")
    os.makedirs(extension_model_path, exist_ok=True)
    
    # Extract the downloaded zip file directly into the correct folder
    with zipfile.ZipFile(io.BytesIO(zip_bytes_from_cloud)) as zip_ref:
        zip_ref.extractall(extension_model_path)
        
    print(f"\n🎉 SUCCESS! The fully trained, local offline model is now installed at: {extension_model_path}")
    print("You can verify the file sizes. It should be incredibly lightweight (under 60MB).")
    print("Just reload your SurakshaNet extension in Chrome!")
