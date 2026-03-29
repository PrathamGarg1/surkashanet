import modal
import os

# Define the Modal App
app = modal.App("surakshannet-macd-evaluator")

# Define an Image with all our required dependencies
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
    timeout=7200 # 2 hour timeout limit
)
def train_and_evaluate():
    """
    Trains the model (exact same as modal_train.py) and then evaluates:
    1. The fine-tuned PyTorch model on the test set.
    2. The quantized ONNX model on the same test set.
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from datasets import load_dataset, Dataset, DatasetDict
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from sklearn.metrics import accuracy_score, f1_score
    import urllib.request
    import pandas as pd
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    OUTPUT_DIR = "/tmp/results_eval"
    ONNX_OUTPUT_DIR = "/tmp/dist_onnx_eval"

    print(f"☁️ [Modal GPU] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "abusive", 1: "non-abusive"},
        label2id={"abusive": 0, "non-abusive": 1}
    )

    print(f"☁️ [Modal GPU] Downloading dataset from GitHub...")
    github_base = "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/"
    splits = ["hindi_train", "hindi_val", "hindi_test"]
    dataset_dict = {}
    
    for split in splits:
        url = f"{github_base}{split}.csv"
        local_path = f"/tmp/{split}.csv"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
            out_file.write(response.read())
        df = pd.read_csv(local_path)
        
        # Mapping columns
        if "comment_text" in df.columns: df = df.rename(columns={"comment_text": "text"})
        elif "tweet" in df.columns: df = df.rename(columns={"tweet": "text"})
        if "class" in df.columns: df = df.rename(columns={"class": "label"})
        
        df = df[["text", "label"]].dropna()
        df["text"] = df["text"].astype(str)
        df["label"] = pd.to_numeric(df["label"], errors='coerce').fillna(0).astype(int)
        
        split_clean = split.replace("hindi_", "")
        hf_split_name = "train" if split_clean == "train" else "validation" if split_clean in ["val", "validation"] else "test"
        dataset_dict[hf_split_name] = Dataset.from_pandas(df)

    # Fallback if no dataset was loaded at all (e.g. 404s)
    if not dataset_dict:
        raise ValueError("Failed to download any datasets from GitHub. URLs might be wrong.")
        
    if "train" not in dataset_dict:
        # Just use whatever downloaded as train if possible
        first_split = list(dataset_dict.keys())[0]
        dataset_dict["train"] = dataset_dict[first_split]

    # --- Add prestigious academic datasets for English and Hinglish ---
    from datasets import load_dataset, concatenate_datasets
    print("☁️ [Modal GPU] Augmenting Hindi dataset with academic English & Hinglish datasets...")
    
    extra_train = []
    extra_test = []
    
    # 1. Davidson et al. 2017 (English Hate Speech)
    try:
        en_ds = load_dataset("hate_speech_offensive")
        # Train split
        en_df_train = en_ds['train'].to_pandas()
        en_df_train["label"] = en_df_train["class"].apply(lambda x: 0 if x in [0, 1] else 1)
        en_df_train = en_df_train.rename(columns={"tweet": "text"})[["text", "label"]]
        en_df_train["text"] = en_df_train["text"].astype(str)
        en_df_train["label"] = en_df_train["label"].astype(int)
        
        # Split English train into 90% train, 10% test
        split_idx = int(0.9 * len(en_df_train))
        extra_train.append(Dataset.from_pandas(en_df_train.iloc[:split_idx]))
        if "test" in dataset_dict:
            extra_test.append(Dataset.from_pandas(en_df_train.iloc[split_idx:]))
            
    except Exception as e:
        print(f"❌ Failed to load Davidson dataset: {e}")

    # 2. L3Cube-Pune (Hinglish Hate Speech)
    try:
        hi_ds = load_dataset("l3cube-pune/hinglish-hate", split="train")
        hi_df = hi_ds.to_pandas()
        if "label" in hi_df.columns:
            hi_df["label"] = hi_df["label"].apply(lambda x: 1 if x == 0 else 0)
        hi_df = hi_df.rename(columns={"tweet": "text"})[["text", "label"]]
        hi_df["text"] = hi_df["text"].astype(str)
        hi_df["label"] = hi_df["label"].astype(int)
        
        split_hi_idx = int(0.9 * len(hi_df))
        extra_train.append(Dataset.from_pandas(hi_df.iloc[:split_hi_idx]))
        if "test" in dataset_dict:
            extra_test.append(Dataset.from_pandas(hi_df.iloc[split_hi_idx:]))
            
    except Exception as e:
        print(f"❌ Failed to load L3Cube dataset: {e}")

    if "train" in dataset_dict and extra_train:
        combined = concatenate_datasets([dataset_dict["train"]] + extra_train)
        dataset_dict["train"] = combined.shuffle(seed=42)
        
    if "test" in dataset_dict and extra_test:
        combined_test = concatenate_datasets([dataset_dict["test"]] + extra_test)
        dataset_dict["test"] = combined_test.shuffle(seed=42)

    # Confirm isolation
    print(f"✅ Training set size: {len(dataset_dict['train'])}")
    if "test" in dataset_dict:
        print(f"✅ Test set size: {len(dataset_dict['test'])}")
    print("🚀 Confirmation: The 'test' split will only be used for final evaluation.")

    dataset = DatasetDict(dataset_dict)

    def preprocess_function(examples):
        return tokenizer([str(t) for t in examples["text"]], padding="max_length", truncation=True, max_length=128)

    print("☁️ [Modal GPU] Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
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

    print("☁️ [Modal GPU] Starting training...")
    trainer.train()
    
    # 1. Evaluate PyTorch Model on Test Set
    print("\n📊 Evaluating PyTorch Fine-tuned Model on Test Set...")
    pytorch_metrics = trainer.evaluate(tokenized_datasets["test"])
    print(f"✅ PyTorch Test Results: {pytorch_metrics}")

    # Manual check for Accuracy/F1 on test set for clarity
    preds = trainer.predict(tokenized_datasets["test"])
    labels = preds.label_ids
    pred_labels = np.argmax(preds.predictions, axis=1)
    pt_acc = accuracy_score(labels, pred_labels)
    pt_f1 = f1_score(labels, pred_labels, average='macro')
    print(f"🔥 PyTorch Manual Accuracy: {pt_acc:.4f}")
    print(f"🔥 PyTorch Manual F1 (Macro): {pt_f1:.4f}")

    # 2. Convert and Quantize to ONNX
    print("\n☁️ Converting to INT8 Quantized ONNX...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    ort_model = ORTModelForSequenceClassification.from_pretrained(f"{OUTPUT_DIR}/final_model", export=True)
    tmp_onnx_dir = f"{OUTPUT_DIR}/tmp_onnx"
    ort_model.save_pretrained(tmp_onnx_dir)
    
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=ONNX_OUTPUT_DIR, quantization_config=dqconfig)
    tokenizer.save_pretrained(ONNX_OUTPUT_DIR)

    # 3. Evaluate ONNX Model on Test Set
    print("\n📊 Evaluating Quantized ONNX Model on Test Set...")
    # Load the quantized model
    ort_model_quant = ORTModelForSequenceClassification.from_pretrained(ONNX_OUTPUT_DIR)
    
    onnx_preds = []
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    
    print(f"Running inference on {len(test_texts)} test samples...")
    # Batch inference for speed
    for i in tqdm(range(0, len(test_texts), BATCH_SIZE)):
        batch_texts = test_texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = ort_model_quant(**inputs)
            logits = outputs.logits
            onnx_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    onnx_acc = accuracy_score(test_labels, onnx_preds)
    onnx_f1 = f1_score(test_labels, onnx_preds, average='macro')
    
    print("\n" + "="*50)
    print("       FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"PyTorch Accuracy:  {pt_acc:.4f}")
    print(f"PyTorch F1 Macro:  {pt_f1:.4f}")
    print("-" * 30)
    print(f"ONNX INT8 Acc:     {onnx_acc:.4f}")
    print(f"ONNX INT8 F1 Macro: {onnx_f1:.4f}")
    print("-" * 30)
    print(f"Accuracy Difference: {onnx_acc - pt_acc:.4f}")
    print("="*50)

    return {
        "pytorch": {"accuracy": pt_acc, "f1": pt_f1},
        "onnx": {"accuracy": onnx_acc, "f1": onnx_f1}
    }

@app.local_entrypoint()
def main():
    print("🚀 Connecting to Modal servers for evaluation...")
    results = train_and_evaluate.remote()
    print("\n✅ Evaluation complete!")
