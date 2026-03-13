import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Configuration
# We use a tiny Multilingual MiniLM model. 
# Base size is ~471MB in PyTorch, but shrinks to ~40-60MB when ONNX int8 quantized!
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
DATASET_NAME = "ShareChat/macd" 
OUTPUT_DIR = "./results_macd_minilm"
EPOCHS = 3
BATCH_SIZE = 16

def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # We are doing binary classification: "Abusive" (0) vs "Non-Abusive" (1) based on MACD labels
    # We explicitly define the label mappings so the ONNX export knows them
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "abusive", 1: "non-abusive"},
        label2id={"abusive": 0, "non-abusive": 1}
    )

    print(f"Loading dataset: {DATASET_NAME}")
    # Load the specific subset for Hindi/Hinglish if available, or the whole dataset.
    # Adjust the subset name based on actual ShareChat dataset structure on HF.
    dataset = load_dataset(DATASET_NAME, "hi") 
    
    # Preprocessing function
    def preprocess_function(examples):
        # Assuming the text column is 'text' and label column is 'label'
        # Labels need to be mapped to binary if they are multi-class originally
        # e.g., mapping severe abuse to 1, rest to 0
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Filter out columns not needed by the model
    # Assuming 'label' column is formatted correctly as 0 or 1.
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    # Define training arguments
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

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], # Ensure validation split exists
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model to {OUTPUT_DIR}/final_model")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print("Training complete!")

if __name__ == "__main__":
    main()
