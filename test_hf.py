from datasets import load_dataset

try:
    print("Testing Davidson et al. 2017 (hate_speech_offensive)...")
    d1 = load_dataset("hate_speech_offensive", split="train[:10]")
    print("Davidson cols:", d1.column_names)
    print("Davidson features:", d1.features)
except Exception as e:
    print("Error:", e)
