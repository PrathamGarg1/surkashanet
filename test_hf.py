import argparse
from datasets import load_dataset

try:
    print("Testing Davidson...")
    d1 = load_dataset("hate_speech_offensive", split="train[:10]")
    print("Davidson cols:", d1.column_names)
    print("Davidson features:", d1.features)

    print("Testing L3Cube...")
    d2 = load_dataset("l3cube-pune/hinglish-hate", split="train[:10]")
    print("L3Cube cols:", d2.column_names)
except Exception as e:
    print("Error:", e)
