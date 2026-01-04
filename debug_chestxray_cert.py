#!/usr/bin/env python
"""Debug chestxray certification pickle to understand the error."""
import pickle
from pathlib import Path
import json

pkl_path = Path("outputs/bulk_certifcation/chestxray/resnet18/results_20260103_053020.pkl")

if not pkl_path.exists():
    print(f"[ERROR] Pickle not found: {pkl_path}")
    exit(1)

print(f"Loading: {pkl_path}")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"\nPickle structure:")
print(f"  Type: {type(data)}")
print(f"  Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")

if isinstance(data, dict):
    for model_name, model_data in data.items():
        print(f"\n  Model: {model_name}")
        print(f"    Type: {type(model_data)}")
        if isinstance(model_data, dict):
            print(f"    Keys: {list(model_data.keys())[:5]}...")  # First 5 keys
            
            # Sample first entry
            if model_data:
                first_key = list(model_data.keys())[0]
                first_val = model_data[first_key]
                print(f"\n    Sample entry (key={first_key}):")
                print(f"      Type: {type(first_val)}")
                print(f"      Keys: {first_val.keys() if isinstance(first_val, dict) else 'N/A'}")
                
                if isinstance(first_val, dict):
                    for sub_key, sub_val in first_val.items():
                        if isinstance(sub_val, (list, tuple)):
                            print(f"        {sub_key}: {type(sub_val)} (len={len(sub_val)})")
                        else:
                            print(f"        {sub_key}: {type(sub_val)}")
                        if isinstance(sub_val, dict):
                            print(f"          Sub-keys: {sub_val.keys()}")
        elif isinstance(model_data, list):
            print(f"    List with {len(model_data)} entries")
            if model_data:
                print(f"    First entry type: {type(model_data[0])}")
