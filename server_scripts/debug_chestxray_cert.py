#!/usr/bin/env python
"""Debug chestxray certification pickle to understand the error.

Windows-compatible version - checks file size first.
"""
import pickle
import time
from pathlib import Path

pkl_path = Path("outputs/bulk_certifcation/chestxray/resnet18/results_20260103_053020.pkl")

if not pkl_path.exists():
    print(f"[ERROR] Pickle not found: {pkl_path}")
    raise SystemExit(1)

file_size_mb = pkl_path.stat().st_size / (1024 * 1024)
print(f"File: {pkl_path}")
print(f"Size: {file_size_mb:.2f} MB")

if file_size_mb > 500:
    print(f"[WARN] File is very large ({file_size_mb:.2f} MB). This may take a while or hang.")
    print("Consider using a smaller sample or re-running certification.")
    raise SystemExit(1)

print(f"\nLoading pickle (may take 10-60s for large files)...")
start = time.time()

try:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    load_time = time.time() - start
    print(f"✓ Loaded in {load_time:.2f}s")
except Exception as e:
    print(f"[ERROR] Failed to load pickle: {e}")
    raise SystemExit(1)

print(f"\nPickle structure:")
print(f"  Type: {type(data)}")
print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

if isinstance(data, dict):
    for model_name, model_data in data.items():
        print(f"\n  Model: {model_name}")
        print(f"    Type: {type(model_data)}")
        if isinstance(model_data, dict):
            keys = list(model_data.keys())
            print(f"    Number of method keys: {len(keys)}")
            print(f"    First 3 keys: {keys[:3]}")
            
            # Check first method
            if keys:
                first_key = keys[0]
                first_val = model_data[first_key]
                print(f"\n    Sample method (key={first_key}):")
                print(f"      Type: {type(first_val)}")
                if isinstance(first_val, dict):
                    k_keys = list(first_val.keys())
                    print(f"      K values: {k_keys}")
                    if k_keys:
                        first_k = k_keys[0]
                        k_data = first_val[first_k]
                        print(f"\n      Sample K={first_k}:")
                        print(f"        Type: {type(k_data)}")
                        if isinstance(k_data, list):
                            print(f"        List length: {len(k_data)}")
                            if k_data:
                                print(f"        First entry type: {type(k_data[0])}")
                                if isinstance(k_data[0], dict):
                                    print(f"        First entry keys: {list(k_data[0].keys())[:5]}")
        elif isinstance(model_data, list):
            print(f"    List with {len(model_data)} entries")
            if model_data:
                print(f"    First entry type: {type(model_data[0])}")

print("\n✓ Pickle structure analyzed successfully")

print(f"\nPickle structure:")
print(f"  Type: {type(data)}")
print(f"  Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")

if isinstance(data, dict):
    for model_name, model_data in data.items():
        print(f"\n  Model: {model_name}")
        print(f"    Type: {type(model_data)}")
        if isinstance(model_data, dict):
            keys = list(model_data.keys())
            print(f"    Keys count: {len(keys)}")
            print(f"    First keys: {keys[:5]}")
            
            if model_data:
                first_key = keys[0]
                first_val = model_data[first_key]
                print(f"\n    Sample entry (key={first_key}):")
                print(f"      Type: {type(first_val)}")
                if isinstance(first_val, dict):
                    print(f"      Keys: {list(first_val.keys())}")
                    for sub_key, sub_val in first_val.items():
                        if isinstance(sub_val, (list, tuple)):
                            print(f"        {sub_key}: {type(sub_val)} (len={len(sub_val)})")
                        else:
                            print(f"        {sub_key}: {type(sub_val)}")
                else:
                    print(f"      Value preview: {first_val}")
        elif isinstance(model_data, list):
            print(f"    List with {len(model_data)} entries")
            if model_data:
                print(f"    First entry type: {type(model_data[0])}")
