#!/usr/bin/env python
"""Debug chestxray certification pickle to understand the error.

Adds a timeout so the terminal does not hang indefinitely during pickle load.
"""
import pickle
import signal
import time
from pathlib import Path

pkl_path = Path("outputs/bulk_certifcation/chestxray/resnet18/results_20260103_053020.pkl")

if not pkl_path.exists():
    print(f"[ERROR] Pickle not found: {pkl_path}")
    raise SystemExit(1)

file_size_mb = pkl_path.stat().st_size / (1024 * 1024)
print(f"Loading: {pkl_path} ({file_size_mb:.2f} MB)")


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out while loading pickle (possible corruption or huge file)")


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(30)  # 30s timeout to avoid hanging terminal
start = time.time()

try:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.2f}s")
finally:
    signal.alarm(0)

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
