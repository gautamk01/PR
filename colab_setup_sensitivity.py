#!/usr/bin/env python3
"""
Google Colab setup script for sensitivity-based training
Run this BEFORE training to ensure sensitivity file is in the right place
"""
import os
import json
import shutil

print("=" * 70)
print("SENSITIVITY FILE SETUP FOR GOOGLE COLAB")
print("=" * 70)

# Check current working directory
cwd = os.getcwd()
print(f"\n1. Current working directory: {cwd}")

# Define possible locations for sensitivity file
possible_locations = [
    "sensitivity_results_llama2_7b.json",  # Same directory
    "../sensitivity_results_llama2_7b.json",  # Parent directory
    "/content/sensitivity_results_llama2_7b.json",  # Colab root
    "/content/EfficientQAT/sensitivity_results_llama2_7b.json",
    "./cache/sensitivity_results_llama2_7b.json",
]

# Find the file
found_path = None
for path in possible_locations:
    if os.path.exists(path):
        found_path = path
        print(f"\n2. ✓ Found sensitivity file at: {path}")
        break

if not found_path:
    print("\n2. ✗ Sensitivity file NOT found in any expected location!")
    print("\nExpected locations checked:")
    for path in possible_locations:
        print(f"   - {path}")
    print("\n" + "=" * 70)
    print("ACTION REQUIRED:")
    print("=" * 70)
    print("Upload 'sensitivity_results_llama2_7b.json' to Colab:")
    print("1. Click the folder icon on the left sidebar")
    print("2. Click 'Upload to session storage'")
    print("3. Select the sensitivity_results_llama2_7b.json file")
    print("4. Re-run this script")
    print("=" * 70)
    exit(1)

# Verify file content
try:
    with open(found_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n3. ✓ File is valid JSON")
    print(f"   - Model: {data.get('model', {}).get('name', 'Unknown')}")
    print(f"   - Method: {data.get('method', 'Unknown')}")
    print(f"   - Layers: {len(data.get('sensitivity_scores', []))}")
    print(f"   - Score range: {min(data['sensitivity_scores']):.2f} to {max(data['sensitivity_scores']):.2f}")
    
except Exception as e:
    print(f"\n3. ✗ Error reading file: {e}")
    exit(1)

# Copy to current directory if not already there
target_path = "./sensitivity_results_llama2_7b.json"
if found_path != target_path and not os.path.exists(target_path):
    shutil.copy(found_path, target_path)
    print(f"\n4. ✓ Copied to: {target_path}")
else:
    print(f"\n4. ✓ File already in place: {target_path}")

# Verify the file will be found during training
print(f"\n5. ✓ File will be accessible during training")
print(f"   - Absolute path: {os.path.abspath(target_path)}")

# Print recommended command
print("\n" + "=" * 70)
print("READY TO TRAIN!")
print("=" * 70)
print("\nRun this command:")
print("""
!python main_block_ap_sensitivity.py \\
  --model meta-llama/Llama-2-7b-hf \\
  --calib_dataset wikitext2 \\
  --train_size 128 \\
  --val_size 16 \\
  --wbits 4 \\
  --group_size 64 \\
  --quant_lr 1e-4 \\
  --weight_lr 2e-5 \\
  --real_quant \\
  --output_dir ./output/sensitivity_logs/w4g64_train128 \\
  --save_quant_dir ./output/sensitivity_models/w4g64_train128 \\
  --sensitivity_file ./sensitivity_results_llama2_7b.json \\
  --adaptive_epochs
""")
print("\nNote: Changed path from '../' to './' (current directory)")
print("=" * 70)


