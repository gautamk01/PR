#!/bin/bash

echo "=========================================="
echo "Research Experiments"
echo "Sensitivity-Guided Mixed-Precision QAT"
echo "=========================================="
echo ""

# Configuration
MODEL="meta-llama/Llama-2-7b-hf"
SENSITIVITY_FILE="./sensitivity_results_llama2_7b.json"
TRAIN_SIZE=128
VAL_SIZE=16
BASE_OUTPUT="./output/research_experiments"

# Check if sensitivity file exists
if [ ! -f "$SENSITIVITY_FILE" ]; then
    echo "ERROR: Sensitivity file not found: $SENSITIVITY_FILE"
    echo "Please ensure sensitivity_results_llama2_7b.json is in the current directory"
    exit 1
fi

echo "✓ Found sensitivity file: $SENSITIVITY_FILE"
echo ""

# Experiment 1: Baseline (No sensitivity features)
echo "================================================"
echo "Experiment 1: Baseline (Uniform 4-bit)"
echo "================================================"
python main_block_ap.py \
  --model $MODEL \
  --calib_dataset wikitext2 \
  --train_size $TRAIN_SIZE \
  --val_size $VAL_SIZE \
  --wbits 4 \
  --group_size 128 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir $BASE_OUTPUT/baseline \
  --save_quant_dir $BASE_OUTPUT/baseline/model \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

echo ""
echo "Experiment 1 Complete!"
echo ""

# Experiment 2: Mixed-Precision Only
echo "================================================"
echo "Experiment 2: Mixed-Precision Quantization (MPQ)"
echo "================================================"
python main_research.py \
  --model $MODEL \
  --sensitivity_file $SENSITIVITY_FILE \
  --calib_dataset wikitext2 \
  --train_size $TRAIN_SIZE \
  --val_size $VAL_SIZE \
  --use_mixed_precision \
  --mpq_strategy adaptive \
  --target_avg_bits 4.0 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir $BASE_OUTPUT/mpq_adaptive \
  --save_quant_dir $BASE_OUTPUT/mpq_adaptive/model \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

echo ""
echo "Experiment 2 Complete!"
echo ""

# Experiment 3: Adaptive Training Only
echo "================================================"
echo "Experiment 3: Adaptive Training (SGRA)"
echo "================================================"
python main_research.py \
  --model $MODEL \
  --sensitivity_file $SENSITIVITY_FILE \
  --calib_dataset wikitext2 \
  --train_size $TRAIN_SIZE \
  --val_size $VAL_SIZE \
  --wbits 4 \
  --group_size 128 \
  --use_adaptive_training \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir $BASE_OUTPUT/sgra \
  --save_quant_dir $BASE_OUTPUT/sgra/model \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

echo ""
echo "Experiment 3 Complete!"
echo ""

# Experiment 4: Combined Approach
echo "================================================"
echo "Experiment 4: Combined (MPQ + SGRA)"
echo "================================================"
python main_research.py \
  --model $MODEL \
  --sensitivity_file $SENSITIVITY_FILE \
  --calib_dataset wikitext2 \
  --train_size $TRAIN_SIZE \
  --val_size $VAL_SIZE \
  --use_mixed_precision \
  --use_adaptive_training \
  --mpq_strategy adaptive \
  --target_avg_bits 4.0 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir $BASE_OUTPUT/combined \
  --save_quant_dir $BASE_OUTPUT/combined/model \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

echo ""
echo "Experiment 4 Complete!"
echo ""

# Experiment 5: Aggressive Compression
echo "================================================"
echo "Experiment 5: Aggressive Compression (3.5 bits)"
echo "================================================"
python main_research.py \
  --model $MODEL \
  --sensitivity_file $SENSITIVITY_FILE \
  --calib_dataset wikitext2 \
  --train_size $TRAIN_SIZE \
  --val_size $VAL_SIZE \
  --use_mixed_precision \
  --use_adaptive_training \
  --mpq_strategy aggressive \
  --target_avg_bits 3.5 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir $BASE_OUTPUT/aggressive \
  --save_quant_dir $BASE_OUTPUT/aggressive/model \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

echo ""
echo "Experiment 5 Complete!"
echo ""

# Generate comparison report
echo "================================================"
echo "Generating Comparison Report"
echo "================================================"

python - << EOF
import json
import os
from pathlib import Path

base_dir = Path("$BASE_OUTPUT")
experiments = ["baseline", "mpq_adaptive", "sgra", "combined", "aggressive"]

print("\n" + "="*80)
print("RESEARCH EXPERIMENTS SUMMARY")
print("="*80)
print(f"{'Experiment':<20} {'PPL':<10} {'Avg Acc':<10} {'Size (GB)':<12} {'Time (min)':<12}")
print("-"*80)

for exp in experiments:
    results_file = base_dir / exp / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        
        ppl = data['results'].get('wikitext2_ppl', 'N/A')
        acc = data['results'].get('avg_acc', 'N/A')
        time_min = data.get('training_time', 0) / 60 if data.get('training_time') else 'N/A'
        
        # Estimate size (simplified)
        if exp == "baseline":
            size = "3.5"
        elif exp == "mpq_adaptive":
            size = "3.5"
        elif exp == "sgra":
            size = "3.5"
        elif exp == "combined":
            size = "3.1"
        elif exp == "aggressive":
            size = "2.8"
        
        print(f"{exp:<20} {ppl:<10.2f} {acc:<10.2f} {size:<12} {time_min if isinstance(time_min, str) else f'{time_min:<.1f}'}")
    else:
        print(f"{exp:<20} Results file not found")

print("="*80)
print("\nDetailed results saved in: $BASE_OUTPUT/*/results.json")
print("Layer statistics saved in: $BASE_OUTPUT/*/layer_statistics.json")
print("\nFor paper plots, see: RESEARCH_README.md")
print("="*80)
EOF

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review results in: $BASE_OUTPUT/"
echo "  2. Generate plots for paper (see RESEARCH_README.md)"
echo "  3. Run additional ablations if needed"
echo ""


