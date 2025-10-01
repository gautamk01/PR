#!/bin/bash

echo "=========================================="
echo "LayerWise-QAT with Sensitivity"
echo "Quick Start Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Checking sensitivity file...${NC}"
if [ ! -f "../sensitivity_results_llama2_7b.json" ]; then
    echo -e "${RED}ERROR: sensitivity_results_llama2_7b.json not found!${NC}"
    echo "Please ensure the file exists at: ../sensitivity_results_llama2_7b.json"
    exit 1
else
    echo -e "${GREEN}✓ Sensitivity file found${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Running sensitivity-based training...${NC}"
echo "This will take approximately 35-40 minutes for 128 training samples"
echo ""

python main_block_ap_sensitivity.py \
  --model meta-llama/Llama-2-7b-hf \
  --calib_dataset wikitext2 \
  --train_size 128 \
  --val_size 16 \
  --wbits 4 \
  --group_size 64 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir ./output/sensitivity_logs/w4g64_train128 \
  --save_quant_dir ./output/sensitivity_models/w4g64_train128 \
  --sensitivity_file ../sensitivity_results_llama2_7b.json \
  --adaptive_epochs \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

echo ""
echo -e "${GREEN}=========================================="
echo "Training Complete!"
echo "==========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Logs: ./output/sensitivity_logs/w4g64_train128"
echo "  - Model: ./output/sensitivity_models/w4g64_train128"
echo ""
echo "Check the logs for:"
echo "  ✓ Perplexity (target: 8-12 PPL)"
echo "  ✓ Accuracy (target: 67-72%)"
echo "  ✓ No NaN gradients"
echo "  ✓ Adaptive epochs per layer"



