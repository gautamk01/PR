#!/bin/bash

# Run baseline EfficientQAT with sensitivity-based adaptive epochs
# This is the WORKING code with MINIMAL modifications

CUDA_VISIBLE_DEVICES=0 python main_block_ap_sensitivity.py \
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

