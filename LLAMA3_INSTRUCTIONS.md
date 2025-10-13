# Llama 3.1 Experiment Instructions

This guide explains how to run experiments with Llama 3.1 models using the `main_research_llama3.py` script, which includes compatibility fixes for Llama 3.1's rope_scaling configuration.

## Prerequisites: Generate Sensitivity Scores

**IMPORTANT**: Before running mixed-precision quantization, you must first generate sensitivity scores:

```bash
python generate_sensitivity_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --num_samples 128 \
    --output_file "./sensitivity_results_llama_3.1_8b.json"
```

This will:
- Analyze which layers are more sensitive to quantization
- Generate `sensitivity_results_llama_3.1_8b.json` file
- Take ~5-10 minutes depending on GPU

**Skip this step only if** you already have the sensitivity file or are not using `--use_mixed_precision`.

## Quick Start

Use `main_research_llama3.py` instead of `main_research.py` for all Llama 3.x models.

### Complete Workflow (Two Steps)

**Step 1: Generate Sensitivity Scores** (Required for MPQ)
```bash
python generate_sensitivity_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --num_samples 128 \
    --output_file "./sensitivity_results_llama_3.1_8b.json"
```
⏱️ Takes ~5-10 minutes

**Step 2: Run Mixed-Precision Quantization**
```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --use_mixed_precision \
    --mpq_strategy aggressive \
    --target_avg_bits 4 \
    --real_quant \
    --train_size 128 \
    --val_size 16 \
    --quant_lr 1e-4 \
    --weight_lr 1e-5 \
    --output_dir "./output/llama3_mpq_4bit" \
    --save_quant_dir "./output/llama3_mpq_4bit/model" \
    --eval_ppl \
    --eval_tasks "piqa,arc_easy,hellaswag,winogrande"
```

## Experiment Types

### 1. Mixed-Precision Quantization (MPQ)

Layer-specific bit-widths based on sensitivity analysis:

```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --use_mixed_precision \
    --mpq_strategy adaptive \
    --target_avg_bits 4.0 \
    --real_quant \
    --quant_lr 1e-4 \
    --weight_lr 1e-5 \
    --output_dir "./output/llama3_mpq_adaptive" \
    --save_quant_dir "./output/llama3_mpq_adaptive/model" \
    --eval_ppl \
    --eval_tasks "piqa,arc_easy"
```

**MPQ Strategies:**
- `adaptive`: Balanced approach (default)
- `aggressive`: More aggressive quantization, lower bits for less sensitive layers
- `conservative`: Conservative quantization, higher bits overall

### 2. Sensitivity-Guided Resource Allocation (SGRA)

Adaptive training resources based on sensitivity:

```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --use_adaptive_training \
    --real_quant \
    --output_dir "./output/llama3_sgra" \
    --eval_ppl
```

### 3. Quantization Budget Optimization (QBO)

Fixed size budget with optimal quality:

```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --target_size_mb 3500 \
    --real_quant \
    --output_dir "./output/llama3_qbo" \
    --eval_ppl
```

### 4. Combined (All Features)

All research contributions combined:

```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --use_mixed_precision \
    --use_adaptive_training \
    --mpq_strategy aggressive \
    --target_avg_bits 3.5 \
    --target_size_mb 3500 \
    --real_quant \
    --train_size 256 \
    --val_size 32 \
    --output_dir "./output/llama3_full" \
    --save_quant_dir "./output/llama3_full/model" \
    --eval_ppl \
    --eval_tasks "piqa,arc_easy,arc_challenge,hellaswag,winogrande"
```

## Important Parameters

### Model & Data
- `--model`: Model name or path (e.g., `meta-llama/Llama-3.1-8B`)
- `--sensitivity_file`: Path to sensitivity results JSON (required)
- `--train_size`: Number of training samples (default: 128)
- `--val_size`: Number of validation samples (default: 16)
- `--training_seqlen`: Training sequence length (default: 2048)

### Quantization
- `--wbits`: Base bit-width (default: 4, overridden by MPQ)
- `--group_size`: Group size for quantization (default: 128)
- `--real_quant`: Enable real quantization (recommended)

### Training
- `--quant_lr`: Learning rate for quantization parameters (default: 1e-4)
- `--weight_lr`: Learning rate for weights (default: 2e-5)
  - **Important**: Use `2e-5` for 2-bit, `1e-5` for 3-bit/4-bit quantization
  - For 4-bit (your case), use: `--weight_lr 1e-5`
- `--epochs`: Number of training epochs (default: 2)
- `--batch_size`: Batch size (default: 2)

### Research Features
- `--use_mixed_precision`: Enable MPQ
- `--mpq_strategy`: MPQ allocation strategy (adaptive/aggressive/conservative)
- `--target_avg_bits`: Target average bit-width for MPQ
- `--use_adaptive_training`: Enable SGRA
- `--target_size_mb`: Target model size for QBO

### Evaluation
- `--eval_ppl`: Evaluate perplexity on wikitext2
- `--eval_tasks`: Comma-separated list of evaluation tasks
  - Available: `piqa`, `arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`, `mmlu`, etc.
- `--eval_batch_size`: Batch size for evaluation (default: 16)

### Output
- `--output_dir`: Directory for logs and results
- `--save_quant_dir`: Directory to save quantized model
- `--cache_dir`: Directory for cached data (default: ./cache)

## Ablation Studies

Run specific ablation experiments:

```bash
# MPQ only
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --ablation mpq_only \
    --real_quant \
    --output_dir "./output/llama3_ablation_mpq"

# SGRA only
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --ablation sgra_only \
    --real_quant \
    --output_dir "./output/llama3_ablation_sgra"

# QBO only
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_8b.json" \
    --ablation qbo_only \
    --target_size_mb 3500 \
    --real_quant \
    --output_dir "./output/llama3_ablation_qbo"
```

## Different Llama 3.1 Model Sizes

### Llama 3.1 70B

```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-70B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_70b.json" \
    --use_mixed_precision \
    --mpq_strategy aggressive \
    --target_avg_bits 3.5 \
    --real_quant \
    --train_size 64 \
    --val_size 8 \
    --max_memory "80GiB" \
    --output_dir "./output/llama3_70b_mpq" \
    --eval_ppl
```

### Llama 3.1 405B (Requires Multi-GPU)

```bash
python main_research_llama3.py \
    --model "meta-llama/Llama-3.1-405B" \
    --sensitivity_file "./sensitivity_results_llama_3.1_405b.json" \
    --use_mixed_precision \
    --mpq_strategy aggressive \
    --target_avg_bits 3.0 \
    --real_quant \
    --train_size 32 \
    --val_size 4 \
    --max_memory "80GiB" \
    --output_dir "./output/llama3_405b_mpq"
```

## What Changed from main_research.py?

The new `main_research_llama3.py` includes:

1. **`load_llama3_config()` function**: Handles Llama 3.1's extended rope_scaling format
2. **Automatic compatibility fix**: 
   - Converts Llama 3.1's extended rope_scaling (with `rope_type='llama3'`) to transformers 4.40.1 compatible format
   - Maps `'llama3'` rope type → `'linear'` (supported by transformers 4.40.1)
   - Preserves the factor value from original config
3. **Same features**: All research contributions (MPQ, SGRA, QBO) work identically

## Learning Rate Recommendations

Based on extensive experiments in the codebase:

| Target Bits | quant_lr | weight_lr | Notes |
|-------------|----------|-----------|-------|
| 2-bit | 1e-4 | 2e-5 | Higher weight_lr for aggressive quantization |
| 3-bit | 1e-4 | 1e-5 | Balanced learning rate |
| 4-bit | 1e-4 | 1e-5 | Standard setting (your case) |
| Mixed (avg 3.5-4) | 1e-4 | 1e-5 | Use 4-bit settings |
| Mixed (avg <3.5) | 1e-4 | 2e-5 | More aggressive, use 2-bit settings |

**Why it matters**: Lower bit-widths require higher `weight_lr` to compensate for more aggressive quantization during optimization.

## Transformers Version Compatibility

### Current Setup (Recommended)
**No upgrade needed!** The `main_research_llama3.py` script works with your current setup:
- ✅ **transformers==4.40.1** (from requirements.txt)
- ✅ Built-in compatibility layer handles Llama 3.1's rope_scaling automatically
- ✅ Preserves compatibility with the rest of your quantization codebase

### Alternative: Upgrade Transformers (Optional)
If you prefer native Llama 3.1 support without compatibility layer:

```bash
pip install --upgrade transformers
```

This will install transformers >= 4.43.0 which has native Llama 3.1 support. However:
- ⚠️ May require testing with the quantization code
- ⚠️ Could have breaking changes for other parts of your codebase
- ⚠️ The compatibility fix in `main_research_llama3.py` is sufficient for most use cases

**Recommendation**: Stick with transformers 4.40.1 and use `main_research_llama3.py`.

## Troubleshooting

### Issue: "rope_scaling must be a dictionary with two fields" or "rope_scaling's type field must be one of ['linear', 'dynamic']"
**Solution**: Use `main_research_llama3.py` instead of `main_research.py`. The script automatically:
- Detects Llama 3.1's extended rope_scaling format
- Maps unsupported `'llama3'` type to `'linear'` 
- Maintains compatibility with transformers 4.40.1

### Issue: Out of Memory
**Solutions**:
- Reduce `--train_size` and `--val_size`
- Reduce `--batch_size`
- Adjust `--max_memory` to lower value
- Use `--off_load_to_disk` flag

### Issue: Sensitivity file not found OR "Sensitivity range: 1.0000 to 1.0000"
**Problem**: The sensitivity file doesn't exist or contains invalid default values.

**Solution**: Generate proper sensitivity scores:
```bash
python generate_sensitivity_llama3.py \
    --model "meta-llama/Llama-3.1-8B" \
    --num_samples 128 \
    --output_file "./sensitivity_results_llama_3.1_8b.json"
```

**Signs of this issue:**
- Log shows: `Sensitivity range: 1.0000 to 1.0000`
- All layers get uniform bit allocation
- Actual average bits differs significantly from target

## Output Files

After running, you'll find:
- `output_dir/results.json`: Final results and configuration
- `output_dir/*.log`: Training logs
- `save_quant_dir/`: Quantized model weights
- `save_quant_dir/layer_statistics.json`: Per-layer quantization statistics (for MPQ)

## Notes

- First run will download the model from HuggingFace (~16GB for 8B model)
- Dataloaders are cached in `cache_dir` for faster subsequent runs
- Use `--real_quant` for actual quantization; without it, only simulates quantization
- Evaluation tasks require downloading additional datasets
