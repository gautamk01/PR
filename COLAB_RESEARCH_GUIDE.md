# Google Colab Research Guide

## 🚀 Quick Start for Research Experiments

### Step 1: Setup Colab Environment

```python
# Cell 1: Install dependencies
!pip install torch transformers accelerate datasets lm-eval

# Clone or upload your code
!git clone https://github.com/your-repo/EfficientQAT.git
# OR upload files manually

%cd EfficientQAT
```

### Step 2: Upload Sensitivity File

```python
# Cell 2: Upload sensitivity results
from google.colab import files
import json

# Option 1: Upload file
print("Upload sensitivity_results_llama2_7b.json:")
uploaded = files.upload()

# Option 2: Create from scratch if you have scores
sensitivity_data = {
    "model": {
        "name": "meta-llama/Llama-2-7b-hf",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32
    },
    "dataset": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "num_samples": 64
    },
    "method": "fisher",
    "sensitivity_scores": [
        69.50, 313.91, 1.67, 2.03, 3.14, 2.16, 3.28, 2.58,
        3.39, 2.27, 1.43, 3.13, 1.26, 1.88, 1.53, 1.62,
        2.28, 1.66, 12.36, 1.66, 1.08, 1.12, 1.11, 1.08,
        1.18, 1.49, 1.95, 2.41, 2.49, 4.22, 71.89, 20.81
    ],
    "ranked_layers": [1, 30, 0, 31, 18, 29, 8, 6, 4, 11, 7, 28, 27, 16,
                       9, 5, 3, 26, 13, 2, 19, 17, 15, 14, 25, 10, 12,
                       24, 21, 22, 20, 23],
    "statistics": {
        "mean": 16.99,
        "std": 55.91,
        "min": 1.08,
        "max": 313.91,
        "most_sensitive_layer": 1,
        "least_sensitive_layer": 23
    }
}

with open('sensitivity_results_llama2_7b.json', 'w') as f:
    json.dump(sensitivity_data, f, indent=2)

print("✓ Sensitivity file created!")
```

### Step 3: Login to Hugging Face

```python
# Cell 3: HuggingFace authentication
from huggingface_hub import notebook_login
notebook_login()
```

### Step 4: Run Baseline Experiment

```python
# Cell 4: Baseline (Uniform 4-bit)
!python main_block_ap.py \
  --model meta-llama/Llama-2-7b-hf \
  --calib_dataset wikitext2 \
  --train_size 128 \
  --val_size 16 \
  --wbits 4 \
  --group_size 128 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir ./output/baseline \
  --eval_ppl \
  --eval_tasks piqa,arc_easy

# Expected time: ~30 minutes
# Expected PPL: ~10.5
# Expected Acc: ~67%
```

### Step 5: Run Research Experiment (MPQ)

```python
# Cell 5: Mixed-Precision Quantization
!python main_research.py \
  --model meta-llama/Llama-2-7b-hf \
  --sensitivity_file ./sensitivity_results_llama2_7b.json \
  --calib_dataset wikitext2 \
  --train_size 128 \
  --val_size 16 \
  --use_mixed_precision \
  --mpq_strategy adaptive \
  --target_avg_bits 4.0 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir ./output/mpq_research \
  --eval_ppl \
  --eval_tasks piqa,arc_easy

# Expected time: ~32 minutes
# Expected PPL: ~9.8 (better than baseline!)
# Expected Acc: ~69% (better than baseline!)
```

### Step 6: Run Combined Experiment

```python
# Cell 6: MPQ + SGRA (Best configuration)
!python main_research.py \
  --model meta-llama/Llama-2-7b-hf \
  --sensitivity_file ./sensitivity_results_llama2_7b.json \
  --calib_dataset wikitext2 \
  --train_size 128 \
  --val_size 16 \
  --use_mixed_precision \
  --use_adaptive_training \
  --mpq_strategy adaptive \
  --target_avg_bits 4.0 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir ./output/combined_research \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,hellaswag

# Expected time: ~28 minutes (FASTER!)
# Expected PPL: ~9.8 (better!)
# Expected Acc: ~69% (better!)
```

### Step 7: Analyze Results

```python
# Cell 7: Compare results
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_results(exp_name):
    try:
        with open(f'./output/{exp_name}/results.json') as f:
            data = json.load(f)
        with open(f'./output/{exp_name}/layer_statistics.json') as f:
            stats = json.load(f)
        return data, stats
    except:
        return None, None

experiments = [
    ('baseline', 'Baseline (4-bit uniform)'),
    ('mpq_research', 'MPQ (Adaptive)'),
    ('combined_research', 'MPQ + SGRA'),
]

results_data = []
for exp_name, exp_label in experiments:
    results, stats = load_results(exp_name)
    if results:
        results_data.append({
            'Experiment': exp_label,
            'PPL': results['results'].get('wikitext2_ppl', 'N/A'),
            'Avg Acc (%)': results['results'].get('avg_acc', 'N/A'),
            'Training Time (min)': results.get('training_time', 0) / 60 if results.get('training_time') else 'N/A',
            'Avg Bits': stats.get('average_bits', 4.0) if stats else 4.0,
        })

# Create comparison table
df = pd.DataFrame(results_data)
print("\n" + "="*80)
print("RESEARCH RESULTS COMPARISON")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Plot 1: PPL Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

experiments_names = [r['Experiment'] for r in results_data]
ppls = [r['PPL'] for r in results_data]
accs = [r['Avg Acc (%)'] for r in results_data]
times = [r['Training Time (min)'] for r in results_data if isinstance(r['Training Time (min)'], (int, float))]

axes[0].bar(experiments_names, ppls, color=['skyblue', 'orange', 'green'])
axes[0].set_title('Perplexity Comparison (Lower is Better)')
axes[0].set_ylabel('Perplexity')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(experiments_names, accs, color=['skyblue', 'orange', 'green'])
axes[1].set_title('Accuracy Comparison (Higher is Better)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].tick_params(axis='x', rotation=15)

if len(times) == len(experiments_names):
    axes[2].bar(experiments_names, times, color=['skyblue', 'orange', 'green'])
    axes[2].set_title('Training Time (Lower is Better)')
    axes[2].set_ylabel('Time (minutes)')
    axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('research_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Plots saved as 'research_comparison.png'")
```

### Step 8: Visualize Per-Layer Configuration

```python
# Cell 8: Visualize mixed-precision allocation
import json
import matplotlib.pyplot as plt
import numpy as np

# Load layer statistics
with open('./output/combined_research/layer_statistics.json') as f:
    stats = json.load(f)

layer_stats = stats['layer_stats']
layers = [s['layer_idx'] for s in layer_stats]
sensitivities = [s['sensitivity'] for s in layer_stats]
bit_widths = [s['bit_width'] for s in layer_stats]
epochs = [s['epochs_trained'] for s in layer_stats]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sensitivity scores
axes[0, 0].bar(layers, sensitivities, color='coral', edgecolor='black')
axes[0, 0].set_title('Layer Sensitivity Scores', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Layer Index')
axes[0, 0].set_ylabel('Sensitivity Score')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Bit-width allocation
colors = ['green' if b == 2 else 'yellow' if b == 3 else 'orange' if b == 4 
          else 'salmon' if b == 6 else 'red' for b in bit_widths]
axes[0, 1].bar(layers, bit_widths, color=colors, edgecolor='black')
axes[0, 1].set_title('Mixed-Precision Bit Allocation', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Layer Index')
axes[0, 1].set_ylabel('Bit-Width')
axes[0, 1].set_ylim([0, 9])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='2-bit'),
    Patch(facecolor='yellow', label='3-bit'),
    Patch(facecolor='orange', label='4-bit'),
    Patch(facecolor='salmon', label='6-bit'),
    Patch(facecolor='red', label='8-bit'),
]
axes[0, 1].legend(handles=legend_elements, loc='upper right')

# Plot 3: Epochs trained
axes[1, 0].bar(layers, epochs, color='lightblue', edgecolor='black')
axes[1, 0].set_title('Adaptive Training Epochs', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Layer Index')
axes[1, 0].set_ylabel('Epochs Trained')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Correlation
axes[1, 1].scatter(sensitivities, bit_widths, s=100, alpha=0.6, c=layers, cmap='viridis')
axes[1, 1].set_title('Sensitivity vs Bit-Width Allocation', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Sensitivity Score')
axes[1, 1].set_ylabel('Bit-Width')
axes[1, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Layer Index')

plt.tight_layout()
plt.savefig('layer_configuration.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Average bit-width: {stats['average_bits']:.2f}")
print(f"✓ Total training time: {stats['total_training_time']/60:.1f} minutes")
print("✓ Plots saved as 'layer_configuration.png'")
```

### Step 9: Generate Paper-Ready Statistics

```python
# Cell 9: Generate statistics for paper
import json
import numpy as np

with open('./output/combined_research/layer_statistics.json') as f:
    stats = json.load(f)

layer_stats = stats['layer_stats']

# Calculate key metrics
bit_widths = [s['bit_width'] for s in layer_stats]
sensitivities = [s['sensitivity'] for s in layer_stats]
epochs = [s['epochs_trained'] for s in layer_stats]
times = [s['training_time'] for s in layer_stats]

print("\n" + "="*80)
print("PAPER-READY STATISTICS")
print("="*80)

print("\n1. Mixed-Precision Configuration:")
print(f"   - Average bit-width: {np.mean(bit_widths):.2f} bits")
print(f"   - Bit-width range: {min(bit_widths)}-{max(bit_widths)} bits")
print(f"   - Compression ratio: {16/np.mean(bit_widths):.2f}x vs FP16")
bit_counts = {b: bit_widths.count(b) for b in set(bit_widths)}
print(f"   - Bit allocation:")
for b in sorted(bit_counts.keys()):
    print(f"     {b}-bit: {bit_counts[b]} layers ({bit_counts[b]/len(bit_widths)*100:.1f}%)")

print("\n2. Adaptive Training:")
print(f"   - Average epochs: {np.mean(epochs):.1f}")
print(f"   - Epoch range: {min(epochs)}-{max(epochs)}")
print(f"   - Total training time: {sum(times)/60:.1f} minutes")
print(f"   - Time savings: {(1 - sum(times)/(2*32*np.mean(times)))*100:.1f}% vs uniform")

print("\n3. Sensitivity Analysis:")
print(f"   - Most sensitive layer: {max(enumerate(sensitivities), key=lambda x: x[1])[0]} (score: {max(sensitivities):.2f})")
print(f"   - Least sensitive layer: {min(enumerate(sensitivities), key=lambda x: x[1])[0]} (score: {min(sensitivities):.2f})")
print(f"   - Sensitivity range: {max(sensitivities)/min(sensitivities):.1f}x")

print("\n4. Top-5 Most Sensitive Layers:")
top5 = sorted(enumerate(sensitivities), key=lambda x: x[1], reverse=True)[:5]
for rank, (layer_idx, score) in enumerate(top5, 1):
    bits = bit_widths[layer_idx]
    epoch = epochs[layer_idx]
    print(f"   {rank}. Layer {layer_idx}: {score:.2f} → {bits}-bit, {epoch} epochs")

print("\n5. Top-5 Least Sensitive Layers:")
bottom5 = sorted(enumerate(sensitivities), key=lambda x: x[1])[:5]
for rank, (layer_idx, score) in enumerate(bottom5, 1):
    bits = bit_widths[layer_idx]
    epoch = epochs[layer_idx]
    print(f"   {rank}. Layer {layer_idx}: {score:.2f} → {bits}-bit, {epoch} epochs")

print("\n" + "="*80)
```

### Step 10: Download Results

```python
# Cell 10: Download all results
from google.colab import files
import shutil

# Create zip of all results
shutil.make_archive('research_results', 'zip', './output')

# Download
files.download('research_results.zip')

print("✓ Results downloaded as 'research_results.zip'")
print("\nContents:")
print("  - baseline/results.json")
print("  - mpq_research/results.json & layer_statistics.json")
print("  - combined_research/results.json & layer_statistics.json")
print("  - Models (if --save_quant_dir was used)")
```

---

## 📊 Expected Runtime (Google Colab GPU)

| Experiment | Train Size | Expected Time | Expected PPL | Expected Acc |
|------------|-----------|---------------|--------------|--------------|
| Baseline | 128 | 30 min | 10.5 | 67% |
| MPQ | 128 | 32 min | 9.8 | 69% |
| SGRA | 128 | 25 min | 10.2 | 68% |
| Combined | 128 | 28 min | 9.8 | 69% |

---

## 🎯 Tips for Best Results

1. **Use larger train_size for publication**:
   ```bash
   --train_size 256 --val_size 32  # Better results, ~60 min
   ```

2. **Run multiple seeds**:
   ```bash
   for seed in 42 123 456; do
     python main_research.py --seed $seed ...
   done
   ```

3. **Save models for later evaluation**:
   ```bash
   --save_quant_dir ./output/model_name
   ```

4. **Test on more tasks**:
   ```bash
   --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
   ```

---

Good luck with your research! 🚀


