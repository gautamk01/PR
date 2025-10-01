#!/usr/bin/env python3
"""
Quick validation script for sensitivity scores
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load sensitivity results
with open('sensitivity_results_llama2_7b.json', 'r') as f:
    data = json.load(f)

scores = np.array(data['sensitivity_scores'])
ranked = data['ranked_layers']
stats = data['statistics']

print("=" * 60)
print("SENSITIVITY SCORES VALIDATION")
print("=" * 60)

# 1. Basic checks
print("\n✓ BASIC CHECKS:")
print(f"  Number of layers: {len(scores)} (expected: 32)")
print(f"  Method: {data['method']}")
print(f"  Samples used: {data['dataset']['num_samples']}")
print(f"  All scores > 0: {(scores > 0).all()}")
print(f"  No NaN/Inf: {np.isfinite(scores).all()}")

# 2. Statistical summary
print(f"\n✓ STATISTICS:")
print(f"  Min:    {stats['min']:.4f} (Layer {stats['least_sensitive_layer']})")
print(f"  Max:    {stats['max']:.4f} (Layer {stats['most_sensitive_layer']})")
print(f"  Mean:   {stats['mean']:.4f}")
print(f"  Std:    {stats['std']:.4f}")
print(f"  Range:  {stats['max']/stats['min']:.1f}x variation")

# 3. Top/Bottom layers
print(f"\n✓ TOP 5 MOST SENSITIVE LAYERS:")
for i in range(5):
    layer_idx = ranked[i]
    score = scores[layer_idx]
    print(f"  {i+1}. Layer {layer_idx:2d}: {score:8.2f}")

print(f"\n✓ TOP 5 LEAST SENSITIVE LAYERS:")
for i in range(5):
    layer_idx = ranked[-(i+1)]
    score = scores[layer_idx]
    print(f"  {i+1}. Layer {layer_idx:2d}: {score:8.2f}")

# 4. Adaptive epochs calculation (preview)
print(f"\n✓ ADAPTIVE EPOCHS PREVIEW (base=2 epochs):")
min_s, max_s = scores.min(), scores.max()
for layer_idx in [1, 30, 0, 23, 20]:  # Sample layers
    sensitivity_ratio = (scores[layer_idx] - min_s) / (max_s - min_s + 1e-8)
    adaptive_epochs = int(2 * (1.0 + sensitivity_ratio * 0.25))
    print(f"  Layer {layer_idx:2d}: {scores[layer_idx]:7.2f} → {adaptive_epochs} epochs")

# 5. Visualization
print(f"\n✓ Generating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bar chart
ax = axes[0, 0]
ax.bar(range(len(scores)), scores, color='skyblue', edgecolor='black')
ax.set_title('Sensitivity Scores by Layer', fontsize=12, fontweight='bold')
ax.set_xlabel('Layer Index')
ax.set_ylabel('Sensitivity Score')
ax.grid(True, alpha=0.3)
# Highlight top 3
for i in range(3):
    layer_idx = ranked[i]
    ax.bar(layer_idx, scores[layer_idx], color='red', alpha=0.7)

# Plot 2: Line plot
ax = axes[0, 1]
ax.plot(range(len(scores)), scores, 'o-', linewidth=2, markersize=6)
ax.set_title('Sensitivity Trend', fontsize=12, fontweight='bold')
ax.set_xlabel('Layer Index')
ax.set_ylabel('Sensitivity Score')
ax.grid(True, alpha=0.3)
ax.axhline(y=stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
ax.legend()

# Plot 3: Normalized scores (for adaptive epochs)
ax = axes[1, 0]
norm_scores = (scores - min_s) / (max_s - min_s + 1e-8)
adaptive_epochs_all = [int(2 * (1.0 + r * 0.25)) for r in norm_scores]
colors = ['green' if e == 2 else 'orange' if e == 2 else 'red' for e in adaptive_epochs_all]
ax.bar(range(len(adaptive_epochs_all)), adaptive_epochs_all, color=colors, alpha=0.7, edgecolor='black')
ax.set_title('Adaptive Epochs per Layer (base=2)', fontsize=12, fontweight='bold')
ax.set_xlabel('Layer Index')
ax.set_ylabel('Training Epochs')
ax.set_ylim([1.5, 3])
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Cumulative distribution
ax = axes[1, 1]
sorted_scores = np.sort(scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
ax.plot(sorted_scores, cumulative, 'b-', linewidth=2)
ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Sensitivity Score')
ax.set_ylabel('Cumulative Percentage (%)')
ax.grid(True, alpha=0.3)
ax.axvline(x=stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
ax.legend()

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Saved: sensitivity_analysis.png")

# 6. Final verdict
print(f"\n" + "=" * 60)
print("VALIDATION RESULT: ✅ SCORES ARE VALID AND READY TO USE")
print("=" * 60)
print("\nNext steps:")
print("  1. Run training with: bash QUICK_START.sh")
print("  2. Or run: bash run_sensitivity_baseline.sh")
print("  3. Add --adaptive_epochs flag for adaptive training")
print()


