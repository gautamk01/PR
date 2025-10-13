#!/usr/bin/env python3
"""
Generate sensitivity scores for Llama 3.1 models
Compatible with transformers 4.40.1
"""
import os
import sys
import random
import numpy as np
import torch
import json
from datautils_block import get_loaders
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from pathlib import Path


def load_llama3_config(model_name_or_path):
    """Load config with Llama 3.1 rope_scaling compatibility fix"""
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        return config
    except ValueError as e:
        if "rope_scaling" not in str(e):
            raise
        
        print(f"[INFO] Detected Llama 3.1 rope_scaling format, applying compatibility fix...")
        
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if 'rope_scaling' in config_dict and isinstance(config_dict['rope_scaling'], dict):
            rope_scaling = config_dict['rope_scaling']
            if 'rope_type' in rope_scaling or rope_scaling.get('type') == 'llama3':
                rope_type = rope_scaling.get('rope_type', rope_scaling.get('type', 'linear'))
                compatible_type = 'linear' if rope_type == 'llama3' else rope_type
                config_dict['rope_scaling'] = {
                    'type': compatible_type,
                    'factor': rope_scaling.get('factor', 8.0)
                }
        
        from transformers import LlamaConfig
        config = LlamaConfig.from_dict(config_dict)
        return config


def compute_layer_sensitivity(model, dataloader, device='cuda'):
    """
    Compute per-layer sensitivity using output magnitude method.
    Higher values = more sensitive layer (needs more precision).
    """
    model.eval()
    layer_outputs = []
    
    print(f"\n[SENSITIVITY] Computing sensitivity scores...")
    print(f"[SENSITIVITY] Using {len(dataloader)} samples")
    
    # Get number of layers
    num_layers = len(model.model.layers)
    layer_magnitudes = [[] for _ in range(num_layers)]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{len(dataloader)}")
            
            input_ids = batch[0].to(device)
            
            # Forward pass through embedding
            hidden_states = model.model.embed_tokens(input_ids)
            
            # Pass through each layer and record output magnitude
            for layer_idx, layer in enumerate(model.model.layers):
                hidden_states = layer(hidden_states)[0]
                
                # Compute magnitude (L2 norm)
                magnitude = torch.norm(hidden_states).item()
                layer_magnitudes[layer_idx].append(magnitude)
    
    # Compute average magnitude per layer
    sensitivity_scores = []
    for layer_idx in range(num_layers):
        avg_magnitude = np.mean(layer_magnitudes[layer_idx])
        sensitivity_scores.append(avg_magnitude)
    
    # Normalize scores (higher = more sensitive)
    sensitivity_scores = np.array(sensitivity_scores)
    sensitivity_scores = sensitivity_scores / np.mean(sensitivity_scores)
    
    return sensitivity_scores


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Sensitivity Scores for Llama 3.1")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="Output JSON file (default: sensitivity_results_{model_name}.json)")
    parser.add_argument("--num_samples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048,
                       help="Sequence length")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Create cache dir
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if args.output_file is None:
        model_name = args.model.split('/')[-1].lower().replace('-', '_')
        args.output_file = f"sensitivity_results_{model_name}.json"
    
    print("="*70)
    print("LLAMA 3.1 SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_file}")
    print("="*70)
    
    # Load model with compatibility fix
    print("\n[1/4] Loading model...")
    config = load_llama3_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
    
    # Load model on CPU first
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        config=config, 
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded: {len(model.model.layers)} layers")
    
    # Load calibration data
    print("\n[2/4] Loading calibration data...")
    trainloader, _ = get_loaders(
        "wikitext2",
        tokenizer,
        args.num_samples,
        16,  # val_size (not used)
        seed=args.seed,
        seqlen=args.seqlen,
    )
    print(f"✓ Loaded {len(trainloader)} batches")
    
    # Compute sensitivity
    print("\n[3/4] Computing sensitivity scores...")
    device = next(model.parameters()).device
    sensitivity_scores = compute_layer_sensitivity(model, trainloader, device=device)
    
    # Statistics
    print(f"\n✓ Sensitivity computed:")
    print(f"  Min:  {sensitivity_scores.min():.4f} (Layer {sensitivity_scores.argmin()})")
    print(f"  Max:  {sensitivity_scores.max():.4f} (Layer {sensitivity_scores.argmax()})")
    print(f"  Mean: {sensitivity_scores.mean():.4f}")
    print(f"  Std:  {sensitivity_scores.std():.4f}")
    print(f"  Range: {sensitivity_scores.max()/sensitivity_scores.min():.2f}x variation")
    
    # Rank layers
    ranked_layers = np.argsort(sensitivity_scores)[::-1].tolist()
    
    print(f"\n  Top 5 most sensitive layers:")
    for i in range(5):
        layer_idx = ranked_layers[i]
        print(f"    {i+1}. Layer {layer_idx:2d}: {sensitivity_scores[layer_idx]:.4f}")
    
    print(f"\n  Top 5 least sensitive layers:")
    for i in range(5):
        layer_idx = ranked_layers[-(i+1)]
        print(f"    {i+1}. Layer {layer_idx:2d}: {sensitivity_scores[layer_idx]:.4f}")
    
    # Save results
    print(f"\n[4/4] Saving results to {args.output_file}...")
    results = {
        "model": args.model,
        "method": "output_magnitude",
        "sensitivity_scores": sensitivity_scores.tolist(),
        "ranked_layers": ranked_layers,
        "statistics": {
            "min": float(sensitivity_scores.min()),
            "max": float(sensitivity_scores.max()),
            "mean": float(sensitivity_scores.mean()),
            "std": float(sensitivity_scores.std()),
            "most_sensitive_layer": int(sensitivity_scores.argmax()),
            "least_sensitive_layer": int(sensitivity_scores.argmin()),
        },
        "dataset": {
            "name": "wikitext2",
            "num_samples": args.num_samples,
            "seqlen": args.seqlen,
        },
        "config": {
            "seed": args.seed,
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved to {args.output_file}")
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nNow you can run MPQ with:")
    print(f"  python main_research_llama3.py \\")
    print(f"    --model {args.model} \\")
    print(f"    --sensitivity_file {args.output_file} \\")
    print(f"    --use_mixed_precision \\")
    print(f"    --mpq_strategy aggressive \\")
    print(f"    --target_avg_bits 4 ...")


if __name__ == "__main__":
    main()
