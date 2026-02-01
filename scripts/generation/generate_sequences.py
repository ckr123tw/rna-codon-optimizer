"""
Generate optimized RNA sequences using the trained pipeline.
"""

import sys
import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import RNACodonOptimizationPipeline
from src.lora_generation.lora_adapter import format_conditional_prompt


def generate_sequences(
    pipeline: RNACodonOptimizationPipeline,
    input_file: str,
    output_file: str,
    target_metrics: Dict[str, float],
    metric_weights: Dict[str, float],
    num_candidates: int = 10,
    batch_size: int = 4
):
    """
    Generate optimized sequences for a batch of inputs.
    """
    print(f"Loading inputs from {input_file}...")
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    print(f"Found {len(df)} samples.")
    
    results = []
    
    for i, row in df.iterrows():
        print(f"\nProcessing sample {i+1}/{len(df)}: {row.get('name', f'Sequence_{i}')}")
        
        # Extract fields
        aa_seq = row.get('amino_acid_sequence') or row.get('sequence')
        utr5 = row.get('utr5', '')
        utr3 = row.get('utr3', '')
        
        if not aa_seq:
            print("  Skipping: No amino acid sequence found")
            continue
            
        # Parse per-sample targets if columns exist, else use global targets
        sample_targets = target_metrics.copy()
        for metric in target_metrics:
            if f'target_{metric}' in row:
                sample_targets[metric] = float(row[f'target_{metric}'])
        
        try:
            result = pipeline.generate_optimized_sequence(
                utr5=utr5,
                utr3=utr3,
                amino_acid_sequence=aa_seq,
                target_metrics=sample_targets,
                metric_weights=metric_weights,
                num_candidates=num_candidates
            )
            
            # Store result
            output_row = row.to_dict()
            output_row['optimized_sequence'] = result['best_sequence']
            
            # Add predictions
            for metric, value in result['predictions'].items():
                output_row[f'predicted_{metric}'] = value
                
            results.append(output_row)
            
            print(f"  Best Sequence Length: {len(result['best_sequence'])}")
            print("  Predictions:")
            for m, v in result['predictions'].items():
                target = result['target_metrics'].get(m, 'N/A')
                print(f"    {m}: {v:.3f} (target: {target})")
                
        except Exception as e:
            print(f"  Error generating sequence: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_df = pd.DataFrame(results)
    if output_file.endswith('.csv'):
        output_df.to_csv(output_file, index=False)
    else:
        output_df.to_excel(output_file, index=False)
        
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate optimized RNA sequences")
    
    parser.add_argument('--input', type=str, required=True, help="Input CSV/Excel with amino_acid_sequence, utr5, utr3")
    parser.add_argument('--output', type=str, default="optimized_sequences.csv", help="Output file path")
    parser.add_argument('--model_name', type=str, default="togethercomputer/evo-1-8k-base", help="Base model name")
    parser.add_argument('--lora_path', type=str, default="models/lora_adapter", help="Path to trained LoRA adapter")
    parser.add_argument('--critic_path', type=str, default="models/critic_best.pt", help="Path to trained critic model")
    
    # Target metrics
    parser.add_argument('--target_te', type=float, default=0.8, help="Target Translation Efficiency (default: 0.8)")
    parser.add_argument('--target_hl', type=float, default=12.0, help="Target Half-Life (default: 12.0)")
    
    # Weights
    parser.add_argument('--weight_te', type=float, default=0.6, help="Weight for TE optimization")
    parser.add_argument('--weight_hl', type=float, default=0.4, help="Weight for Half-Life optimization")
    
    parser.add_argument('--num_candidates', type=int, default=10, help="Number of candidates to generate per sequence")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RNA Sequence Generator")
    print("=" * 60)
    
    # Initialize pipeline (inference only)
    pipeline = RNACodonOptimizationPipeline(
        model_name=args.model_name,
        device=args.device
    )
    
    # Load trained models
    print("\nLoading models...")
    pipeline.load_models(
        lora_path=args.lora_path,
        critic_path=args.critic_path
    )
    
    # Define targets
    target_metrics = {
        'translation_efficiency': args.target_te,
        'half_life': args.target_hl
    }
    
    metric_weights = {
        'translation_efficiency': args.weight_te,
        'half_life': args.weight_hl
    }
    
    generate_sequences(
        pipeline=pipeline,
        input_file=args.input,
        output_file=args.output,
        target_metrics=target_metrics,
        metric_weights=metric_weights,
        num_candidates=args.num_candidates
    )


if __name__ == "__main__":
    main()
