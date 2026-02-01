#!/usr/bin/env python3
"""
Training script for RNA Codon Optimizer using PPO.
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import RNACodonOptimizationPipeline

def train(args):
    print("=" * 60)
    print(" RNA CODON OPTIMIZER - PPO TRAINING")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RNACodonOptimizationPipeline(
        data_path=args.data_path,
        model_name=args.model_name
    )
    
    # Step 1: Prepare data
    pipeline.step1_prepare_data(max_samples=args.max_samples)
    
    # Step 2: Train critic (if not loading pretrained)
    if args.pretrained_critic:
        # Todo: Implement loading pretrained critic
        print("Loading pretrained critic not yet implemented, training from scratch...")
    
    pipeline.step2_train_critic(
        num_epochs=args.critic_epochs,
        batch_size=args.critic_batch_size,
        learning_rate=args.critic_lr
    )
    
    # Step 3: Initialize LoRA
    pipeline.step3_initialize_lora(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Step 4: PPO Training
    print("\nStarting PPO optimization...")
    pipeline.step4_ppo_training(
        num_epochs=args.ppo_epochs,
        steps_per_epoch=args.steps_per_epoch
    )
    
    print("\nTraining workflow completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for RNA Optimization")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to Zheng et al. data (Excel/CSV)')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default="togethercomputer/evo-1-8k-base", help='Foundation model name')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    
    # Training arguments
    parser.add_argument('--critic_epochs', type=int, default=10, help='Critic training epochs')
    parser.add_argument('--critic_batch_size', type=int, default=32, help='Critic batch size')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic learning rate')
    
    parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=50, help='Steps per PPO epoch')
    parser.add_argument('--pretrained_critic', action='store_true', help='Load pretrained critic')
    
    args = parser.parse_args()
    
    train(args)
