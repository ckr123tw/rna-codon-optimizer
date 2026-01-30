"""
PPO (Proximal Policy Optimization) training for LoRA adapter.
Uses the critic model to provide rewards for generated sequences.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training."""
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    kl_penalty: str = "kl"  # or "abs", "mse"
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    adap_kl_ctrl: bool = True
    init_kl_coef: float = 0.2
    target_kl: float = 6.0


class RNAPPOTrainer:
    """
    PPO trainer for optimizing RNA sequence generation using critic feedback.
    """
    
    def __init__(
        self,
        lora_model,
        critic_model,
        embedder,
        config: PPOTrainingConfig,
        device: Optional[str] = None
    ):
        """
        Initialize PPO trainer.
        
        Args:
            lora_model: LoRA-adapted Evo model
            critic_model: Trained translation efficiency critic
            embedder: Evo embedder for generating sequence embeddings
            config: PPO training configuration
            device: Device to use
        """
        self.lora_model = lora_model
        self.critic_model = critic_model
        self.embedder = embedder
        self.config = config
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.critic_model = self.critic_model.to(self.device)
        self.critic_model.eval()  # Critic is frozen during RL
        
        # Training metrics
        self.training_stats = {
            'rewards': [],
            'kl_divergences': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def compute_reward(
        self,
        generated_sequence: str,
        target_efficiency: float,
        amino_acid_sequence: str,
        utr5: Optional[str] = None,
        utr3: Optional[str] = None
    ) -> float:
        """
        Compute reward for a generated RNA sequence.
        
        Args:
            generated_sequence: Generated RNA sequence
            target_efficiency: Target translation efficiency
            amino_acid_sequence: Expected amino acid sequence
            utr5: Expected 5'UTR (for validation)
            utr3: Expected 3'UTR (for validation)
            
        Returns:
            Reward value (higher is better)
        """
        # Get sequence embedding
        try:
            embedding = self.embedder.embed_sequence(generated_sequence, return_numpy=False)
            embedding = embedding.unsqueeze(0).to(self.device)
        except Exception as e:
            # Large penalty for invalid sequences
            return -100.0
        
        # Predict translation efficiency
        with torch.no_grad():
            critic_output = self.critic_model(embedding)
            if isinstance(critic_output, dict):
                predicted_te = critic_output['translation_efficiency'].item()
            else:
                predicted_te = critic_output.item()
        
        # Compute reward based on how close to target
        te_reward = -abs(predicted_te - target_efficiency)
        
        # Validate amino acid sequence preservation
        validation_bonus = 0.0
        if utr5 is not None and utr3 is not None:
            from ..sequence_generation.validation import validate_full_rna_sequence
            
            validation = validate_full_rna_sequence(
                generated_sequence,
                utr5,
                utr3,
                amino_acid_sequence,
                allow_stop=True
            )
            
            if validation['is_valid']:
                validation_bonus = 2.0  # Bonus for valid sequence
            else:
                validation_bonus = -15.0  # Large penalty for invalid
        
        total_reward = te_reward + validation_bonus
        
        return total_reward
    
    def generate_and_score_batch(
        self,
        prompts: List[str],
        target_efficiencies: List[float],
        amino_acid_sequences: List[str]
    ) -> Dict:
        """
        Generate sequences for a batch of prompts and compute rewards.
        
        Args:
            prompts: List of conditioning prompts
            target_efficiencies: List of target TE values
            amino_acid_sequences: List of amino acid sequences
            
        Returns:
            Dictionary with generated sequences, rewards, and other info
        """
        # Generate sequences
        generated_sequences = []
        for prompt, aa_seq in zip(prompts, amino_acid_sequences):
            seqs = self.lora_model.generate_sequences(
                prompt=prompt,
                num_sequences=1,
                max_length=500,
                temperature=0.8,
                amino_acid_constraint=aa_seq
            )
            generated_sequences.append(seqs[0] if seqs else "")
        
        # Compute rewards
        rewards = []
        for seq, target_te, aa_seq in zip(generated_sequences, target_efficiencies, amino_acid_sequences):
            reward = self.compute_reward(seq, target_te, aa_seq)
            rewards.append(reward)
        
        return {
            'generated_sequences': generated_sequences,
            'rewards': rewards,
            'prompts': prompts
        }
    
    def train_step(
        self,
        prompts: List[str],
        target_efficiencies: List[float],
        amino_acid_sequences: List[str]
    ) -> Dict:
        """
        Perform one PPO training step.
        
        Args:
            prompts: Batch of conditioning prompts
            target_efficiencies: Target TE values
            amino_acid_sequences: Amino acid sequences
            
        Returns:
            Training statistics
        """
        # Generate and score
        batch_data = self.generate_and_score_batch(
            prompts, target_efficiencies, amino_acid_sequences
        )
        
        rewards = batch_data['rewards']
        avg_reward = np.mean(rewards)
        
        # TODO: Implement actual PPO update using TRL library
        # For now, we just track statistics
        
        self.training_stats['rewards'].append(avg_reward)
        
        return {
            'avg_reward': avg_reward,
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards)
        }
    
    def train(
        self,
        prompts: List[str],
        target_efficiencies: List[float],
        amino_acid_sequences: List[str],
        num_epochs: int = 10,
        steps_per_epoch: int = 100
    ):
        """
        Train the LoRA model using PPO.
        
        Args:
            prompts: Training prompts
            target_efficiencies: Target TE values
            amino_acid_sequences: Amino acid sequences
            num_epochs: Number of training epochs
            steps_per_epoch: Steps per epoch
        """
        print("=" * 60)
        print("Starting PPO Training")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            epoch_rewards = []
            
            for step in range(steps_per_epoch):
                # Sample a batch
                batch_size = self.config.batch_size
                indices = np.random.choice(len(prompts), size=batch_size, replace=True)
                
                batch_prompts = [prompts[i] for i in indices]
                batch_targets = [target_efficiencies[i] for i in indices]
                batch_aa_seqs = [amino_acid_sequences[i] for i in indices]
                
                # Training step
                stats = self.train_step(batch_prompts, batch_targets, batch_aa_seqs)
                epoch_rewards.append(stats['avg_reward'])
                
                if (step + 1) % 20 == 0:
                    print(f"  Step {step + 1}/{steps_per_epoch}, "
                          f"Avg Reward: {stats['avg_reward']:.4f}")
            
            avg_epoch_reward = np.mean(epoch_rewards)
            print(f"Epoch {epoch + 1} Average Reward: {avg_epoch_reward:.4f}")
    
    def save_training_stats(self, path: str):
        """Save training statistics to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"Training stats saved to {path}")


if __name__ == "__main__":
    # Test PPO trainer setup
    print("=" * 60)
    print("PPO Trainer Test")
    print("=" * 60)
    
    config = PPOTrainingConfig(
        learning_rate=1e-5,
        batch_size=4,
        ppo_epochs=4
    )
    
    print("\nPPO Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  PPO epochs: {config.ppo_epochs}")
    print(f"  KL penalty: {config.kl_penalty}")
    print(f"  Clip range: {config.cliprange}")
