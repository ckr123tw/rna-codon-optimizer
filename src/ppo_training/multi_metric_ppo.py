"""
Enhanced PPO trainer with multi-metric optimization support.
"""

import torch
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

from .ppo_trainer import PPOTrainingConfig


class MultiMetricRewardFunction:
    """
    Reward function for multi-objective optimization.
    Combines rewards from multiple metrics with configurable weights.
    """
    
    def __init__(
        self,
        multi_metric_critic,
        embedder,
        metric_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize multi-metric reward function.
        
        Args:
            multi_metric_critic: Multi-metric critic model
            embedder: Evo embedder for sequence encoding
            metric_weights: Weights for each metric in reward calculation
            device: Device to use
        """
        self.critic = multi_metric_critic
        self.embedder = embedder
        
        # Default equal weights
        if metric_weights is None:
            metric_weights = {m: 1.0 for m in self.critic.metrics}
        self.metric_weights = metric_weights
        
        # Normalize weights to sum to 1
        total_weight = sum(metric_weights.values())
        self.metric_weights = {k: v / total_weight for k, v in metric_weights.items()}
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.critic = self.critic.to(self.device)
        self.critic.eval()
    
    def compute_reward(
        self,
        generated_sequence: str,
        targets: Dict[str, float],
        amino_acid_sequence: Optional[str] = None,
        utr5: Optional[str] = None,
        utr3: Optional[str] = None
    ) -> float:
        """
        Compute multi-metric reward for a generated sequence.
        
        Args:
            generated_sequence: Generated RNA sequence
            targets: Dictionary of target values for each metric
            amino_acid_sequence: Expected amino acid sequence (for validation)
            utr5: Expected 5'UTR (for validation)
            utr3: Expected 3'UTR (for validation)
            
        Returns:
            Combined reward value (higher is better)
        """
        # Get sequence embedding
        try:
            embedding = self.embedder.embed_sequence(generated_sequence, return_numpy=False)
            embedding = embedding.unsqueeze(0).to(self.device)
        except Exception as e:
            # Large penalty for invalid sequences
            return -100.0
        
        # Predict all metrics
        with torch.no_grad():
            predictions = self.critic.predict(embedding, return_numpy=True)
        
        # Compute per-metric rewards
        total_reward = 0.0
        metric_rewards = {}
        
        for metric, target_value in targets.items():
            if metric in predictions:
                pred_value = predictions[metric][0]
                
                # Negative L1 distance (closer is better)
                metric_reward = -abs(pred_value - target_value)
                
                # Weight and accumulate
                weighted_reward = self.metric_weights.get(metric, 1.0) * metric_reward
                total_reward += weighted_reward
                metric_rewards[metric] = metric_reward
        
        # Validate amino acid sequence preservation
        if amino_acid_sequence is not None and utr5 is not None and utr3 is not None:
            from ..sequence_generation.validation import validate_full_rna_sequence
            
            validation = validate_full_rna_sequence(
                generated_sequence,
                utr5,
                utr3,
                amino_acid_sequence,
                allow_stop=True
            )
            
            if validation['is_valid']:
                # Bonus for preserving amino acid sequence
                total_reward += 2.0
            else:
                # Large penalty for incorrect amino acid sequence
                total_reward -= 15.0
        
        return total_reward


class MultiMetricPPOTrainer:
    """
    PPO trainer for multi-metric optimization.
    """
    
    def __init__(
        self,
        lora_model,
        multi_metric_critic,
        embedder,
        metric_weights: Dict[str, float],
        config: PPOTrainingConfig,
        device: Optional[str] = None
    ):
        """
        Initialize multi-metric PPO trainer.
        
        Args:
            lora_model: LoRA-adapted Evo model
            multi_metric_critic: Multi-metric critic model
            embedder: Evo embedder
            metric_weights: Weights for each metric in reward
            config: PPO training configuration
            device: Device to use
        """
        self.lora_model = lora_model
        self.critic = multi_metric_critic
        self.embedder = embedder
        self.config = config
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize reward function
        self.reward_function = MultiMetricRewardFunction(
            multi_metric_critic=multi_metric_critic,
            embedder=embedder,
            metric_weights=metric_weights,
            device=str(self.device)
        )
        
        # Training metrics
        self.training_stats = {
            'rewards': [],
            'kl_divergences': [],
            'per_metric_rewards': {m: [] for m in self.critic.metrics}
        }
    
    def generate_and_score_batch(
        self,
        prompts: List[str],
        targets_list: List[Dict[str, float]],
        amino_acid_sequences: List[str]
    ) -> Dict:
        """
        Generate sequences and compute multi-metric rewards.
        
        Args:
            prompts: List of conditioning prompts
            targets_list: List of target metric dictionaries
            amino_acid_sequences: List of amino acid sequences
            
        Returns:
            Dictionary with sequences, rewards, and metrics
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
        for seq, targets, aa_seq in zip(generated_sequences, targets_list, amino_acid_sequences):
            reward = self.reward_function.compute_reward(seq, targets, aa_seq)
            rewards.append(reward)
        
        return {
            'generated_sequences': generated_sequences,
            'rewards': rewards,
            'prompts': prompts
        }
    
    def train_step(
        self,
        prompts: List[str],
        targets_list: List[Dict[str, float]],
        amino_acid_sequences: List[str]
    ) -> Dict:
        """
        Perform one PPO training step with multi-metric objectives.
        
        Args:
            prompts: Batch of conditioning prompts
            targets_list: List of target metric dictionaries
            amino_acid_sequences: Amino acid sequences
            
        Returns:
            Training statistics
        """
        # Generate and score
        batch_data = self.generate_and_score_batch(
            prompts, targets_list, amino_acid_sequences
        )
        
        rewards = batch_data['rewards']
        avg_reward = np.mean(rewards)
        
        # TODO: Implement actual PPO update using TRL library
        # For now, track statistics
        
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
        targets_list: List[Dict[str, float]],
        amino_acid_sequences: List[str],
        num_epochs: int = 10,
        steps_per_epoch: int = 100
    ):
        """
        Train the LoRA model using multi-metric PPO.
        
        Args:
            prompts: Training prompts
            targets_list: List of target metric dictionaries
            amino_acid_sequences: Amino acid sequences
            num_epochs: Number of training epochs
            steps_per_epoch: Steps per epoch
        """
        print("=" * 60)
        print("Multi-Metric PPO Training")
        print("=" * 60)
        print(f"Optimizing metrics: {list(self.critic.metrics)}")
        print(f"Metric weights: {self.reward_function.metric_weights}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            epoch_rewards = []
            
            for step in range(steps_per_epoch):
                # Sample a batch
                batch_size = self.config.batch_size
                indices = np.random.choice(len(prompts), size=batch_size, replace=True)
                
                batch_prompts = [prompts[i] for i in indices]
                batch_targets = [targets_list[i] for i in indices]
                batch_aa_seqs = [amino_acid_sequences[i] for i in indices]
                
                # Training step
                stats = self.train_step(batch_prompts, batch_targets, batch_aa_seqs)
                epoch_rewards.append(stats['avg_reward'])
                
                if (step + 1) % 20 == 0:
                    print(f"  Step {step + 1}/{steps_per_epoch}, "
                          f"Avg Reward: {stats['avg_reward']:.4f}")
            
            avg_epoch_reward = np.mean(epoch_rewards)
            print(f"Epoch {epoch + 1} Average Reward: {avg_epoch_reward:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Metric PPO Trainer Test")
    print("=" * 60)
    
    config = PPOTrainingConfig(batch_size=4, ppo_epochs=4)
    
    print("\nMulti-metric configuration:")
    print("  Metrics: translation_efficiency, half_life")
    print("  Weights: {'translation_efficiency': 0.6, 'half_life': 0.4}")
