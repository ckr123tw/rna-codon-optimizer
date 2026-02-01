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
from dataclasses import dataclass, field


@dataclass
class PPOTrainingConfig:
    """
    Configuration for PPO training with multi-metric reward support.
    
    Hyperparameters (Tuning Guide):
        learning_rate: Policy learning rate (default: 1e-5)
            - 1e-5 to 1e-6 for stable LoRA fine-tuning
            - Higher (1e-4) may cause instability
            
        batch_size: Number of samples per PPO update (default: 4)
            - Larger (8-16) for more stable gradients
            - Smaller (2-4) for faster iteration
            
        mini_batch_size: Samples per gradient step (default: 1)
            - Increase if memory allows for better gradients
            
        ppo_epochs: PPO updates per batch of data (default: 4)
            - Higher (6-8) for more learning per sample
            - Too high may cause overfitting to batch
            
        cliprange: PPO clipping parameter (default: 0.2)
            - Limits policy updates to prevent instability
            - Lower (0.1) for more conservative updates
            
        target_kl: Target KL divergence (default: 6.0)
            - Training stops early if KL exceeds this
            - Lower (0.01-0.1) for more stability
            
        init_kl_coef: Initial KL penalty coefficient (default: 0.2)
            - Balances reward vs staying close to original policy
    
    Multi-Metric Reward Configuration:
        metric_weights: Dictionary of metric -> weight for reward computation
            Example: {'translation_efficiency': 0.7, 'half_life': 0.3}
            - Weights should sum to 1.0 for normalized rewards
            - Set weight to 0.0 to ignore a metric
            - Default: Only translation_efficiency with weight 1.0
            
        reward_aggregation: How to combine multi-metric rewards
            - 'weighted_sum': Sum of (weight * metric_reward) - default
            - 'pareto': Pareto-based, uses minimum normalized reward
            - 'product': Product of normalized rewards (all must be positive)
            - 'tchebyshev': Minimizes max weighted deviation from ideal
            
        normalize_rewards: Whether to normalize individual metric rewards
            to [0, 1] range before aggregation (default: True)
            Recommended when using 'pareto' or 'product' aggregation.
            
    Examples:
        # Single metric (default behavior)
        >>> config = PPOTrainingConfig()
        
        # Weighted multi-objective
        >>> config = PPOTrainingConfig(
        ...     metric_weights={'translation_efficiency': 0.7, 'half_life': 0.3},
        ...     reward_aggregation='weighted_sum'
        ... )
        
        # Pareto-based (balance all objectives equally)
        >>> config = PPOTrainingConfig(
        ...     metric_weights={'translation_efficiency': 1.0, 'half_life': 1.0},
        ...     reward_aggregation='pareto'
        ... )
    """
    # PPO hyperparameters
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
    
    # Multi-metric reward configuration
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'translation_efficiency': 1.0
    })
    reward_aggregation: str = 'weighted_sum'  # 'weighted_sum', 'pareto', 'product', 'tchebyshev'
    normalize_rewards: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        valid_aggregations = {'weighted_sum', 'pareto', 'product', 'tchebyshev'}
        if self.reward_aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid reward_aggregation '{self.reward_aggregation}'. "
                f"Must be one of: {valid_aggregations}"
            )


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
        target_metrics: Dict[str, float],
        amino_acid_sequence: str,
        utr5: Optional[str] = None,
        utr3: Optional[str] = None,
        # Backward compatibility
        target_efficiency: Optional[float] = None
    ) -> float:
        """
        Compute multi-metric reward for a generated RNA sequence.
        
        Supports multiple reward aggregation strategies:
        - weighted_sum: Linear combination of metric rewards
        - pareto: Minimum normalized reward (encourages balanced improvement)
        - product: Product of rewards (requires all metrics to improve)
        - tchebyshev: Minimizes worst-case deviation from targets
        
        Args:
            generated_sequence: Generated RNA sequence
            target_metrics: Dictionary of target metric values
                Example: {'translation_efficiency': 0.8, 'half_life': 12.0}
            amino_acid_sequence: Expected amino acid sequence
            utr5: Expected 5'UTR (for validation)
            utr3: Expected 3'UTR (for validation)
            target_efficiency: DEPRECATED - Use target_metrics instead
            
        Returns:
            Reward value (higher is better)
            
        Aggregation Strategies:
            weighted_sum (default):
                reward = sum(weight_i * metric_reward_i)
                Best for: Clear priority ordering of metrics
                
            pareto:
                reward = min(normalized_reward_i for all metrics)
                Best for: Balanced multi-objective optimization
                Ensures no single metric is neglected
                
            product:
                reward = product(normalized_reward_i for all metrics)
                Best for: When all metrics must improve together
                Warning: Sensitive to any metric being near zero
                
            tchebyshev:
                reward = -max(weight_i * |predicted_i - target_i|)
                Best for: Minimax approach, bounds worst-case deviation
        """
        # Backward compatibility
        if target_metrics is None or len(target_metrics) == 0:
            if target_efficiency is not None:
                target_metrics = {'translation_efficiency': target_efficiency}
            else:
                target_metrics = {'translation_efficiency': 0.5}
        
        # Get sequence embedding
        try:
            embedding = self.embedder.embed_sequence(generated_sequence, return_numpy=False)
            embedding = embedding.unsqueeze(0).to(self.device)
        except Exception as e:
            # Large penalty for invalid sequences
            return -100.0
        
        # Get predictions for all metrics
        with torch.no_grad():
            critic_output = self.critic_model(embedding, return_dict=True)
            if not isinstance(critic_output, dict):
                # Single output model - assume translation_efficiency
                critic_output = {'translation_efficiency': critic_output}
        
        # Compute individual metric rewards
        metric_rewards = {}
        for metric_name, weight in self.config.metric_weights.items():
            if weight <= 0:
                continue
                
            if metric_name not in critic_output:
                continue
                
            predicted = critic_output[metric_name]
            if isinstance(predicted, torch.Tensor):
                predicted = predicted.item()
            
            target = target_metrics.get(metric_name, 0.0)
            
            # Raw reward: negative absolute error (closer = better)
            raw_reward = -abs(predicted - target)
            metric_rewards[metric_name] = {
                'predicted': predicted,
                'target': target,
                'raw_reward': raw_reward,
                'weight': weight
            }
        
        # Normalize rewards if configured
        if self.config.normalize_rewards and metric_rewards:
            # Normalize to [0, 1] range based on typical error bounds
            for metric_name, data in metric_rewards.items():
                # Use sigmoid-like normalization: 1 / (1 + |error|)
                raw = data['raw_reward']
                normalized = 1.0 / (1.0 + abs(raw))
                data['normalized_reward'] = normalized
        else:
            for data in metric_rewards.values():
                data['normalized_reward'] = data['raw_reward']
        
        # Aggregate rewards based on strategy
        aggregation = self.config.reward_aggregation
        
        if not metric_rewards:
            total_reward = -10.0  # No valid metrics to evaluate
        elif aggregation == 'weighted_sum':
            total_reward = sum(
                data['weight'] * data['raw_reward']
                for data in metric_rewards.values()
            )
        elif aggregation == 'pareto':
            # Pareto: use minimum normalized reward
            # This encourages balanced improvement across all metrics
            normalized_rewards = [data['normalized_reward'] for data in metric_rewards.values()]
            total_reward = min(normalized_rewards)
        elif aggregation == 'product':
            # Product of normalized rewards (all must be positive)
            normalized_rewards = [data['normalized_reward'] for data in metric_rewards.values()]
            total_reward = np.prod(normalized_rewards)
        elif aggregation == 'tchebyshev':
            # Tchebyshev: minimize max weighted deviation
            weighted_deviations = [
                data['weight'] * abs(data['predicted'] - data['target'])
                for data in metric_rewards.values()
            ]
            total_reward = -max(weighted_deviations)
        else:
            total_reward = sum(data['raw_reward'] for data in metric_rewards.values())
        
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
        
        total_reward = total_reward + validation_bonus
        
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
        steps_per_epoch: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_targets: Optional[List[float]] = None,
        eval_aa_seqs: Optional[List[str]] = None,
        early_stopping_patience: int = 5,
        save_stats_path: Optional[str] = "models/ppo_training_stats.json",
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Train the LoRA model using PPO with evaluation tracking.
        
        Args:
            prompts: Training prompts
            target_efficiencies: Target TE values
            amino_acid_sequences: Amino acid sequences
            num_epochs: Number of training epochs (default: 10)
                - 5-20 epochs typical for PPO
                - More epochs may lead to reward hacking
            steps_per_epoch: Training steps per epoch (default: 100)
                - More steps = more samples per epoch
            eval_prompts: Optional held-out prompts for evaluation
            eval_targets: Optional held-out target values
            eval_aa_seqs: Optional held-out amino acid sequences
            early_stopping_patience: Epochs without reward improvement before stopping (default: 5)
            save_stats_path: Path to save training statistics JSON
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results and statistics
            
        Monitoring PPO Training:
            Watch these metrics during training:
            
            1. **Average Reward** ↑: Should increase, indicates policy improvement
            2. **Reward Variance**: High initially, should stabilize
            3. **Min/Max Reward**: Shows range of outcomes
            4. **Early Stopping**: Triggers if rewards plateau
            
        Example:
            >>> results = ppo_trainer.train(
            ...     prompts, targets, aa_seqs,
            ...     num_epochs=20,
            ...     steps_per_epoch=50,
            ...     early_stopping_patience=5
            ... )
            >>> print(f"Best epoch: {results['best_epoch']}")
        """
        print("=" * 60)
        print("Starting PPO Training")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print("=" * 60)
        
        # Tracking
        all_epoch_stats = []
        best_avg_reward = -float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            epoch_rewards = []
            epoch_step_stats = []
            
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
                epoch_step_stats.append(stats)
                
                if verbose and (step + 1) % 20 == 0:
                    print(f"  Step {step + 1}/{steps_per_epoch}, "
                          f"Avg Reward: {stats['avg_reward']:.4f}")
            
            # Epoch statistics
            avg_epoch_reward = np.mean(epoch_rewards)
            std_epoch_reward = np.std(epoch_rewards)
            min_epoch_reward = np.min(epoch_rewards)
            max_epoch_reward = np.max(epoch_rewards)
            
            epoch_summary = {
                'epoch': epoch + 1,
                'avg_reward': avg_epoch_reward,
                'std_reward': std_epoch_reward,
                'min_reward': min_epoch_reward,
                'max_reward': max_epoch_reward,
                'steps': epoch_step_stats
            }
            all_epoch_stats.append(epoch_summary)
            
            # Check improvement
            improvement_marker = ""
            if avg_epoch_reward > best_avg_reward:
                best_avg_reward = avg_epoch_reward
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                improvement_marker = " ★ (best)"
            else:
                epochs_without_improvement += 1
            
            print(f"\nEpoch {epoch + 1} Summary:{improvement_marker}")
            print(f"  Avg Reward:  {avg_epoch_reward:.4f} ± {std_epoch_reward:.4f}")
            print(f"  Range:       [{min_epoch_reward:.4f}, {max_epoch_reward:.4f}]")
            
            # Evaluate on held-out set if provided
            if eval_prompts is not None and eval_targets is not None and eval_aa_seqs is not None:
                eval_data = self.generate_and_score_batch(eval_prompts, eval_targets, eval_aa_seqs)
                eval_avg = np.mean(eval_data['rewards'])
                print(f"  Eval Reward: {eval_avg:.4f}")
                epoch_summary['eval_reward'] = eval_avg
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
                print(f"  No improvement for {early_stopping_patience} epochs")
                print(f"  Best avg reward: {best_avg_reward:.4f} at epoch {best_epoch}")
                break
        
        # Save statistics
        if save_stats_path:
            self.training_stats['epoch_stats'] = all_epoch_stats
            self.training_stats['best_epoch'] = best_epoch
            self.training_stats['best_avg_reward'] = best_avg_reward
            self.save_training_stats(save_stats_path)
        
        # Final summary
        print("\n" + "=" * 60)
        print("PPO Training Complete")
        print("=" * 60)
        print(f"  Total epochs: {epoch + 1}")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best avg reward: {best_avg_reward:.4f}")
        
        return {
            'best_epoch': best_epoch,
            'best_avg_reward': best_avg_reward,
            'total_epochs': epoch + 1,
            'epoch_stats': all_epoch_stats,
            'stopped_early': epochs_without_improvement >= early_stopping_patience
        }
    
    def save_training_stats(self, path: str):
        """Save training statistics to file."""
        import json
        from pathlib import Path as FilePath
        FilePath(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
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
