"""
Critic Ensemble for combining multiple independent critic models.
Enables multi-objective optimization with critics trained on separate datasets.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import numpy as np
import os
import json


class CriticEnsemble:
    """
    Ensemble of multiple critic models for different metrics.
    
    Each critic can be trained on a separate dataset, then combined
    during reward computation using configurable aggregation strategies.
    
    This enables:
        - Training a TE critic on Dataset A
        - Training a half-life critic on Dataset B
        - Combining both for PPO reward computation
    
    Usage:
        >>> ensemble = CriticEnsemble()
        >>> ensemble.add_critic('translation_efficiency', te_critic, weight=0.7)
        >>> ensemble.add_critic('half_life', hl_critic, weight=0.3)
        >>> 
        >>> # Get predictions from all critics
        >>> predictions = ensemble.predict_all(embeddings)
        >>> 
        >>> # Compute composite reward
        >>> reward = ensemble.compute_reward(
        ...     embeddings,
        ...     target_metrics={'translation_efficiency': 0.85, 'half_life': 12.0}
        ... )
    """
    
    def __init__(self, device: str = None):
        """
        Initialize critic ensemble.
        
        Args:
            device: Device to use for inference
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.critics: Dict[str, nn.Module] = {}
        self.metric_weights: Dict[str, float] = {}
        self.metric_info: Dict[str, Dict] = {}  # Additional metadata per metric
    
    def add_critic(
        self,
        metric_name: str,
        critic: nn.Module,
        weight: float = 1.0,
        description: str = ""
    ):
        """
        Add a trained critic for a specific metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'translation_efficiency')
            critic: Trained critic model
            weight: Weight for reward aggregation (default: 1.0)
            description: Optional description of this critic
        """
        critic = critic.to(self.device)
        critic.eval()
        self.critics[metric_name] = critic
        self.metric_weights[metric_name] = weight
        self.metric_info[metric_name] = {
            'description': description,
            'added_at': str(np.datetime64('now'))
        }
        print(f"Added critic for '{metric_name}' with weight {weight}")
    
    def remove_critic(self, metric_name: str):
        """Remove a critic from the ensemble."""
        if metric_name in self.critics:
            del self.critics[metric_name]
            del self.metric_weights[metric_name]
            del self.metric_info[metric_name]
            print(f"Removed critic for '{metric_name}'")
    
    def set_weight(self, metric_name: str, weight: float):
        """Update weight for a metric."""
        if metric_name in self.metric_weights:
            self.metric_weights[metric_name] = weight
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics in the ensemble."""
        return list(self.critics.keys())
    
    def predict_all(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get predictions from all critics.
        
        Args:
            embeddings: RNA embeddings [batch_size, embedding_dim]
            cell_line_indices: Optional cell line indices [batch_size]
            return_dict: If True, return dict; else return stacked tensor
            
        Returns:
            Dictionary mapping metric names to predictions,
            or stacked tensor [batch_size, n_metrics]
        """
        embeddings = embeddings.to(self.device)
        if cell_line_indices is not None:
            cell_line_indices = cell_line_indices.to(self.device)
        
        predictions = {}
        for metric_name, critic in self.critics.items():
            with torch.no_grad():
                # Handle different critic interfaces
                if hasattr(critic, 'predict'):
                    pred = critic.predict(embeddings, cell_line_indices)
                else:
                    pred = critic(embeddings, cell_line_indices)
                
                if isinstance(pred, dict):
                    pred = pred.get(metric_name, list(pred.values())[0])
                
                if pred.dim() > 1:
                    pred = pred.squeeze(-1)
                
                predictions[metric_name] = pred
        
        if return_dict:
            return predictions
        else:
            return torch.stack([predictions[m] for m in self.critics.keys()], dim=1)
    
    def compute_reward(
        self,
        embeddings: torch.Tensor,
        target_metrics: Dict[str, float],
        cell_line_indices: Optional[torch.Tensor] = None,
        aggregation: str = 'weighted_sum',
        normalize: bool = True
    ) -> float:
        """
        Compute composite reward from all critics.
        
        Args:
            embeddings: RNA embeddings [1, embedding_dim] (single sequence)
            target_metrics: Dictionary of target values for each metric
            cell_line_indices: Optional cell line indices
            aggregation: Reward aggregation strategy
                - 'weighted_sum': Sum of weighted rewards
                - 'pareto': Minimum normalized reward
                - 'product': Product of normalized rewards
                - 'tchebyshev': Minimize max weighted deviation
            normalize: Whether to normalize individual rewards
            
        Returns:
            Composite reward value
        """
        predictions = self.predict_all(embeddings, cell_line_indices)
        
        # Compute individual metric rewards
        metric_rewards = {}
        for metric_name, pred in predictions.items():
            if metric_name not in target_metrics:
                continue
            
            predicted = pred.item() if isinstance(pred, torch.Tensor) else pred
            target = target_metrics[metric_name]
            weight = self.metric_weights.get(metric_name, 1.0)
            
            # Raw reward: negative absolute error
            raw_reward = -abs(predicted - target)
            
            metric_rewards[metric_name] = {
                'predicted': predicted,
                'target': target,
                'raw_reward': raw_reward,
                'weight': weight
            }
        
        if not metric_rewards:
            return -10.0  # No valid metrics
        
        # Normalize if requested
        if normalize:
            for data in metric_rewards.values():
                # Sigmoid-like normalization to [0, 1]
                data['normalized'] = 1.0 / (1.0 + abs(data['raw_reward']))
        else:
            for data in metric_rewards.values():
                data['normalized'] = data['raw_reward']
        
        # Aggregate based on strategy
        if aggregation == 'weighted_sum':
            total = sum(d['weight'] * d['raw_reward'] for d in metric_rewards.values())
        elif aggregation == 'pareto':
            total = min(d['normalized'] for d in metric_rewards.values())
        elif aggregation == 'product':
            total = np.prod([d['normalized'] for d in metric_rewards.values()])
        elif aggregation == 'tchebyshev':
            total = -max(
                d['weight'] * abs(d['predicted'] - d['target'])
                for d in metric_rewards.values()
            )
        else:
            total = sum(d['raw_reward'] for d in metric_rewards.values())
        
        return float(total)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Alias for predict_all to match nn.Module interface."""
        return self.predict_all(embeddings, cell_line_indices, return_dict)
    
    def __call__(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Make ensemble callable like nn.Module."""
        return self.predict_all(embeddings, cell_line_indices, return_dict)
    
    def save(self, directory: str):
        """
        Save all critics to a directory.
        
        Creates:
            - {directory}/ensemble_config.json
            - {directory}/{metric_name}_critic.pt for each critic
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save config
        config = {
            'metrics': list(self.critics.keys()),
            'weights': self.metric_weights,
            'info': self.metric_info
        }
        with open(os.path.join(directory, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save each critic
        for metric_name, critic in self.critics.items():
            path = os.path.join(directory, f'{metric_name}_critic.pt')
            torch.save({
                'model_state_dict': critic.state_dict(),
                'metric_name': metric_name
            }, path)
        
        print(f"Ensemble saved to {directory}")
    
    def load(self, directory: str, critic_class=None):
        """
        Load critics from a directory.
        
        Args:
            directory: Directory containing saved ensemble
            critic_class: Class to instantiate critics (autodetected if not provided)
        """
        # Load config
        config_path = os.path.join(directory, 'ensemble_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.metric_weights = config['weights']
        self.metric_info = config.get('info', {})
        
        # Load each critic
        for metric_name in config['metrics']:
            path = os.path.join(directory, f'{metric_name}_critic.pt')
            checkpoint = torch.load(path, map_location=self.device)
            
            if critic_class is not None:
                critic = critic_class(metric_name=metric_name)
                critic.load_state_dict(checkpoint['model_state_dict'])
                self.critics[metric_name] = critic.to(self.device)
        
        print(f"Loaded ensemble with metrics: {list(self.critics.keys())}")
    
    def summary(self) -> str:
        """Return summary of the ensemble."""
        lines = ["CriticEnsemble Summary", "=" * 40]
        lines.append(f"Device: {self.device}")
        lines.append(f"Number of critics: {len(self.critics)}")
        lines.append("")
        lines.append("Critics:")
        for metric_name in self.critics:
            weight = self.metric_weights.get(metric_name, 1.0)
            info = self.metric_info.get(metric_name, {})
            desc = info.get('description', '')
            lines.append(f"  - {metric_name} (weight: {weight:.2f})")
            if desc:
                lines.append(f"    {desc}")
        return "\n".join(lines)
    
    def __repr__(self):
        return f"CriticEnsemble(metrics={list(self.critics.keys())})"


if __name__ == "__main__":
    print("=" * 60)
    print("CriticEnsemble Test")
    print("=" * 60)
    
    # Import SingleMetricCritic for testing
    from single_metric_critic import SingleMetricCritic
    
    # Create mock critics
    te_critic = SingleMetricCritic(
        input_dim=1024,
        hidden_dims=[256, 128],
        metric_name='translation_efficiency'
    )
    
    hl_critic = SingleMetricCritic(
        input_dim=1024,
        hidden_dims=[256, 128],
        metric_name='half_life'
    )
    
    # Create ensemble
    ensemble = CriticEnsemble()
    ensemble.add_critic('translation_efficiency', te_critic, weight=0.7)
    ensemble.add_critic('half_life', hl_critic, weight=0.3)
    
    print("\n" + ensemble.summary())
    
    # Test predictions
    mock_embeddings = torch.randn(4, 1024)
    predictions = ensemble.predict_all(mock_embeddings)
    
    print("\nPredictions:")
    for metric, preds in predictions.items():
        print(f"  {metric}: shape={preds.shape}")
    
    # Test reward computation
    single_embedding = torch.randn(1, 1024)
    reward = ensemble.compute_reward(
        single_embedding,
        target_metrics={'translation_efficiency': 0.8, 'half_life': 12.0},
        aggregation='weighted_sum'
    )
    print(f"\nWeighted sum reward: {reward:.4f}")
    
    reward_pareto = ensemble.compute_reward(
        single_embedding,
        target_metrics={'translation_efficiency': 0.8, 'half_life': 12.0},
        aggregation='pareto'
    )
    print(f"Pareto reward: {reward_pareto:.4f}")
    
    print("\n✓ CriticEnsemble test passed!")
