"""
Evaluation and training tracking utilities.
Provides metrics computation, progress logging, and early stopping.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
import json
from datetime import datetime


@dataclass
class TrainingConfig:
    """
    Configuration for training with evaluation.
    
    Hyperparameters:
        num_epochs: Total training epochs (default: 50)
            - Critic: 30-100 epochs typically sufficient
            - PPO: 5-20 epochs due to RL sample efficiency
            
        learning_rate: Optimizer learning rate (default: 1e-3)
            - Critic: 1e-3 to 1e-4 works well
            - PPO/LoRA: 1e-5 to 1e-6 for stable fine-tuning
            
        early_stopping_patience: Epochs without improvement before stopping (default: 10)
            - Higher (15-20) for noisy validation metrics
            - Lower (5-7) for quick experiments
            
        early_stopping_min_delta: Minimum improvement to reset patience (default: 1e-4)
            - Increase if validation is noisy
            
        eval_every_n_epochs: Frequency of validation evaluation (default: 1)
            - Set higher (5-10) for very large datasets
            
        log_every_n_steps: Frequency of training progress logs (default: 10)
            - Lower for debugging, higher for cleaner output
            
        save_best_model: Whether to save best checkpoint (default: True)
        
        model_save_path: Path for saving checkpoints (default: "models/")
        
        primary_metric: Metric for early stopping/best model (default: "loss")
            - "loss": Lower is better
            - "r2": Higher is better
            
        higher_is_better: Whether higher metric values are better (default: False)
            - False for loss, True for R², accuracy
            
    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=100,
        ...     learning_rate=1e-3,
        ...     early_stopping_patience=15,
        ...     primary_metric="r2",
        ...     higher_is_better=True
        ... )
    """
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    eval_every_n_epochs: int = 1
    log_every_n_steps: int = 10
    save_best_model: bool = True
    model_save_path: str = "models/"
    primary_metric: str = "loss"
    higher_is_better: bool = False
    verbose: bool = True


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute regression metrics (R², MSE, MAE).
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        prefix: Optional prefix for metric names (e.g., "val_")
        
    Returns:
        Dictionary with r2, mse, mae metrics
        
    Metrics Explained:
        - R² (coefficient of determination): 1.0 is perfect, 0.0 is baseline
          Values > 0.7 are generally good for biological data
        - MSE (mean squared error): Penalizes large errors more
        - MAE (mean absolute error): More interpretable, same units as target
        
    Example:
        >>> metrics = compute_regression_metrics(preds, targets, prefix="val_")
        >>> print(f"Validation R²: {metrics['val_r2']:.4f}")
    """
    # Ensure tensors
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.tensor(targets)
    
    # Flatten if needed
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # MSE
    mse = torch.mean((predictions - targets) ** 2).item()
    
    # MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # R²
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = (1 - (ss_res / (ss_tot + 1e-8))).item()
    
    return {
        f"{prefix}r2": r2,
        f"{prefix}mse": mse,
        f"{prefix}mae": mae
    }


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    metrics_to_evaluate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model on a DataLoader.
    
    Args:
        model: PyTorch model (MultiMetricCritic or similar)
        data_loader: DataLoader yielding (inputs_dict, targets_dict)
        device: Device for computation
        metrics_to_evaluate: List of metric names to evaluate (None = all)
        
    Returns:
        Dictionary mapping metric names to their evaluation results
        
    Example:
        >>> results = evaluate_model(critic, test_loader, device)
        >>> print(f"TE R²: {results['translation_efficiency']['r2']:.4f}")
    """
    model.eval()
    
    all_predictions = {}
    all_targets = {}
    
    with torch.no_grad():
        for inputs_dict, targets_dict in data_loader:
            # Move to device
            emb = inputs_dict['embedding'].to(device)
            cell_idx = inputs_dict.get('cell_line_idx')
            if cell_idx is not None:
                cell_idx = cell_idx.to(device)
            
            # Forward pass
            predictions = model(emb, cell_line_indices=cell_idx, return_dict=True)
            
            # Collect predictions and targets
            for metric_name, pred in predictions.items():
                if metrics_to_evaluate and metric_name not in metrics_to_evaluate:
                    continue
                    
                if metric_name not in all_predictions:
                    all_predictions[metric_name] = []
                    all_targets[metric_name] = []
                
                all_predictions[metric_name].append(pred.squeeze(-1).cpu())
                
                if metric_name in targets_dict:
                    all_targets[metric_name].append(targets_dict[metric_name].cpu())
    
    # Compute metrics for each target
    results = {}
    for metric_name in all_predictions:
        preds = torch.cat(all_predictions[metric_name])
        targets = torch.cat(all_targets[metric_name])
        
        results[metric_name] = compute_regression_metrics(preds, targets)
    
    return results


class EvaluationTracker:
    """
    Tracks training progress, computes metrics, and handles early stopping.
    
    Features:
        - Logs training/validation metrics per epoch
        - Computes and tracks R², MSE, MAE
        - Early stopping with configurable patience
        - Saves training history to JSON
        - Progress visualization helpers
        
    Monitoring Training Progress:
        The tracker logs metrics that help you understand training health:
        
        1. **Loss decreasing**: Both train and val loss should decrease initially
        2. **R² increasing**: Should approach 1.0 for good models
        3. **Train-Val gap**: Large gaps indicate overfitting
        4. **Early stopping**: Triggers when validation stops improving
        
    Example:
        >>> tracker = EvaluationTracker(TrainingConfig(num_epochs=50))
        >>> 
        >>> for epoch in range(config.num_epochs):
        ...     # Training
        ...     train_loss = train_one_epoch()
        ...     tracker.log_train_metrics(epoch, {'loss': train_loss})
        ...     
        ...     # Validation
        ...     val_metrics = evaluate(model, val_loader)
        ...     tracker.log_val_metrics(epoch, val_metrics)
        ...     
        ...     # Check early stopping
        ...     if tracker.should_stop():
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
        >>> 
        >>> tracker.save_history("training_history.json")
        >>> tracker.print_summary()
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the evaluation tracker.
        
        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        
        # History storage
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []
        self.test_results: Optional[Dict[str, Any]] = None
        
        # Early stopping state
        self.best_metric_value = float('inf') if not self.config.higher_is_better else float('-inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self._stopped_early = False
        
        # Timing
        self.start_time = None
        self.epoch_times: List[float] = []
    
    def start_training(self):
        """Mark the start of training for timing."""
        self.start_time = datetime.now()
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Configuration:")
            print(f"  Epochs: {self.config.num_epochs}")
            print(f"  Early stopping patience: {self.config.early_stopping_patience}")
            print(f"  Primary metric: {self.config.primary_metric} ({'↑' if self.config.higher_is_better else '↓'})")
            print(f"{'='*60}\n")
    
    def log_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dictionary of metric names to values
        """
        entry = {'epoch': epoch, **metrics}
        self.train_history.append(entry)
    
    def log_val_metrics(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Log validation metrics and check for improvement.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names to values
            
        Returns:
            True if this is the best epoch so far
        """
        entry = {'epoch': epoch, **metrics}
        self.val_history.append(entry)
        
        # Check improvement
        current_value = metrics.get(self.config.primary_metric, metrics.get('loss', 0))
        
        is_improvement = False
        if self.config.higher_is_better:
            is_improvement = current_value > self.best_metric_value + self.config.early_stopping_min_delta
        else:
            is_improvement = current_value < self.best_metric_value - self.config.early_stopping_min_delta
        
        if is_improvement:
            self.best_metric_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def should_stop(self) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if patience has been exceeded
        """
        if self.epochs_without_improvement >= self.config.early_stopping_patience:
            self._stopped_early = True
            if self.config.verbose:
                print(f"\n⚠ Early stopping triggered at epoch {len(self.val_history)}")
                print(f"  Best {self.config.primary_metric}: {self.best_metric_value:.4f} at epoch {self.best_epoch + 1}")
            return True
        return False
    
    def log_test_results(self, results: Dict[str, Any]):
        """Log final test set evaluation results."""
        self.test_results = results
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Test Set Evaluation Results")
            print(f"{'='*60}")
            for metric_name, values in results.items():
                if isinstance(values, dict):
                    print(f"  {metric_name}:")
                    for k, v in values.items():
                        print(f"    {k}: {v:.4f}")
                else:
                    print(f"  {metric_name}: {values:.4f}")
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Print a summary line for the current epoch."""
        if not self.config.verbose:
            return
            
        train_loss = train_metrics.get('loss', train_metrics.get('total', 0))
        val_loss = val_metrics.get('loss', val_metrics.get('total', 0))
        val_r2 = val_metrics.get('r2', val_metrics.get('avg_r2', 'N/A'))
        
        improvement = "★" if self.epochs_without_improvement == 0 else " "
        
        if isinstance(val_r2, float):
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val R²: {val_r2:.4f} {improvement}")
        else:
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} {improvement}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training run.
        
        Returns:
            Dictionary with training statistics
        """
        summary = {
            'total_epochs': len(self.train_history),
            'best_epoch': self.best_epoch + 1,
            'best_metric': self.config.primary_metric,
            'best_value': self.best_metric_value,
            'stopped_early': self._stopped_early,
        }
        
        if self.start_time:
            duration = datetime.now() - self.start_time
            summary['training_duration_seconds'] = duration.total_seconds()
        
        if self.train_history:
            summary['final_train_loss'] = self.train_history[-1].get('loss', 
                                          self.train_history[-1].get('total', None))
        
        if self.val_history:
            summary['final_val_loss'] = self.val_history[-1].get('loss',
                                        self.val_history[-1].get('total', None))
        
        return summary
    
    def print_summary(self):
        """Print a comprehensive training summary."""
        summary = self.get_training_summary()
        
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"  Total epochs: {summary['total_epochs']}")
        print(f"  Best epoch: {summary['best_epoch']}")
        print(f"  Best {summary['best_metric']}: {summary['best_value']:.4f}")
        
        if summary.get('stopped_early'):
            print(f"  ⚠ Training stopped early (patience exceeded)")
        
        if 'training_duration_seconds' in summary:
            mins = summary['training_duration_seconds'] / 60
            print(f"  Duration: {mins:.1f} minutes")
        
        if self.test_results:
            print(f"\nTest Results:")
            for metric, values in self.test_results.items():
                if isinstance(values, dict) and 'r2' in values:
                    print(f"  {metric} R²: {values['r2']:.4f}")
    
    def save_history(self, filepath: str):
        """
        Save training history to JSON file.
        
        Args:
            filepath: Path to save the history
        """
        history = {
            'config': {
                'num_epochs': self.config.num_epochs,
                'learning_rate': self.config.learning_rate,
                'early_stopping_patience': self.config.early_stopping_patience,
                'primary_metric': self.config.primary_metric,
            },
            'train_history': self.train_history,
            'val_history': self.val_history,
            'test_results': self.test_results,
            'summary': self.get_training_summary()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        if self.config.verbose:
            print(f"\n✓ Training history saved to {filepath}")
    
    def get_learning_curves(self) -> Dict[str, List[float]]:
        """
        Get data for plotting learning curves.
        
        Returns:
            Dictionary with 'epochs', 'train_loss', 'val_loss', 'val_r2'
            
        Example:
            >>> curves = tracker.get_learning_curves()
            >>> plt.plot(curves['epochs'], curves['train_loss'], label='Train')
            >>> plt.plot(curves['epochs'], curves['val_loss'], label='Val')
        """
        epochs = [e['epoch'] for e in self.train_history]
        train_loss = [e.get('loss', e.get('total', 0)) for e in self.train_history]
        val_loss = [e.get('loss', e.get('total', 0)) for e in self.val_history]
        val_r2 = [e.get('r2', e.get('avg_r2', None)) for e in self.val_history]
        
        return {
            'epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_r2': val_r2
        }


class PPOEvaluationTracker(EvaluationTracker):
    """
    Specialized evaluation tracker for PPO training.
    
    Tracks RL-specific metrics like rewards and policy statistics.
    
    PPO Training Monitoring:
        Key metrics to watch:
        
        1. **Average Reward**: Should increase over training
        2. **Reward Variance**: High variance is normal initially, should stabilize
        3. **KL Divergence**: Keep below target_kl (usually ~0.01)
        4. **Policy Loss**: Should decrease but can be noisy
        
    Example:
        >>> ppo_tracker = PPOEvaluationTracker(TrainingConfig(num_epochs=20))
        >>> 
        >>> for epoch in range(20):
        ...     for step in range(steps_per_epoch):
        ...         step_stats = ppo_step()
        ...         ppo_tracker.log_ppo_step(epoch, step, step_stats)
        ...     
        ...     ppo_tracker.log_epoch(epoch)
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        # PPO typically uses fewer epochs
        if config is None:
            config = TrainingConfig(
                num_epochs=20,
                early_stopping_patience=5,
                primary_metric="avg_reward",
                higher_is_better=True  # Higher rewards are better
            )
        super().__init__(config)
        
        self.step_history: List[Dict[str, Any]] = []
        self.epoch_rewards: List[List[float]] = []
        self.current_epoch_rewards: List[float] = []
    
    def log_ppo_step(
        self,
        epoch: int,
        step: int,
        stats: Dict[str, float]
    ):
        """
        Log metrics from a PPO training step.
        
        Args:
            epoch: Current epoch
            step: Current step within epoch
            stats: Step statistics (avg_reward, policy_loss, etc.)
        """
        entry = {'epoch': epoch, 'step': step, **stats}
        self.step_history.append(entry)
        
        if 'avg_reward' in stats:
            self.current_epoch_rewards.append(stats['avg_reward'])
        
        # Log progress
        if self.config.verbose and (step + 1) % self.config.log_every_n_steps == 0:
            reward = stats.get('avg_reward', 0)
            print(f"  Step {step+1}: Avg Reward = {reward:.4f}")
    
    def log_epoch(self, epoch: int):
        """Finalize logging for an epoch."""
        if self.current_epoch_rewards:
            epoch_stats = {
                'avg_reward': np.mean(self.current_epoch_rewards),
                'min_reward': np.min(self.current_epoch_rewards),
                'max_reward': np.max(self.current_epoch_rewards),
                'std_reward': np.std(self.current_epoch_rewards)
            }
            
            self.epoch_rewards.append(self.current_epoch_rewards.copy())
            self.current_epoch_rewards = []
            
            # Log as validation metrics for early stopping
            self.log_val_metrics(epoch, epoch_stats)
            
            if self.config.verbose:
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Avg Reward: {epoch_stats['avg_reward']:.4f}")
                print(f"  Min/Max Reward: {epoch_stats['min_reward']:.4f} / {epoch_stats['max_reward']:.4f}")


if __name__ == "__main__":
    # Test evaluation utilities
    print("=" * 60)
    print("Evaluation Utilities Test")
    print("=" * 60)
    
    # Test metric computation
    print("\n--- Testing compute_regression_metrics ---")
    preds = torch.randn(100)
    targets = preds + torch.randn(100) * 0.1  # Near-perfect predictions
    
    metrics = compute_regression_metrics(preds, targets)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    # Test tracker
    print("\n--- Testing EvaluationTracker ---")
    config = TrainingConfig(num_epochs=10, early_stopping_patience=3, verbose=True)
    tracker = EvaluationTracker(config)
    tracker.start_training()
    
    # Simulate training
    for epoch in range(10):
        # Simulate decreasing loss
        train_loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
        val_loss = 1.0 / (epoch + 1) + np.random.random() * 0.15
        val_r2 = 1 - val_loss
        
        tracker.log_train_metrics(epoch, {'loss': train_loss})
        tracker.log_val_metrics(epoch, {'loss': val_loss, 'r2': val_r2})
        tracker.print_epoch_summary(epoch, {'loss': train_loss}, {'loss': val_loss, 'r2': val_r2})
        
        if tracker.should_stop():
            break
    
    tracker.print_summary()
    
    print("\n✓ Evaluation utilities test passed!")
