"""
Multi-metric critic model for predicting multiple RNA properties.
Supports translation efficiency, mRNA half-life, and other metrics.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

# Conditional import for type hints
if TYPE_CHECKING:
    from src.training.evaluation import EvaluationTracker, TrainingConfig


class MultiMetricCritic(nn.Module):
    """
    Multi-task critic that predicts multiple RNA properties from embeddings.
    
    Architecture:
        - Shared encoder: processes embeddings
        - Separate heads: one per metric (TE, half-life, etc.)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        shared_dims: List[int] = [512, 256],
        metrics: List[str] = ['translation_efficiency', 'half_life'],
        head_dims: List[int] = [128],
        dropout: float = 0.3,
        activation: str = "relu",
        num_cell_lines: int = 0,
        cell_embedding_dim: int = 32
    ):
        """
        Initialize multi-metric critic.
        
        Args:
            input_dim: Dimension of RNA embeddings
            shared_dims: Hidden dimensions for shared encoder
            metrics: List of metrics to predict
            head_dims: Hidden dimensions for each metric head
            dropout: Dropout probability
            activation: Activation function
            num_cell_lines: Number of cell lines to embed (0 to disable)
            cell_embedding_dim: Dimension of cell line embeddings
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.metrics = metrics
        self.n_metrics = len(metrics)
        self.num_cell_lines = num_cell_lines
        
        # Cell Line Embedding
        if num_cell_lines > 0:
            self.cell_embedding = nn.Embedding(num_cell_lines, cell_embedding_dim)
            self.combined_input_dim = input_dim + cell_embedding_dim
        else:
            self.cell_embedding = None
            self.combined_input_dim = input_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared encoder
        encoder_layers = []
        prev_dim = self.combined_input_dim
        for hidden_dim in shared_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.shared_encoder = nn.Sequential(*encoder_layers)
        
        # Metric-specific heads
        self.heads = nn.ModuleDict()
        for metric in metrics:
            head_layers = []
            head_prev_dim = prev_dim
            for head_dim in head_dims:
                head_layers.extend([
                    nn.Linear(head_prev_dim, head_dim),
                    self.activation,
                    nn.Dropout(dropout)
                ])
                head_prev_dim = head_dim
            # Output layer (single value)
            head_layers.append(nn.Linear(head_prev_dim, 1))
            self.heads[metric] = nn.Sequential(*head_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            embeddings: RNA embeddings [B, D]
            cell_line_indices: Optional cell line IDs [B]
            return_dict: If True, return dict; else return stacked tensor
            
        Returns:
            Dictionary mapping metric names to predictions [batch_size, 1]
            or stacked tensor [batch_size, n_metrics]
        """
        # Combine inputs if cell line is used
        # Combine inputs if cell line is used
        if self.cell_embedding is not None:
             if cell_line_indices is not None:
                cell_emb = self.cell_embedding(cell_line_indices)
                x = torch.cat([embeddings, cell_emb], dim=1)
             else:
                # Fallback: concatenate zeros if cell info missing but model expects it
                # Create zeros on correct device
                device = embeddings.device
                batch_size = embeddings.shape[0]
                zeros = torch.zeros(
                    batch_size, 
                    self.cell_embedding.embedding_dim, 
                    device=device
                )
                x = torch.cat([embeddings, zeros], dim=1)
        else:
            x = embeddings
            
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Metric-specific predictions
        predictions = {}
        for metric in self.metrics:
            predictions[metric] = self.heads[metric](shared_features)
        
        if return_dict:
            return predictions
        else:
            # Stack predictions for multi-task loss
            return torch.cat([predictions[m] for m in self.metrics], dim=1)
    
    def predict(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None,
        metrics: Optional[List[str]] = None,
        return_numpy: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions.
        
        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            cell_line_indices: Optional cell line IDs [batch_size]
            metrics: Which metrics to predict (default: all)
            return_numpy: If True, return numpy arrays
            
        Returns:
            Dictionary mapping metric names to predictions
        """
        self.eval()
        with torch.no_grad():
            all_preds = self.forward(embeddings, cell_line_indices=cell_line_indices, return_dict=True)
        
        if metrics is None:
            metrics = self.metrics
        
        predictions = {}
        for metric in metrics:
            pred = all_preds[metric].squeeze(-1)
            if return_numpy:
                pred = pred.cpu().numpy()
            predictions[metric] = pred
        
        return predictions
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiMetricTrainer:
    """
    Trainer for multi-metric critic model.
    """
    
    def __init__(
        self,
        model: MultiMetricCritic,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        metric_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize multi-metric trainer.
        
        Args:
            model: Multi-metric critic model
            learning_rate: Learning rate
            weight_decay: L2 regularization
            metric_weights: Loss weights for each metric (default: equal)
            device: Device to train on
        """
        self.model = model
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Metric weights for loss
        if metric_weights is None:
            metric_weights = {m: 1.0 for m in model.metrics}
        self.metric_weights = metric_weights
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function (MSE for each metric)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = {m: [] for m in model.metrics}
        self.val_losses = {m: [] for m in model.metrics}
        self.train_losses['total'] = []
        self.val_losses['total'] = []
    
    def train_epoch(
        self,
        train_loader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        Expects train_loader to yield (inputs_dict, targets_dict).
        
        Args:
            train_loader: DataLoader with (inputs_dict, targets_dict) tuples
            verbose: Whether to print progress
            
        Returns:
            Dictionary of average losses per metric
        """
        self.model.train()
        epoch_losses = {m: 0.0 for m in self.model.metrics}
        epoch_losses['total'] = 0.0
        num_batches = 0
        
        for batch_idx, (inputs_dict, targets_dict) in enumerate(train_loader):
            # Move inputs to device
            emb = inputs_dict['embedding'].to(self.device)
            cell_idx = inputs_dict.get('cell_line_idx')
            if cell_idx is not None:
                cell_idx = cell_idx.to(self.device)
            
            # Move targets
            targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}
            
            # Forward pass
            predictions = self.model(emb, cell_line_indices=cell_idx, return_dict=True)
            
            # Compute loss for each metric
            total_loss = 0.0
            metric_losses = {}
            for metric in self.model.metrics:
                if metric in targets_dict:
                    pred = predictions[metric].squeeze(-1)
                    target = targets_dict[metric]
                    loss = self.criterion(pred, target)
                    weighted_loss = self.metric_weights.get(metric, 1.0) * loss
                    metric_losses[metric] = loss.item()
                    total_loss += weighted_loss
                    epoch_losses[metric] += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                loss_str = ', '.join([f"{m}: {metric_losses.get(m, 0):.4f}" 
                                     for m in self.model.metrics])
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, {loss_str}")
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        for metric in self.model.metrics:
            self.train_losses[metric].append(avg_losses[metric])
        self.train_losses['total'].append(avg_losses['total'])
        
        return avg_losses
    
    def validate(self, val_loader) -> Dict[str, tuple]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Dictionary mapping metric names to (loss, R²) tuples
        """
        self.model.eval()
        total_losses = {m: 0.0 for m in self.model.metrics}
        num_batches = 0
        
        all_predictions = {m: [] for m in self.model.metrics}
        all_targets = {m: [] for m in self.model.metrics}
        
        with torch.no_grad():
            for inputs_dict, targets_dict in val_loader:
                emb = inputs_dict['embedding'].to(self.device)
                cell_idx = inputs_dict.get('cell_line_idx')
                if cell_idx is not None:
                    cell_idx = cell_idx.to(self.device)
                
                targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}
                
                predictions = self.model(emb, cell_line_indices=cell_idx, return_dict=True)
                
                for metric in self.model.metrics:
                    if metric in targets_dict:
                        pred = predictions[metric].squeeze(-1)
                        target = targets_dict[metric]
                        loss = self.criterion(pred, target)
                        total_losses[metric] += loss.item()
                        
                        all_predictions[metric].append(pred.cpu())
                        all_targets[metric].append(target.cpu())
                
                num_batches += 1
        
        # Compute metrics
        results = {}
        for metric in self.model.metrics:
            if all_predictions[metric]:
                avg_loss = total_losses[metric] / num_batches
                self.val_losses[metric].append(avg_loss)
                
                # Calculate R²
                preds = torch.cat(all_predictions[metric])
                targets = torch.cat(all_targets[metric])
                ss_res = torch.sum((targets - preds) ** 2)
                ss_tot = torch.sum((targets - targets.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                results[metric] = (avg_loss, r2.item())
        
        return results
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metric_weights': self.metric_weights,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', self.train_losses)
        self.val_losses = checkpoint.get('val_losses', self.val_losses)
        self.metric_weights = checkpoint.get('metric_weights', self.metric_weights)
        print(f"Checkpoint loaded from {path}")
    
    def train_with_evaluation(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_path: str = "models/critic_best.pt",
        save_history_path: Optional[str] = "models/training_history.json",
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Complete training loop with evaluation tracking and early stopping.
        
        This method provides a comprehensive training workflow:
        1. Trains for specified epochs with validation after each
        2. Tracks metrics (loss, R²) for all target metrics
        3. Implements early stopping to prevent overfitting
        4. Saves best model checkpoint automatically
        5. Evaluates on test set at the end
        
        Args:
            train_loader: Training data DataLoader
            val_loader: Validation/dev data DataLoader
            test_loader: Optional test data DataLoader (for final evaluation)
            num_epochs: Maximum number of training epochs (default: 50)
                - 30-100 epochs typically sufficient for critic
                - More epochs if data is large and model is complex
            early_stopping_patience: Epochs without improvement before stopping (default: 10)
                - Higher (15-20) for noisy data
                - Lower (5-7) for quick experiments
            checkpoint_path: Where to save best model checkpoint
            save_history_path: Where to save training history JSON (None to skip)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results:
                - 'best_epoch': Epoch with best validation performance
                - 'best_val_r2': Best validation R² achieved
                - 'final_test_results': Test set results (if test_loader provided)
                - 'training_history': Dict with train/val loss/R² per epoch
                
        Example:
            >>> trainer = MultiMetricTrainer(model)
            >>> results = trainer.train_with_evaluation(
            ...     train_loader, val_loader, test_loader,
            ...     num_epochs=100,
            ...     early_stopping_patience=15
            ... )
            >>> print(f"Best R²: {results['best_val_r2']:.4f}")
            
        Monitoring Progress:
            Watch these indicators during training:
            - Train/val loss should decrease (lower is better)
            - R² should increase toward 1.0 (higher is better)
            - Gap between train/val metrics indicates overfitting
            - ★ symbol indicates best epoch so far
        """
        # Lazy import to avoid circular dependencies
        from src.training.evaluation import EvaluationTracker, TrainingConfig
        
        # Create tracker with config
        config = TrainingConfig(
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            primary_metric='loss',
            higher_is_better=False,
            verbose=verbose
        )
        tracker = EvaluationTracker(config)
        tracker.start_training()
        
        best_val_r2 = -float('inf')
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, verbose=False)
            train_loss = train_metrics.get('total', sum(train_metrics.values()) / len(train_metrics))
            tracker.log_train_metrics(epoch, {'loss': train_loss, **train_metrics})
            
            # Validate
            val_results = self.validate(val_loader)
            
            # Aggregate validation metrics
            val_loss = np.mean([v[0] for v in val_results.values()])
            val_r2 = np.mean([v[1] for v in val_results.values()])
            
            val_metrics = {
                'loss': val_loss,
                'avg_r2': val_r2,
                **{f"{k}_r2": v[1] for k, v in val_results.items()}
            }
            
            is_best = tracker.log_val_metrics(epoch, val_metrics)
            
            # Track best R²
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
            
            # Print progress
            if verbose:
                tracker.print_epoch_summary(
                    epoch,
                    {'loss': train_loss},
                    {'loss': val_loss, 'r2': val_r2}
                )
            
            # Check early stopping
            if tracker.should_stop():
                break
        
        # Evaluate on test set if provided
        test_results = None
        if test_loader is not None:
            if verbose:
                print("\nEvaluating on test set...")
            
            from src.training.evaluation import evaluate_model
            test_results = evaluate_model(
                self.model,
                test_loader,
                self.device
            )
            tracker.log_test_results(test_results)
        
        # Save history
        if save_history_path:
            tracker.save_history(save_history_path)
        
        # Print summary
        if verbose:
            tracker.print_summary()
        
        return {
            'best_epoch': tracker.best_epoch + 1,
            'best_val_r2': best_val_r2,
            'final_test_results': test_results,
            'training_history': {
                'train': tracker.train_history,
                'val': tracker.val_history
            },
            'stopped_early': tracker._stopped_early
        }


if __name__ == "__main__":
    # Test multi-metric critic
    print("=" * 60)
    print("Multi-Metric Critic Test")
    print("=" * 60)
    
    # Create model
    critic = MultiMetricCritic(
        input_dim=1024,
        shared_dims=[512, 256],
        metrics=['translation_efficiency', 'half_life'],
        head_dims=[128],
        dropout=0.3
    )
    
    print(f"Model architecture:")
    print(critic)
    print(f"\nTotal parameters: {critic.get_num_parameters():,}")
    print(f"Metrics: {critic.metrics}")
    
    # Test forward pass
    batch_size = 16
    test_embeddings = torch.randn(batch_size, 1024)
    
    predictions = critic(test_embeddings, return_dict=True)
    print(f"\nInput shape: {test_embeddings.shape}")
    for metric, pred in predictions.items():
        print(f"  {metric}: {pred.shape}")
    
    # Test prediction method
    pred_dict = critic.predict(test_embeddings, return_numpy=True)
    print(f"\nPredictions (first 3 samples):")
    for metric, values in pred_dict.items():
        print(f"  {metric}: {values[:3]}")
