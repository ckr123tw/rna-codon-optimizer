"""
Single-metric critic model for predicting individual RNA properties.
Designed to be trained independently on separate datasets.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class SingleMetricCritic(nn.Module):
    """
    Single-metric critic that predicts one RNA property from embeddings.
    
    Use this when training critics on separate datasets for different metrics,
    then combine them in a CriticEnsemble for multi-objective optimization.
    
    Architecture:
        - Encoder: processes embeddings with optional cell line conditioning
        - Single output head for the target metric
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = [512, 256],
        metric_name: str = 'translation_efficiency',
        dropout: float = 0.3,
        activation: str = "relu",
        num_cell_lines: int = 0,
        cell_embedding_dim: int = 32
    ):
        """
        Initialize single-metric critic.
        
        Args:
            input_dim: Dimension of RNA embeddings
            hidden_dims: Hidden layer dimensions
            metric_name: Name of the metric this critic predicts
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu", "tanh")
            num_cell_lines: Number of cell lines (0 to disable conditioning)
            cell_embedding_dim: Dimension of cell line embeddings
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.metric_name = metric_name
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
        
        # Build encoder
        encoder_layers = []
        prev_dim = self.combined_input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Output head
        self.output_head = nn.Linear(prev_dim, 1)
        
        # Normalization parameters (set during training)
        self.register_buffer('target_mean', torch.tensor(0.0))
        self.register_buffer('target_std', torch.tensor(1.0))
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: RNA embeddings [batch_size, input_dim]
            cell_line_indices: Optional cell line IDs [batch_size]
            
        Returns:
            Predictions [batch_size, 1]
        """
        # Combine with cell line embedding if available
        if self.cell_embedding is not None and cell_line_indices is not None:
            cell_emb = self.cell_embedding(cell_line_indices)
            x = torch.cat([embeddings, cell_emb], dim=1)
        elif self.cell_embedding is not None:
            # Fallback: use zeros
            batch_size = embeddings.shape[0]
            zeros = torch.zeros(
                batch_size,
                self.cell_embedding.embedding_dim,
                device=embeddings.device
            )
            x = torch.cat([embeddings, zeros], dim=1)
        else:
            x = embeddings
        
        # Encode and predict
        features = self.encoder(x)
        return self.output_head(features)
    
    def predict(
        self,
        embeddings: torch.Tensor,
        cell_line_indices: Optional[torch.Tensor] = None,
        denormalize: bool = True
    ) -> torch.Tensor:
        """
        Make predictions (inference mode).
        
        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            cell_line_indices: Optional cell line IDs
            denormalize: Whether to denormalize predictions
            
        Returns:
            Predictions on original scale
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(embeddings, cell_line_indices).squeeze(-1)
            if denormalize:
                preds = preds * self.target_std + self.target_mean
        return preds
    
    def set_normalization(self, mean: float, std: float):
        """Set normalization parameters."""
        self.target_mean = torch.tensor(mean, device=self.target_mean.device)
        self.target_std = torch.tensor(std, device=self.target_std.device)


class SingleMetricTrainer:
    """
    Trainer for single-metric critic models.
    
    Provides training with evaluation tracking and early stopping.
    """
    
    def __init__(
        self,
        model: SingleMetricCritic,
        learning_rate: float = 1e-3,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: SingleMetricCritic model
            learning_rate: Learning rate
            device: Device to use
        """
        self.model = model
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the critic model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_path: Path to save best model
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                embeddings, targets = batch[0].to(self.device), batch[1].to(self.device)
                cell_indices = batch[2].to(self.device) if len(batch) > 2 else None
                
                self.optimizer.zero_grad()
                predictions = self.model(embeddings, cell_indices).squeeze(-1)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    embeddings, targets = batch[0].to(self.device), batch[1].to(self.device)
                    cell_indices = batch[2].to(self.device) if len(batch) > 2 else None
                    
                    predictions = self.model(embeddings, cell_indices).squeeze(-1)
                    loss = self.criterion(predictions, targets)
                    val_loss += loss.item()
                    
                    all_preds.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Compute R²
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            ss_res = np.sum((all_targets - all_preds) ** 2)
            ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
            
            if verbose:
                marker = "★" if is_best else " "
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"R²: {r2:.4f} {marker}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        if verbose:
            print(f"\nBest epoch: {best_epoch} with val_loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric_name': self.model.metric_name,
            'target_mean': self.model.target_mean.item(),
            'target_std': self.model.target_std.item(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.set_normalization(
            checkpoint['target_mean'],
            checkpoint['target_std']
        )
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


def create_single_metric_loaders(
    embeddings: np.ndarray,
    targets: np.ndarray,
    cell_line_indices: Optional[np.ndarray] = None,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    normalize: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create data loaders for single-metric training.
    
    Args:
        embeddings: RNA embeddings [N, dim]
        targets: Target values [N]
        cell_line_indices: Optional cell line indices [N]
        train_ratio: Fraction for training
        batch_size: Batch size
        normalize: Whether to normalize targets
        random_seed: Random seed
        
    Returns:
        (train_loader, val_loader, normalization_params)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    n_samples = len(embeddings)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_ratio)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # Convert to tensors
    X_train = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    X_val = torch.tensor(embeddings[val_idx], dtype=torch.float32)
    
    # Normalize targets
    if normalize:
        mean = targets[train_idx].mean()
        std = targets[train_idx].std()
        y_train = torch.tensor((targets[train_idx] - mean) / (std + 1e-8), dtype=torch.float32)
        y_val = torch.tensor((targets[val_idx] - mean) / (std + 1e-8), dtype=torch.float32)
        norm_params = {'mean': mean, 'std': std}
    else:
        y_train = torch.tensor(targets[train_idx], dtype=torch.float32)
        y_val = torch.tensor(targets[val_idx], dtype=torch.float32)
        norm_params = {'mean': 0.0, 'std': 1.0}
    
    # Create datasets
    if cell_line_indices is not None:
        c_train = torch.tensor(cell_line_indices[train_idx], dtype=torch.long)
        c_val = torch.tensor(cell_line_indices[val_idx], dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train, c_train)
        val_dataset = TensorDataset(X_val, y_val, c_val)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, norm_params


if __name__ == "__main__":
    print("=" * 60)
    print("Single Metric Critic Test")
    print("=" * 60)
    
    # Create mock data
    n_samples = 200
    embedding_dim = 1024
    
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    targets = np.random.rand(n_samples).astype(np.float32) * 10
    
    # Create data loaders
    train_loader, val_loader, norm_params = create_single_metric_loaders(
        embeddings, targets, train_ratio=0.8, batch_size=32
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Normalization: mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")
    
    # Create and train model
    model = SingleMetricCritic(
        input_dim=embedding_dim,
        hidden_dims=[256, 128],
        metric_name='translation_efficiency'
    )
    
    trainer = SingleMetricTrainer(model, learning_rate=1e-3)
    model.set_normalization(norm_params['mean'], norm_params['std'])
    
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=10,
        early_stopping_patience=5,
        verbose=True
    )
    
    print("\n✓ SingleMetricCritic test passed!")
