"""
MLP critic model for predicting translation efficiency from RNA embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional


class TranslationEfficiencyCritic(nn.Module):
    """
    Multi-layer perceptron that predicts translation efficiency from RNA embeddings.
    
    Architecture:
        Input: RNA sequence embedding from Evo model
        Hidden layers: 2-3 fully connected layers with ReLU + Dropout
        Output: Single value for translation efficiency prediction
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        activation: str = "relu"
    ):
        """
        Initialize the critic model.
        
        Args:
            input_dim: Dimension of input embeddings (should match Evo output)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            activation: Activation function ('relu', 'gelu', or 'tanh')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single value for TE prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic.
        
        Args:
            embeddings: RNA sequence embeddings [batch_size, input_dim]
            
        Returns:
            Translation efficiency predictions [batch_size, 1]
        """
        return self.model(embeddings)
    
    def predict(
        self,
        embeddings: torch.Tensor,
        return_numpy: bool = False
    ):
        """
        Make predictions on embeddings.
        
        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            return_numpy: If True, return numpy array
            
        Returns:
            Predictions [batch_size]
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(embeddings).squeeze(-1)
        
        if return_numpy:
            return preds.cpu().numpy()
        return preds
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CriticTrainer:
    """
    Trainer for the translation efficiency critic model.
    """
    
    def __init__(
        self,
        model: TranslationEfficiencyCritic,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Critic model to train
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        train_loader,
        verbose: bool = True
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with (embeddings, targets) tuples
            verbose: Whether to print progress
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            # Move to device
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(embeddings).squeeze(-1)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader) -> tuple:
        """
        Validate the model.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Tuple of (loss, R^2 score)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for embeddings, targets in val_loader:
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(embeddings).squeeze(-1)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Calculate R^2
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return avg_loss, r2.item()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test the critic model
    print("=" * 60)
    print("Translation Efficiency Critic Test")
    print("=" * 60)
    
    # Create model
    critic = TranslationEfficiencyCritic(
        input_dim=1024,
        hidden_dims=[512, 256],
        dropout=0.3
    )
    
    print(f"Model architecture:")
    print(critic)
    print(f"\nTotal parameters: {critic.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 16
    test_embeddings = torch.randn(batch_size, 1024)
    
    predictions = critic(test_embeddings)
    print(f"\nInput shape: {test_embeddings.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].squeeze().tolist()}")
