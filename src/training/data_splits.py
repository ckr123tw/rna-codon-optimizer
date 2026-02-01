"""
Data splitting utilities for train/dev/test sets.
Provides helpers for creating reproducible data splits for training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

# Import the existing dataset class
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.critic.dataset import TranslationEfficiencyDataset


@dataclass
class DataSplitConfig:
    """
    Configuration for data splitting.
    
    Hyperparameters:
        train_ratio: Fraction of data for training (default: 0.70)
            - Higher values (0.8-0.9) when data is limited
            - Lower values (0.6-0.7) when you want robust validation
            
        dev_ratio: Fraction of data for development/validation (default: 0.15)
            - Used during training for early stopping and hyperparameter tuning
            - Should be large enough to get stable validation metrics
            
        test_ratio: Fraction of data for final testing (default: 0.15)
            - Held out completely during training
            - Used only for final evaluation
            - Set to 0.0 if you want traditional train/val split only
            
        batch_size: Number of samples per batch (default: 32)
            - Larger batches (64-128) for faster training with more memory
            - Smaller batches (8-16) for more stochastic updates
            
        random_seed: Seed for reproducibility (default: 42)
            - Change to get different splits for cross-validation
            
        shuffle_train: Whether to shuffle training data (default: True)
            - Usually True for training, False for reproducible debugging
    
    Example:
        >>> config = DataSplitConfig(
        ...     train_ratio=0.7,
        ...     dev_ratio=0.15, 
        ...     test_ratio=0.15,
        ...     batch_size=32
        ... )
    """
    train_ratio: float = 0.70
    dev_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    random_seed: int = 42
    shuffle_train: bool = True
    num_workers: int = 0
    
    def __post_init__(self):
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, dev={self.dev_ratio}, test={self.test_ratio})"
            )
        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be positive")


def create_train_dev_test_splits(
    embeddings: np.ndarray,
    target_values: Dict[str, np.ndarray],
    cell_line_indices: Optional[np.ndarray] = None,
    config: Optional[DataSplitConfig] = None
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """
    Split data into train, dev, and test sets.
    
    Args:
        embeddings: RNA sequence embeddings [N, embedding_dim]
        target_values: Dict mapping metric names to values [N]
        cell_line_indices: Optional cell line integer indices [N]
        config: Data split configuration (uses defaults if None)
        
    Returns:
        Three tuples of (embeddings, targets_dict, cell_indices) for train, dev, test
        
    Example:
        >>> train_data, dev_data, test_data = create_train_dev_test_splits(
        ...     embeddings, {'translation_efficiency': te_values}
        ... )
        >>> train_emb, train_targets, train_cells = train_data
    """
    if config is None:
        config = DataSplitConfig()
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    
    n_samples = len(embeddings)
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    train_end = int(n_samples * config.train_ratio)
    dev_end = train_end + int(n_samples * config.dev_ratio)
    
    train_idx = indices[:train_end]
    dev_idx = indices[train_end:dev_end]
    test_idx = indices[dev_end:]
    
    def slice_data(idxs):
        emb = embeddings[idxs]
        targets = {k: v[idxs] for k, v in target_values.items()}
        cells = cell_line_indices[idxs] if cell_line_indices is not None else None
        return (emb, targets, cells)
    
    train_data = slice_data(train_idx)
    dev_data = slice_data(dev_idx)
    test_data = slice_data(test_idx)
    
    print(f"Data split (seed={config.random_seed}):")
    print(f"  Train: {len(train_idx)} samples ({config.train_ratio*100:.0f}%)")
    print(f"  Dev:   {len(dev_idx)} samples ({config.dev_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_idx)} samples ({config.test_ratio*100:.0f}%)")
    
    return train_data, dev_data, test_data


def create_data_loaders_with_test(
    embeddings: np.ndarray,
    target_values: Dict[str, np.ndarray],
    cell_line_indices: Optional[np.ndarray] = None,
    config: Optional[DataSplitConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, dev, and test DataLoaders with proper normalization.
    
    The training set statistics are used to normalize all sets for consistency.
    
    Args:
        embeddings: RNA sequence embeddings [N, embedding_dim]
        target_values: Dict mapping metric names to values [N]
        cell_line_indices: Optional cell line integer indices [N]
        config: Data split configuration
        
    Returns:
        Tuple of (train_loader, dev_loader, test_loader)
        
    Example:
        >>> train_loader, dev_loader, test_loader = create_data_loaders_with_test(
        ...     embeddings, 
        ...     {'translation_efficiency': te_values},
        ...     config=DataSplitConfig(batch_size=64)
        ... )
    """
    if config is None:
        config = DataSplitConfig()
    
    # Set seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    # Get splits
    train_data, dev_data, test_data = create_train_dev_test_splits(
        embeddings, target_values, cell_line_indices, config
    )
    
    train_emb, train_targets, train_cells = train_data
    dev_emb, dev_targets, dev_cells = dev_data
    test_emb, test_targets, test_cells = test_data
    
    # Create training dataset (this computes normalization stats)
    train_dataset = TranslationEfficiencyDataset(
        train_emb,
        train_targets,
        cell_line_indices=train_cells,
        normalize_targets=True
    )
    
    # Get normalization stats from training set
    target_stats = train_dataset.target_stats
    
    # Normalize dev and test using training stats
    def normalize_with_stats(targets_dict, stats):
        normalized = {}
        for k, v in targets_dict.items():
            if k in stats:
                mean = stats[k]['mean']
                std = stats[k]['std']
                # Handle tensor vs float
                mean_val = mean.item() if hasattr(mean, 'item') else mean
                std_val = std.item() if hasattr(std, 'item') else std
                normalized[k] = (v - mean_val) / (std_val + 1e-8)
            else:
                normalized[k] = v
        return normalized
    
    dev_targets_norm = normalize_with_stats(dev_targets, target_stats)
    test_targets_norm = normalize_with_stats(test_targets, target_stats)
    
    # Create dev and test datasets (don't re-normalize)
    dev_dataset = TranslationEfficiencyDataset(
        dev_emb,
        dev_targets_norm,
        cell_line_indices=dev_cells,
        normalize_targets=False
    )
    dev_dataset.target_stats = target_stats
    
    test_dataset = TranslationEfficiencyDataset(
        test_emb,
        test_targets_norm,
        cell_line_indices=test_cells,
        normalize_targets=False
    )
    test_dataset.target_stats = target_stats
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # Test the data splitting utilities
    print("=" * 60)
    print("Data Splitting Utilities Test")
    print("=" * 60)
    
    # Create mock data
    n_samples = 1000
    embedding_dim = 1024
    
    mock_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    mock_targets = {
        'translation_efficiency': np.random.rand(n_samples).astype(np.float32) * 10,
        'half_life': np.random.rand(n_samples).astype(np.float32) * 24
    }
    mock_cells = np.random.randint(0, 3, n_samples)
    
    # Test with default config
    print("\n--- Default Configuration ---")
    config = DataSplitConfig()
    print(f"Config: train={config.train_ratio}, dev={config.dev_ratio}, test={config.test_ratio}")
    
    train_loader, dev_loader, test_loader = create_data_loaders_with_test(
        mock_embeddings,
        mock_targets,
        cell_line_indices=mock_cells,
        config=config
    )
    
    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"  Dev:   {len(dev_loader)} batches ({len(dev_loader.dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_loader.dataset)} samples)")
    
    # Test a batch
    for batch in train_loader:
        inputs, targets = batch
        print(f"\nSample batch:")
        print(f"  Embedding shape: {inputs['embedding'].shape}")
        print(f"  Cell line indices: {inputs.get('cell_line_idx', 'N/A')}")
        print(f"  Target metrics: {list(targets.keys())}")
        break
    
    print("\n✓ Data splitting test passed!")
