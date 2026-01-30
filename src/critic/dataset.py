"""
Dataset utilities for loading translation efficiency data from Zheng et al. 2025.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class TranslationEfficiencyDataset(Dataset):
    """
    PyTorch Dataset for RNA sequences and translation efficiency values.
    
    Expects data with columns:
        - RNA sequence (full sequence or UTRs + CDS)
        - Translation efficiency (TE) value
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        te_values: np.ndarray,
        normalize_te: bool = True
    ):
        """
        Initialize dataset with precomputed embeddings.
        
        Args:
            embeddings: RNA sequence embeddings [N, embedding_dim]
            te_values: Translation efficiency values [N]
            normalize_te: Whether to normalize TE values to [0, 1]
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.te_values = torch.tensor(te_values, dtype=torch.float32)
        
        if normalize_te:
            # Normalize to mean=0, std=1 for better training
            self.te_mean = self.te_values.mean()
            self.te_std = self.te_values.std()
            self.te_values = (self.te_values - self.te_mean) / (self.te_std + 1e-8)
        else:
            self.te_mean = 0.0
            self.te_std = 1.0
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.te_values[idx]
    
    def denormalize_te(self, normalized_te: torch.Tensor) -> torch.Tensor:
        """Convert normalized TE values back to original scale."""
        return normalized_te * self.te_std + self.te_mean


def load_zheng_data(
    filepath: str,
    sequence_column: str = "sequence",
    te_column: str = "TE",
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Load translation efficiency data from Supplementary Table 1 (Zheng et al. 2025).
    
    Args:
        filepath: Path to the Excel file
        sequence_column: Name of column containing RNA sequences
        te_column: Name of column containing translation efficiency values
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        DataFrame with sequences and TE values
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please download Supplementary Table 1 from Zheng et al. 2025.\n"
            f"See data/README.md for instructions."
        )
    
    print(f"Loading data from: {filepath}")
    
    # Read Excel file
    if filepath.suffix == '.xlsx' or filepath.suffix == '.xls':
        df = pd.read_excel(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    if sequence_column not in df.columns:
        print(f"Warning: '{sequence_column}' column not found.")
        print("Available columns with 'seq' in name:")
        seq_cols = [col for col in df.columns if 'seq' in col.lower()]
        for col in seq_cols:
            print(f"  - {col}")
        if seq_cols:
            sequence_column = seq_cols[0]
            print(f"Using '{sequence_column}' as sequence column")
    
    if te_column not in df.columns:
        print(f"Warning: '{te_column}' column not found.")
        print("Available columns with 'te' or 'efficiency' in name:")
        te_cols = [col for col in df.columns if 'te' in col.lower() or 'efficiency' in col.lower()]
        for col in te_cols:
            print(f"  - {col}")
        if te_cols:
            te_column = te_cols[0]
            print(f"Using '{te_column}' as TE column")
    
    # Filter out rows with missing data
    df_clean = df[[sequence_column, te_column]].dropna()
    print(f"After removing NaN: {len(df_clean)} samples")
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(df_clean):
        df_clean = df_clean.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} for testing")
    
    return df_clean.rename(columns={sequence_column: 'sequence', te_column: 'TE'})


def create_data_loaders(
    embeddings: np.ndarray,
    te_values: np.ndarray,
    train_split: float = 0.8,
    batch_size: int = 32,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        embeddings: RNA sequence embeddings [N, embedding_dim]
        te_values: Translation efficiency values [N]
        train_split: Fraction of data for training
        batch_size: Batch size for data loaders
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Shuffle and split
    n_samples = len(embeddings)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create datasets
    train_dataset = TranslationEfficiencyDataset(
        embeddings[train_indices],
        te_values[train_indices],
        normalize_te=True
    )
    
    # Use same normalization stats for validation
    val_embeddings = embeddings[val_indices]
    val_te = te_values[val_indices]
    val_te_normalized = (val_te - train_dataset.te_mean.item()) / (train_dataset.te_std.item() + 1e-8)
    
    val_dataset = TranslationEfficiencyDataset(
        val_embeddings,
        val_te_normalized,
        normalize_te=False  # Already normalized
    )
    val_dataset.te_mean = train_dataset.te_mean
    val_dataset.te_std = train_dataset.te_std
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"TE normalization: mean={train_dataset.te_mean:.4f}, std={train_dataset.te_std:.4f}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("=" * 60)
    print("Dataset Utilities Test")
    print("=" * 60)
    
    # Create mock data
    print("\nCreating mock data for testing...")
    n_samples = 100
    embedding_dim = 1024
    
    mock_embeddings = np.random.randn(n_samples, embedding_dim)
    mock_te = np.random.rand(n_samples) * 10  # TE values between 0-10
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        mock_embeddings,
        mock_te,
        train_split=0.8,
        batch_size=16
    )
    
    # Test data loader
    for embeddings, te_values in train_loader:
        print(f"\nBatch embeddings shape: {embeddings.shape}")
        print(f"Batch TE values shape: {te_values.shape}")
        print(f"TE values (normalized): {te_values[:5]}")
        break
