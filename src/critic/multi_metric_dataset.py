"""
Enhanced dataset utilities with multi-metric support and cell line/tissue conditioning.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import numpy as np
from pathlib import Path


class MultiMetricDataset(Dataset):
    """
    PyTorch Dataset for multiple RNA metrics (TE, half-life, etc.) with metadata.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        metrics_dict: Dict[str, np.ndarray],
        metadata: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ):
        """
        Initialize multi-metric dataset.
        
        Args:
            embeddings: RNA sequence embeddings [N, embedding_dim]
            metrics_dict: Dictionary mapping metric names to values [N]
            metadata: Optional DataFrame with cell_line, tissue, etc.
            normalize: Whether to normalize metric values
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.metadata = metadata
        
        # Store metrics
        self.metric_names = list(metrics_dict.keys())
        self.metrics = {}
        self.normalization_params = {}
        
        for metric_name, values in metrics_dict.items():
            values_tensor = torch.tensor(values, dtype=torch.float32)
            
            if normalize:
                mean = values_tensor.mean()
                std = values_tensor.std()
                values_tensor = (values_tensor - mean) / (std + 1e-8)
                self.normalization_params[metric_name] = {'mean': mean, 'std': std}
            else:
                self.normalization_params[metric_name] = {'mean': 0.0, 'std': 1.0}
            
            self.metrics[metric_name] = values_tensor
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        embedding = self.embeddings[idx]
        targets = {name: self.metrics[name][idx] for name in self.metric_names}
        return embedding, targets
    
    def denormalize(self, metric_name: str, normalized_values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original scale."""
        params = self.normalization_params[metric_name]
        return normalized_values * params['std'] + params['mean']


def load_cetnar_half_life_data(
    filepath: str,
    sequence_column: str = "sequence",
    half_life_column: str = "half_life",
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Load mRNA half-life data from Cetnar et al. 2024.
    
    Args:
        filepath: Path to the data file
        sequence_column: Name of column containing RNA sequences
        half_life_column: Name of column containing half-life values
        max_samples: Maximum number of samples to load
        
    Returns:
        DataFrame with sequences and half-life values
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please download data from Cetnar et al. 2024.\n"
            f"DOI: 10.1038/s41467-024-54059-7"
        )
    
    print(f"Loading half-life data from: {filepath}")
    
    # Read file based on extension
    if filepath.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Auto-detect columns if needed
    if sequence_column not in df.columns:
        seq_cols = [col for col in df.columns if 'seq' in col.lower()]
        if seq_cols:
            sequence_column = seq_cols[0]
            print(f"Using '{sequence_column}' as sequence column")
    
    if half_life_column not in df.columns:
        hl_cols = [col for col in df.columns if 'half' in col.lower() or 'stability' in col.lower()]
        if hl_cols:
            half_life_column = hl_cols[0]
            print(f"Using '{half_life_column}' as half-life column")
    
    # Clean data
    df_clean = df[[sequence_column, half_life_column]].dropna()
    print(f"After removing NaN: {len(df_clean)} samples")
    
    if max_samples is not None and max_samples < len(df_clean):
        df_clean = df_clean.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} for testing")
    
    return df_clean.rename(columns={sequence_column: 'sequence', half_life_column: 'half_life'})


def merge_multi_metric_datasets(
    te_df: pd.DataFrame,
    hl_df: Optional[pd.DataFrame] = None,
    on: str = 'sequence'
) -> pd.DataFrame:
    """
    Merge datasets with different metrics.
    
    Args:
        te_df: DataFrame with translation efficiency
        hl_df: Optional DataFrame with half-life
        on: Column to merge on (default: 'sequence')
        
    Returns:
        Merged DataFrame with all available metrics
    """
    if hl_df is None:
        return te_df
    
    # Inner join to keep only sequences with both metrics
    merged = pd.merge(te_df, hl_df, on=on, how='inner', suffixes=('_te', '_hl'))
    
    print(f"Merged dataset: {len(merged)} samples with multiple metrics")
    
    return merged


def create_multi_metric_loaders(
    embeddings: np.ndarray,
    metrics_dict: Dict[str, np.ndarray],
    metadata: Optional[pd.DataFrame] = None,
    train_split: float = 0.8,
    batch_size: int = 32,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders for multi-metric training.
    
    Args:
        embeddings: RNA sequence embeddings [N, embedding_dim]
        metrics_dict: Dictionary of metric names to values
        metadata: Optional metadata DataFrame
        train_split: Fraction of data for training
        batch_size: Batch size
        random_seed: Random seed
        
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
    
    # Split metadata if provided
    train_metadata = metadata.iloc[train_indices] if metadata is not None else None
    val_metadata = metadata.iloc[val_indices] if metadata is not None else None
    
    # Create datasets
    train_metrics = {k: v[train_indices] for k, v in metrics_dict.items()}
    train_dataset = MultiMetricDataset(
        embeddings[train_indices],
        train_metrics,
        metadata=train_metadata,
        normalize=True
    )
    
    # Use training normalization for validation
    val_metrics = {}
    for metric_name, values in metrics_dict.items():
        val_values = values[val_indices]
        params = train_dataset.normalization_params[metric_name]
        val_values_norm = (val_values - params['mean'].item()) / (params['std'].item() + 1e-8)
        val_metrics[metric_name] = val_values_norm
    
    val_dataset = MultiMetricDataset(
        embeddings[val_indices],
        val_metrics,
        metadata=val_metadata,
        normalize=False  # Already normalized
    )
    val_dataset.normalization_params = train_dataset.normalization_params
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Metrics: {list(metrics_dict.keys())}")
    for metric in metrics_dict.keys():
        params = train_dataset.normalization_params[metric]
        print(f"  {metric}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test multi-metric dataset
    print("=" * 60)
    print("Multi-Metric Dataset Test")
    print("=" * 60)
    
    # Create mock data
    n_samples = 100
    embedding_dim = 1024
    
    mock_embeddings = np.random.randn(n_samples, embedding_dim)
    mock_metrics = {
        'translation_efficiency': np.random.rand(n_samples) * 10,
        'half_life': np.random.rand(n_samples) * 8  # hours
    }
    
    # Create metadata
    mock_metadata = pd.DataFrame({
        'cell_line': np.random.choice(['HEK293', 'HeLa', 'K562'], n_samples),
        'tissue': np.random.choice(['kidney', 'cervix', 'blood'], n_samples)
    })
    
    # Create data loaders
    train_loader, val_loader = create_multi_metric_loaders(
        mock_embeddings,
        mock_metrics,
        metadata=mock_metadata,
        train_split=0.8,
        batch_size=16
    )
    
    # Test data loader
    for embeddings, targets in train_loader:
        print(f"\nBatch embeddings shape: {embeddings.shape}")
        print(f"Batch targets:")
        for metric, values in targets.items():
            print(f"  {metric}: shape={values.shape}, range=[{values.min():.2f}, {values.max():.2f}]")
        break
