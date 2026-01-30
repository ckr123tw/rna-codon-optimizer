"""
Dataset utilities for loading translation efficiency data from Zheng et al. 2025.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
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
        cell_line_indices: Optional[np.ndarray] = None,
        normalize_te: bool = True
    ):
        """
        Initialize dataset with precomputed embeddings.
        
        Args:
            embeddings: RNA sequence embeddings [N, embedding_dim]
            te_values: Translation efficiency values [N]
            cell_line_indices: Optional cell line integer indices [N]
            normalize_te: Whether to normalize TE values to [0, 1]
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.te_values = torch.tensor(te_values, dtype=torch.float32)
        
        if cell_line_indices is not None:
            self.cell_line_indices = torch.tensor(cell_line_indices, dtype=torch.long)
        else:
            self.cell_line_indices = None
        
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
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
            inputs: {'embedding': ..., 'cell_line_idx': ...}
            targets: {'translation_efficiency': ...} (formatted for MultiMetricTrainer)
        """
        inputs = {'embedding': self.embeddings[idx]}
        if self.cell_line_indices is not None:
            inputs['cell_line_idx'] = self.cell_line_indices[idx]
            
        targets = {'translation_efficiency': self.te_values[idx]}
        return inputs, targets
    
    def denormalize_te(self, normalized_te: torch.Tensor) -> torch.Tensor:
        """Convert normalized TE values back to original scale."""
        return normalized_te * self.te_std + self.te_mean


def load_zheng_data(
    filepath: str,
    sequence_column: str = "sequence",
    te_column: str = "TE",
    cell_line_column: Optional[str] = "cell_line",
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load translation efficiency data.
    
    Returns:
        DataFrame with sequences, TE, and cell_line_idx
        Dictionary mapping cell line names to indices
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        # Fallback for mock/demo if file missing
        print(f"Warning: File {filepath} not found. Using mock data.")
        df = pd.DataFrame({
            'sequence': ['AUG' * 10] * 100,
            'TE': np.random.rand(100),
            'cell_line': np.random.choice(['HEK293', 'HeLa'], 100)
        })
    else:
        print(f"Loading data from: {filepath}")
        if filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Column mapping logic (omitted specific column search for brevity, assumes standard names or args)
    # Ensure required columns
    if sequence_column not in df.columns:
        # Simple fallback
        cols = [c for c in df.columns if 'seq' in c.lower()]
        if cols: sequence_column = cols[0]
        
    if te_column not in df.columns:
        cols = [c for c in df.columns if 'te' in c.lower()]
        if cols: te_column = cols[0]
        
    # Cell Line Handling
    cell_line_map = {}
    if cell_line_column and (cell_line_column in df.columns):
        print(f"Found cell line column: {cell_line_column}")
        unique_cells = sorted(df[cell_line_column].dropna().unique())
        cell_line_map = {name: i for i, name in enumerate(unique_cells)}
        print(f"Cell lines found: {cell_line_map}")
        
        df = df.dropna(subset=[sequence_column, te_column, cell_line_column])
        df['cell_line_idx'] = df[cell_line_column].map(cell_line_map)
    else:
        print("No cell line column found or specified.")
        df = df.dropna(subset=[sequence_column, te_column])
    
    if max_samples:
        df = df.head(max_samples)
        
    return df.rename(columns={sequence_column: 'sequence', te_column: 'TE'}), cell_line_map


def create_data_loaders(
    embeddings: np.ndarray,
    te_values: np.ndarray,
    cell_line_indices: Optional[np.ndarray] = None,
    train_split: float = 0.8,
    batch_size: int = 32,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Shuffle and split
    n_samples = len(embeddings)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # helper to slice if exists
    def slice_cells(arr, idxs):
        return arr[idxs] if arr is not None else None
        
    train_cells = slice_cells(cell_line_indices, train_indices)
    val_cells = slice_cells(cell_line_indices, val_indices)
    
    # Create datasets
    train_dataset = TranslationEfficiencyDataset(
        embeddings[train_indices],
        te_values[train_indices],
        cell_line_indices=train_cells,
        normalize_te=True
    )
    
    # Use same normalization stats for validation
    val_te = te_values[val_indices]
    val_te_normalized = (val_te - train_dataset.te_mean.item()) / (train_dataset.te_std.item() + 1e-8)
    
    val_dataset = TranslationEfficiencyDataset(
        embeddings[val_indices],
        val_te_normalized,
        cell_line_indices=val_cells,
        normalize_te=False 
    )
    val_dataset.te_mean = train_dataset.te_mean
    val_dataset.te_std = train_dataset.te_std
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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
