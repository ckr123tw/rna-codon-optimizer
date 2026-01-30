
import unittest
import sys
import os
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.critic.multi_metric_critic import MultiMetricCritic, MultiMetricTrainer
from src.critic.dataset import TranslationEfficiencyDataset, load_zheng_data
from src.lora_generation.lora_adapter import format_conditional_prompt

class TestCellLineSpecificity(unittest.TestCase):
    
    def test_dataset_cell_line_indices(self):
        """Test that dataset handles cell line indices correctly."""
        embeddings = np.random.randn(10, 32)
        te_values = np.random.rand(10)
        cell_indices = np.array([0, 1] * 5)
        
        dataset = TranslationEfficiencyDataset(
            embeddings=embeddings, 
            te_values=te_values, 
            cell_line_indices=cell_indices
        )
        
        inputs, targets = dataset[0]
        self.assertIn('cell_line_idx', inputs)
        self.assertEqual(inputs['cell_line_idx'].item(), 0)
        self.assertIn('translation_efficiency', targets)

    def test_load_zheng_data_mock(self):
        """Test loading mock data with cell lines."""
        # Using non-existent file to trigger mock generation
        df, cell_map = load_zheng_data("non_existent_file.xlsx")
        
        self.assertIn('cell_line', df.columns)
        self.assertIn('cell_line_idx', df.columns)
        self.assertTrue(len(cell_map) > 0)
        print(f"Mock data cell map: {cell_map}")

    def test_critic_with_cell_embeddings(self):
        """Test MultiMetricCritic forward pass with cell embeddings."""
        batch_size = 4
        input_dim = 16
        num_cells = 3
        cell_dim = 8
        
        model = MultiMetricCritic(
            input_dim=input_dim,
            metrics=['translation_efficiency'],
            num_cell_lines=num_cells,
            cell_embedding_dim=cell_dim
        )
        
        embeddings = torch.randn(batch_size, input_dim)
        cell_indices = torch.tensor([0, 1, 2, 0])
        
        # Test forward
        preds = model(embeddings, cell_line_indices=cell_indices)
        self.assertIn('translation_efficiency', preds)
        self.assertEqual(preds['translation_efficiency'].shape, (batch_size, 1))
        
        # Test without cell indices 
        preds_no_cell = model(embeddings)
        self.assertIn('translation_efficiency', preds_no_cell)

    def test_prompt_formatting(self):
        """Test prompt formatting includes cell line."""
        prompt = format_conditional_prompt(
            utr5="AAA",
            utr3="TTT",
            amino_acid_sequence="M", 
            target_efficiency=1.0, 
            cell_line="HEK293"
        )
        
        self.assertIn("in HEK293 cells", prompt)
        self.assertIn("Cell Line: HEK293", prompt)
        
        # Test backward compatibility
        prompt_old = format_conditional_prompt(
             utr5="AAA", 
             utr3="TTT", 
             amino_acid_sequence="M", 
             target_efficiency=1.0
        )
        self.assertNotIn("Cell Line:", prompt_old)

if __name__ == '__main__':
    unittest.main()
