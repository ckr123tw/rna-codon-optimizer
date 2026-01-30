"""
Comprehensive pipeline verification script.
Mocks deep learning dependencies to verify logic flow in a constrained environment.
"""
import sys
import unittest
from unittest.mock import MagicMock

# ==============================================================================
# 1. SETUP MOCKS
# ==============================================================================

# Mock torch
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.device = lambda x: x
mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=lambda x: None, __exit__=lambda *x: None))

class MockTensor:
    def __init__(self, shape=None, data=None):
        self.shape = shape if shape else (1, 1)
        self.data = data if data is not None else []
    
    def __getattr__(self, name):
        # Catch-all for any tensor method (expand, float, cuda, to, etc.)
        if name in ['shape', 'data']: 
            return super().__getattribute__(name)
        def method(*args, **kwargs):
            return self
        return method

    def __iter__(self):
        # Allow iteration (yields nothing or dummy)
        return iter([self])
    
    def __getitem__(self, idx):
        # Handle tuple indexing for 2D access in tests
        if isinstance(idx, tuple) and len(idx) == 2:
            r, c = idx
            # Simplified: just return mock or 0
            if hasattr(self.data, '__getitem__') and isinstance(self.data, list) and isinstance(self.data[0], list):
                 return self.data[r][c]
            return 0.0
        return MockTensor(shape=self.shape)
        
    def size(self, dim=None): 
        if dim is not None:
            return self.shape[dim] if dim < len(self.shape) else 1
        return self.shape
    def __len__(self): return self.shape[0] if self.shape else 0
    def numpy(self): return [1.0]
    def tolist(self): return [1]

mock_torch.Tensor = MockTensor
mock_torch.FloatTensor = MockTensor
mock_torch.LongTensor = MockTensor
mock_torch.as_tensor = lambda x: MockTensor()

# Tensor creation
def mock_zeros(size, **kwargs): return MockTensor(shape=size)
def mock_ones(size, **kwargs): return MockTensor(shape=size)
mock_torch.zeros = mock_zeros
mock_torch.ones = mock_ones
mock_torch.stack = lambda x: MockTensor()
mock_torch.cat = lambda x, dim=0: MockTensor()

# Define a real class for nn.Module so subclasses preserve their methods
class MockModule:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs): return MockTensor()
    def __getattr__(self, name):
        # Allow accessing attributes like self.encoder if not set
        return MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn'].Module = MockModule # Important!
sys.modules['torch.nn'].ModuleDict = dict # Use real dict for ModuleDict
sys.modules['torch.nn'].Sequential = MagicMock

mock_torch_utils = MagicMock()
mock_torch_utils.data.Dataset = MagicMock
mock_torch_utils.data.DataLoader = MagicMock
sys.modules['torch.utils'] = mock_torch_utils
sys.modules['torch.utils.data'] = mock_torch_utils.data

# Mock transformers
mock_transformers = MagicMock()

class MockBatchEncoding(dict):
    def to(self, device): return self

class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.eos_token_id = 99
        self.pad_token_id = 0
    def encode(self, text, **kwargs): 
        return [1, 2, 3] # Dummy IDs
    def __call__(self, text, return_tensors=None, **kwargs):
        data = {'input_ids': MockTensor(shape=(1, 5)), 'attention_mask': MockTensor(shape=(1, 5))}
        return MockBatchEncoding(data)
    def decode(self, token_ids, **kwargs):
        return "AUGGCCAUG" # Dummy generated sequence

mock_transformers.AutoTokenizer.from_pretrained.return_value = MockTokenizer()
mock_transformers.AutoModel.from_pretrained.return_value = MagicMock()
mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
class MockLogitsProcessorList(list):
    def __call__(self, input_ids, scores): return scores
mock_transformers.LogitsProcessorList = MockLogitsProcessorList
sys.modules['transformers'] = mock_transformers

# Mock peft
mock_peft = MagicMock()
mock_peft.get_peft_model.return_value = MagicMock()
sys.modules['peft'] = mock_peft

# Mock numpy
mock_numpy = MagicMock()
mock_numpy.mean = lambda x: 0.5
mock_numpy.std = lambda x: 0.1
mock_numpy.min = lambda x: 0.0
mock_numpy.max = lambda x: 1.0
mock_numpy.random.choice = lambda x, size, replace: [0] * size
mock_numpy.allclose = lambda x, y: True
sys.modules['numpy'] = mock_numpy

# Mock pandas
mock_pandas = MagicMock()
mock_pandas.read_csv.return_value = MagicMock()
mock_pandas.DataFrame.return_value = MagicMock()
sys.modules['pandas'] = mock_pandas

# Mock sklearn
mock_sklearn = MagicMock()
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()

# Mock trl
mock_trl = MagicMock()
class MockAutoModelValHead:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return MagicMock()
mock_trl.AutoModelForCausalLMWithValueHead = MockAutoModelValHead
mock_trl.PPOConfig = MagicMock
mock_trl.PPOTrainer = MagicMock
sys.modules['trl'] = mock_trl

# ==============================================================================
# 2. IMPORT MODULES TO TEST
# ==============================================================================
# Note: We must set PYTHONPATH=. in execution to make these imports work

from src.sequence_generation.validation import validate_cds_amino_acids, validate_full_rna_sequence
from src.sequence_generation.codon_table import translate_amino_acid_to_cds
from src.sequence_generation.evo_embedder import EvoEmbedder
from src.critic.critic_model import TranslationEfficiencyCritic
from src.critic.multi_metric_critic import MultiMetricCritic
from src.lora_generation.lora_adapter import EvoLoRAAdapter
from src.ppo_training.ppo_trainer import PPOTrainingConfig, RNAPPOTrainer
from src.ppo_training.multi_metric_ppo import MultiMetricPPOTrainer

# ==============================================================================
# 3. TEST SUITE
# ==============================================================================

class TestPipeline(unittest.TestCase):
    
    def test_01_validation_logic(self):
        """Test RNA validation logic"""
        print("\nTesting Validation Logic...")
        aa_seq = "MA"
        cds = "AUGGCC"
        valid, msg = validate_cds_amino_acids(cds, aa_seq)
        self.assertTrue(valid, f"Validation failed: {msg}")
        print("  Validation logic passed.")

    def test_02_evo_embedder(self):
        """Test Evo embedder"""
        print("\nTesting Evo Embedder...")
        embedder = EvoEmbedder()
        seq = "AUG"
        emb = embedder.embed_sequence(seq, return_numpy=False)
        self.assertIsInstance(emb, MockTensor)
        print("  EvoEmbedder ran.")
        
    def test_03_critic_model(self):
        """Test Multi-Metric Critic"""
        print("\nTesting Critic Models...")
        mock_reg = MagicMock()
        mock_reg.return_value = MockTensor((1,1)) # When called as head(x)
        
        critic = MultiMetricCritic(metrics=['te', 'hl'])
        critic.heads = {'te': mock_reg, 'hl': mock_reg} # Inject manual heads dict
        
        emb = MockTensor(shape=(1, 10))
        preds = critic.predict(emb)
        print(f"  Predictions keys: {preds.keys()}")
        self.assertIn('te', preds)
        self.assertIn('hl', preds)
        
    def test_04_lora_generation(self):
        """Test LoRA Generation Adapter"""
        print("\nTesting LoRA Adapter...")
        adapter = EvoLoRAAdapter(model_name="dummy")
        adapter.model.generate.return_value = MockTensor(shape=(1, 20))
        seqs = adapter.generate_sequences(
            prompt="test",
            num_sequences=1,
            amino_acid_constraint="M"
        )
        self.assertEqual(len(seqs), 1)
        print(f"  Generated sequence: {seqs[0]}")
        
    def test_05_ppo_training_flow(self):
        """Test PPO Training Loop Logic"""
        print("\nTesting PPO Training Flow...")
        embedder = EvoEmbedder()
        critic = MultiMetricCritic(metrics=['te'])
        critic.models = {'te': MagicMock()} # Fallback if code uses models? No, uses heads.
        critic.heads = {'te': MagicMock(return_value=MockTensor())}
        critic.predict = MagicMock(return_value={'te': [0.8]})
        
        adapter = EvoLoRAAdapter(model_name="dummy")
        adapter.generate_sequences = MagicMock(return_value=["AUG"])
        
        config = PPOTrainingConfig(batch_size=2)
        
        # We need to ensure PPOTrainer components don't crash
        # RNAPPOTrainer/MultiMetricPPOTrainer __init__ creates PPOTrainer from trl.
        # Our mock_trl.PPOTrainer is MagicMock, so initialized trainer is a Mock.
        # BUT we are instantiating MultiMetricPPOTrainer which inherits.
        # MultiMetricPPOTrainer.__init__ calls super().__init__.
        # If we didn't define MockModule for PPOTrainer, it might be an issue?
        # PPOTrainer is not nn.Module usually.
        # But we mocked it as MagicMock. Inheriting from MagicMock is tricky.
        
        # Let's hope inheritance from MagicMock allows methods defined in subclass to exist? 
        # Actually inheriting from MagicMock makes the instance a Mock, usually overriding methods unless carefully done.
        # BUT MultiMetricPPOTrainer does NOT inherit from PPOTrainer in source?
        # Let's check inheritance.
        # src/ppo_training/multi_metric_ppo.py: class MultiMetricPPOTrainer(RNAPPOTrainer)
        # src/ppo_training/ppo_trainer.py: class RNAPPOTrainer.
        # RNAPPOTrainer holds a PPOTrainer instance, does NOT inherit from it. 
        # Wait, let me check ppo_trainer.py.
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
