"""
Evo model integration for RNA sequence embedding.
Uses the Evo-1-8k model to convert RNA sequences into dense vector representations.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional
import numpy as np


class EvoEmbedder:
    """
    Wrapper for Evo foundation model to generate RNA sequence embeddings.
    
    The Evo model is a genomic foundation model trained on DNA/RNA sequences.
    We use it to extract contextualized embeddings for translation efficiency prediction.
    """
    
    def __init__(
        self,
        model_name: str = "togethercomputer/evo-1-8k-base",
        device: Optional[str] = None,
        pooling: str = "mean"
    ):
        """
        Initialize the Evo embedder.
        
        Args:
            model_name: Hugging Face model identifier for Evo
            device: Device to run model on ('cuda' or 'cpu'). Auto-detects if None.
            pooling: Pooling strategy for embeddings ('mean', 'max', or 'cls')
        """
        self.model_name = model_name
        self.pooling = pooling
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading Evo model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        if model_name == "mock":
            print("Mock model requested. Using mock embedder.")
            self.model = None
            self.tokenizer = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"Model loaded successfully!")
            print(f"Embedding dimension: {self.model.config.hidden_size}")
            
        except Exception as e:
            print(f"Error loading Evo model: {e}")
            print("Note: Evo model may require acceptance of terms on Hugging Face")
            print("Fallback: Using a mock embedder for demonstration")
            self.model = None
            self.tokenizer = None
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.model is not None:
            return self.model.config.hidden_size
        else:
            return 1024  # Mock dimension
    
    def _convert_rna_to_dna(self, rna_sequence: str) -> str:
        """
        Convert RNA sequence (A, U, G, C) to DNA sequence (A, T, G, C).
        Evo model is trained on DNA, so we convert U -> T.
        
        Args:
            rna_sequence: RNA sequence
            
        Returns:
            DNA sequence
        """
        return rna_sequence.replace('U', 'T').upper()
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token-level embeddings into a single sequence embedding.
        
        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "mean":
            # Mean pooling (average over non-padding tokens)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling == "max":
            # Max pooling
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling == "cls":
            # Use first token (CLS-like)
            return hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def embed_sequence(
        self,
        rna_sequence: str,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embedding for a single RNA sequence.
        
        Args:
            rna_sequence: RNA sequence string
            return_numpy: If True, return numpy array; otherwise torch.Tensor
            
        Returns:
            Embedding vector
        """
        return self.embed_sequences([rna_sequence], return_numpy=return_numpy)[0]
    
    def embed_sequences(
        self,
        rna_sequences: List[str],
        batch_size: int = 8,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for multiple RNA sequences.
        
        Args:
            rna_sequences: List of RNA sequence strings
            batch_size: Batch size for processing
            return_numpy: If True, return numpy array; otherwise torch.Tensor
            
        Returns:
            Embedding matrix [num_sequences, embedding_dim]
        """
        if self.model is None:
            # Mock embedder for testing when model is not available
            print("Warning: Using mock embedder (model not loaded)")
            embeddings = np.random.randn(len(rna_sequences), 1024)
            if return_numpy:
                return embeddings
            else:
                return torch.tensor(embeddings, dtype=torch.float32)
        
        # Convert RNA to DNA
        dna_sequences = [self._convert_rna_to_dna(seq) for seq in rna_sequences]
        
        all_embeddings = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(dna_sequences), batch_size):
                batch_seqs = dna_sequences[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=8192  # Evo-1-8k max length
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings
                outputs = self.model(**encoded)
                hidden_states = outputs.last_hidden_state
                
                # Pool to get sequence-level embeddings
                pooled = self._pool_embeddings(
                    hidden_states,
                    encoded['attention_mask']
                )
                
                all_embeddings.append(pooled.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if return_numpy:
            return embeddings.numpy()
        else:
            return embeddings


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Evo Embedder Test")
    print("=" * 60)
    
    # Initialize embedder
    embedder = EvoEmbedder(pooling="mean")
    
    # Test sequences
    test_sequences = [
        "AUGGGCUACUUUGAUCGAUAA",  # Short test sequence
        "AUGCCCGGGAAAUUUCCCGGGAAAUAA"  # Another test sequence
    ]
    
    print(f"\nTest sequences: {len(test_sequences)}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    # Generate embeddings
    embeddings = embedder.embed_sequences(test_sequences)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"First embedding (first 10 dims): {embeddings[0, :10]}")
    
    # Test single sequence
    single_emb = embedder.embed_sequence(test_sequences[0])
    print(f"\nSingle sequence embedding shape: {single_emb.shape}")
    print(f"Match with batch result: {np.allclose(single_emb, embeddings[0])}")
