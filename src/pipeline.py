"""
End-to-end RNA codon optimization pipeline.
Integrates all components: sequence generation, embedding, critic, LoRA, and PPO.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json

# Import pipeline components
from src.sequence_generation import RNASequenceGenerator
from src.sequence_generation.evo_embedder import EvoEmbedder
from src.critic import MultiMetricCritic, MultiMetricTrainer, load_zheng_data, create_data_loaders
from src.lora_generation.lora_adapter import EvoLoRAAdapter, format_conditional_prompt
from src.ppo_training.ppo_trainer import RNAPPOTrainer, PPOTrainingConfig


class RNACodonOptimizationPipeline:
    """
    Complete pipeline for RNA codon optimization using RL and foundation models.
    
    Workflow:
        1. Load and prepare translation efficiency dataset
        2. Train critic model on embeddings -> TE
        3. Initialize LoRA adapter for Evo model
        4. Use PPO to optimize LoRA for desired TE targets
        5. Generate optimized sequences
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        model_name: str = "togethercomputer/evo-1-8k-base",
        device: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to translation efficiency dataset (Zheng et al. 2025)
            model_name: Evo model identifier
            device: Device to use ('cuda' or 'cpu')
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing RNA Codon Optimization Pipeline")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_name}")
        
        self.model_name = model_name
        
        # Initialize components
        self.embedder = EvoEmbedder(model_name=model_name, device=str(self.device))
        self.critic = None
        self.lora_model = None
        self.ppo_trainer = None
        
        self.data_path = data_path
        self.trained = False
        
        # Cell line support
        self.cell_line_map = {}
        self.num_cell_lines = 0
        
        # Multi-metric support
        self.target_metrics = ['translation_efficiency']
        self.targets_dict = {}
    
    def step1_prepare_data(
        self,
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        Step 1: Load and prepare translation efficiency dataset.
        
        Args:
            max_samples: Limit number of samples (for testing)
            
        Returns:
            Statistics about loaded data
        """
        print("\n" + "=" * 60)
        print("STEP 1: Loading Translation Efficiency Dataset")
        print("=" * 60)
        
        if self.data_path is None:
            print("Warning: No data path provided. Using mock data.")
            # Create mock data for demonstration
            n_samples = max_samples or 1000
            self.embeddings = np.random.randn(n_samples, self.embedder.embedding_dim)
            self.te_values = np.random.rand(n_samples) * 10
            self.sequences = [f"MOCK_SEQ_{i}" for i in range(n_samples)]
        else:
            # Load real data
            # Check for cell line column if supported
            cell_line_col = "cell_line" # Default assumption or make configurable
            df, self.cell_line_map = load_zheng_data(
                self.data_path, 
                max_samples=max_samples,
                cell_line_column=cell_line_col
            )
            self.sequences = df['sequence'].tolist()
            
            # Extract TE
            self.targets_dict = {}
            if 'TE' in df.columns:
                self.targets_dict['translation_efficiency'] = df['TE'].values
                self.te_values = df['TE'].values # Legacy back-compat
            else:
                 raise ValueError("Dataset must contain TE column")

            # Extract HalfLife if present
            if 'HalfLife' in df.columns:
                print("Found HalfLife column. Enabling multi-metric training.")
                self.targets_dict['half_life'] = df['HalfLife'].values
                if 'half_life' not in self.target_metrics:
                    self.target_metrics.append('half_life')
            
            self.num_cell_lines = len(self.cell_line_map)
            
            # Store indices if available
            if 'cell_line_idx' in df.columns:
                self.cell_line_indices = df['cell_line_idx'].values
            else:
                self.cell_line_indices = None
            
            # Generate embeddings
            print(f"\nGenerating embeddings for {len(self.sequences)} sequences...")
            self.embeddings = self.embedder.embed_sequences(
                self.sequences,
                batch_size=8,
                return_numpy=True
            )
        
        print(f"\nDataset prepared:")
        print(f"  Samples: {len(self.sequences)}")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
        print(f"  Metrics: {self.target_metrics}")
        
        stats = {
            'n_samples': len(self.sequences),
            'embedding_dim': self.embeddings.shape[1],
            'metrics': self.target_metrics
        }
        return stats
    
    def step2_train_critic(
        self,
        hidden_dims: List[int] = [512, 256],
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> Dict:
        """
        Step 2: Train critic model to predict translation efficiency.
        
        Args:
            hidden_dims: Hidden layer dimensions for MLP
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training statistics
        """
        print("\n" + "=" * 60)
        print("STEP 2: Training Translation Efficiency Critic")
        print("=" * 60)
        
        # Create critic model
        self.critic = MultiMetricCritic(
            input_dim=self.embeddings.shape[1],
            metrics=self.target_metrics,
            shared_dims=hidden_dims,
            dropout=0.3,
            num_cell_lines=self.num_cell_lines,
            cell_embedding_dim=32 if self.num_cell_lines > 0 else 0
        )
        
        print(f"Critic architecture initialized for {self.num_cell_lines} cell lines.")
        print(f"Total parameters: {self.critic.get_num_parameters():,}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.embeddings,
            self.targets_dict,
            cell_line_indices=getattr(self, 'cell_line_indices', None),
            train_split=0.8,
            batch_size=batch_size
        )
        
        # Create trainer
        trainer = MultiMetricTrainer(
            self.critic,
            learning_rate=learning_rate,
            device=str(self.device)
        )
        
        # Training loop
        print(f"\nTraining for {num_epochs} epochs...")
        best_val_r2 = -float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = trainer.train_epoch(train_loader, verbose=False)
            
            # Validate
            val_results = trainer.validate(val_loader)
            
            # Aggregate metrics
            val_loss = np.mean([v[0] for v in val_results.values()])
            val_r2 = np.mean([v[1] for v in val_results.values()])
            
            if (epoch + 1) % 10 == 0:
                metrics_str = ", ".join([f"{k}: R²={v[1]:.2f}" for k, v in val_results.items()])
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Avg Loss: {val_loss:.4f}, "
                      f"Avg R²: {val_r2:.4f} "
                      f"({metrics_str})")
            
            # Save best model (using Average R2)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                trainer.save_checkpoint('models/critic_best.pt')
        
        print(f"\nTraining complete! Best validation R²: {best_val_r2:.4f}")
        
        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_r2': best_val_r2
        }
    
    def step3_initialize_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32
    ):
        """
        Step 3: Initialize LoRA adapter for Evo model.
        
        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
        """
        print("\n" + "=" * 60)
        print("STEP 3: Initializing LoRA Adapter")
        print("=" * 60)
        
        self.lora_model = EvoLoRAAdapter(
            model_name=self.model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=str(self.device)
        )
        
        print("LoRA adapter initialized!")
    
    def step4_ppo_training(
        self,
        num_epochs: int = 10,
        steps_per_epoch: int = 50
    ):
        """
        Step 4: Train LoRA using PPO with critic rewards.
        
        Args:
            num_epochs: Number of PPO epochs
            steps_per_epoch: Training steps per epoch
        """
        print("\n" + "=" * 60)
        print("STEP 4: PPO Training")
        print("=" * 60)
        
        if self.critic is None:
            raise ValueError("Critic must be trained before PPO (run step2_train_critic)")
        if self.lora_model is None:
            raise ValueError("LoRA must be initialized before PPO (run step3_initialize_lora)")
        
        # Create PPO trainer
        config = PPOTrainingConfig(
            batch_size=4,
            ppo_epochs=4,
            learning_rate=1e-5
        )
        
        from src.sequence_generation.codon_table import reverse_translate_cds

        self.ppo_trainer = RNAPPOTrainer(
            lora_model=self.lora_model,
            critic_model=self.critic,
            embedder=self.embedder,
            config=config,
            device=str(self.device)
        )
        
        # Generate training prompts from dataset
        print("Generating training data from dataset sequences...")
        n_train = min(1000, len(self.sequences)) # Use more samples if available
        sample_indices = np.random.choice(len(self.sequences), n_train, replace=False)
        
        prompts = []
        target_tes = []
        aa_seqs = []
        
        for idx in sample_indices:
            seq = self.sequences[idx]
            te = self.te_values[idx]
            
            # Assuming sequence is CDS + UTRs or just CDS
            # For simplicity in this demo, we assume the input sequence is valid CDS or we extract CDS
            # If full sequence, we might need to know UTR lengths. 
            # For now, let's try to translate the whole sequence or handle errors
            try:
                # Normalize sequence for translation
                # 1. Convert T to U (if DNA)
                # 2. Truncate to multiple of 3
                # 3. Ensure uppercase
                clean_seq = seq.upper().replace('T', 'U')
                remainder = len(clean_seq) % 3
                if remainder > 0:
                    clean_seq = clean_seq[:-remainder]
                
                # Naive attempt: assume sequence is CDS
                aa_seq = reverse_translate_cds(clean_seq)
                
                # Create prompt
                prompts.append(f"Generate RNA with TE {te:.2f}")
                target_tes.append(te)
                aa_seqs.append(aa_seq)
            except Exception as e:
                # If translation fails (e.g. valid UTRs + CDS mixed, or restricted chars)
                # For debug, print first failure
                if len(prompts) == 0:
                   print(f"Failed to translate seq: {e}")
                continue
                
        if len(prompts) == 0:
            raise ValueError("No valid sequences found for training! Ensure data contains valid CDS.")
            
        print(f"Prepared {len(prompts)} training samples with constraints.")
        
        # Train
        self.ppo_trainer.train(
            prompts=prompts,
            target_efficiencies=target_tes,
            amino_acid_sequences=aa_seqs,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch
        )
        
        self.trained = True
        print("\nPPO training complete!")
    
    def generate_optimized_sequence(
        self,
        utr5: str,
        utr3: str,
        amino_acid_sequence: str,
        target_efficiency: float,
        cell_line: Optional[str] = None,
        num_candidates: int = 10
    ) -> Dict:
        """
        Generate optimized RNA sequence for given constraints.
        
        Args:
            utr5: 5' UTR sequence
            utr3: 3' UTR sequence
            amino_acid_sequence: Amino acid sequence to encode
            target_efficiency: Target translation efficiency
            cell_line: Target cell line name (e.g. 'HEK293')
            num_candidates: Number of candidate sequences to generate
            
        Returns:
            Dictionary with best sequence and predictions
        """
        if not self.trained:
            print("Warning: Pipeline not fully trained. Results may be suboptimal.")
        
        # Format prompt
        prompt = format_conditional_prompt(
            utr5=utr5,
            utr3=utr3,
            amino_acid_sequence=amino_acid_sequence,
            target_efficiency=target_efficiency,
            cell_line=cell_line
        )
        
        # Generate candidates using LoRA model
        candidates = self.lora_model.generate_sequences(
            prompt=prompt,
            num_sequences=num_candidates,
            temperature=0.8
        )
        
        # Prepare cell line index if needed
        cell_idx_tensor = None
        if cell_line and self.cell_line_map:
            if cell_line in self.cell_line_map:
                idx = self.cell_line_map[cell_line]
                cell_idx_tensor = torch.tensor([idx], device=self.device)
            else:
                print(f"Warning: Cell line '{cell_line}' not in training data. Using generic prediction.")

        # Score candidates with critic
        scores = []
        for seq in candidates:
            embedding = self.embedder.embed_sequence(seq, return_numpy=False)
            embedding = embedding.unsqueeze(0).to(self.device)
            
            # Predict
            pred_dict = self.critic.predict(
                embedding, 
                cell_line_indices=cell_idx_tensor
            )
            # Default to TE metric
            pred_te = pred_dict['translation_efficiency'].item()
            scores.append(pred_te)
        
        # Find best sequence
        best_idx = np.argmin([abs(score - target_efficiency) for score in scores])
        
        return {
            'best_sequence': candidates[best_idx],
            'predicted_te': scores[best_idx],
            'target_te': target_efficiency,
            'all_candidates': candidates,
            'all_scores': scores
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RNA Codon Optimization Pipeline - Example")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RNACodonOptimizationPipeline(
        data_path=None,  # Using mock data
        model_name="togethercomputer/evo-1-8k-base"
    )
    
    # Run pipeline steps
    pipeline.step1_prepare_data(max_samples=100)
    pipeline.step2_train_critic(num_epochs=10, batch_size=16)
    pipeline.step3_initialize_lora(lora_r=8, lora_alpha=16)
    
    print("\nPipeline initialized successfully!")
    print("To complete training, run: pipeline.step4_ppo_training()")
