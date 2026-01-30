"""
Example usage of the RNA codon optimization pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import RNACodonOptimizationPipeline
from sequence_generation import create_template_sequence


def example_full_pipeline():
    """
    Example: Run the complete pipeline from data loading to sequence generation.
    """
    print("=" * 70)
    print(" RNA CODON OPTIMIZATION PIPELINE - FULL EXAMPLE")
    print("=" * 70)
    
    # Initialize pipeline
    # Set data_path to the downloaded Supplementary Table 1
    data_path = "data/supplementary_table1.xlsx"
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"\nWarning: Data file not found at {data_path}")
        print("Running with mock data for demonstration...")
        data_path = None
    
    pipeline = RNACodonOptimizationPipeline(
        data_path=data_path,
        model_name="togethercomputer/evo-1-8k-base"
    )
    
    # Step 1: Prepare data
    print("\n" + "▶" * 35)
    data_stats = pipeline.step1_prepare_data(max_samples=500)  # Limit for faster demo
    print(f"Loaded {data_stats['n_samples']} samples")
    
    # Step 2: Train critic
    print("\n" + "▶" * 35)
    critic_stats = pipeline.step2_train_critic(
        hidden_dims=[512, 256],
        num_epochs=30,  # Increase for better performance
        batch_size=32,
        learning_rate=1e-3
    )
    print(f"Critic trained with R² = {critic_stats['best_val_r2']:.4f}")
    
    # Step 3: Initialize LoRA
    print("\n" + "▶" * 35)
    pipeline.step3_initialize_lora(
        lora_r=16,
        lora_alpha=32
    )
    
    # Step 4: PPO training
    print("\n" + "▶" * 35)
    print("Note: PPO training can take several hours on CPU")
    print("For quick demo, using reduced epochs...")
    # Uncomment for full training:
    # pipeline.step4_ppo_training(num_epochs=20, steps_per_epoch=100)
    
    # For demo, skip PPO training
    print("Skipping PPO training in this demo...")
    pipeline.trained = True
    
    # Generate optimized sequence
    print("\n" + "=" * 70)
    print(" GENERATING OPTIMIZED SEQUENCE")
    print("=" * 70)
    
    # Example input
    test_aa_sequence = "MYPFIRTARMFGAQLRK"
    test_utr5 = "GCCGCCACCAUGGGCUACUUUGAU"
    test_utr3 = "UGACUGACUAGCUAGCUUAA"
    target_te = 5.0
    
    print(f"\nInput:")
    print(f"  Amino acids: {test_aa_sequence}")
    print(f"  5'UTR: {test_utr5}")
    print(f"  3'UTR: {test_utr3}")
    print(f"  Target TE: {target_te}")
    
    result = pipeline.generate_optimized_sequence(
        utr5=test_utr5,
        utr3=test_utr3,
        amino_acid_sequence=test_aa_sequence,
        target_efficiency=target_te,
        num_candidates=5
    )
    
    print(f"\nOutput:")
    print(f"  Best sequence: {result['best_sequence'][:80]}...")
    print(f"  Predicted TE: {result['predicted_te']:.4f}")
    print(f"  Target TE: {result['target_te']:.4f}")
    print(f"  Difference: {abs(result['predicted_te'] - result['target_te']):.4f}")
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70)


def example_quick_test():
    """
    Quick test of individual components.
    """
    print("=" * 70)
    print(" QUICK COMPONENT TEST")
    print("=" * 70)
    
    # Test sequence generation
    print("\n1. Testing Sequence Generation...")
    from sequence_generation import RNASequenceGenerator
    
    generator = RNASequenceGenerator(use_codon_weights=True)
    template = create_template_sequence("MYPFIRTARM", utr5_length=30, utr3_length=50)
    
    seq_result = generator.generate_sequence(
        utr5=template['utr5'],
        utr3=template['utr3'],
        amino_acid_sequence="MYPFIRTARM"
    )
    
    print(f"   ✓ Generated sequence: {seq_result['length']} nucleotides")
    print(f"   ✓ Valid: {generator.validate_sequence(seq_result)}")
    
    # Test embedder
    print("\n2. Testing Evo Embedder...")
    from sequence_generation.evo_embedder import EvoEmbedder
    
    embedder = EvoEmbedder()
    embedding = embedder.embed_sequence(seq_result['full_sequence'])
    print(f"   ✓ Embedding shape: {embedding.shape}")
    
    # Test critic model
    print("\n3. Testing Critic Model...")
    from critic import TranslationEfficiencyCritic
    import torch
    
    critic = TranslationEfficiencyCritic(input_dim=embedder.embedding_dim)
    test_emb = torch.tensor(embedding).unsqueeze(0)
    prediction = critic(test_emb)
    print(f"   ✓ Predicted TE: {prediction.item():.4f}")
    
    print("\n" + "=" * 70)
    print(" ALL COMPONENTS WORKING!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RNA Codon Optimization Pipeline Examples")
    parser.add_argument(
        '--mode',
        choices=['quick', 'full'],
        default='quick',
        help='Run quick test or full pipeline'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        example_quick_test()
    else:
        example_full_pipeline()
