# Training Configuration & Monitoring Guide

This guide explains how to configure hyperparameters for the RNA codon optimizer training pipeline and how to monitor training progress effectively.

## Table of Contents

1. [Training Overview](#training-overview)
2. [Critic Training Configuration](#critic-training-configuration)
3. [PPO Training Configuration](#ppo-training-configuration)
4. [Data Splitting](#data-splitting)
5. [Monitoring Training Progress](#monitoring-training-progress)
6. [Troubleshooting](#troubleshooting)

---

## Training Overview

The RNA codon optimizer has two main training phases:

1. **Critic Training**: Trains an MLP to predict translation efficiency from RNA embeddings
2. **PPO Training**: Fine-tunes the LoRA adapter using the critic as a reward signal

```python
from src.pipeline import RNACodonOptimizationPipeline

# Initialize pipeline
pipeline = RNACodonOptimizationPipeline(
    data_path="data/your_dataset.csv",
    model_name="togethercomputer/evo-1-8k-base"
)

# Step 1: Prepare data
pipeline.step1_prepare_data(max_samples=None)

# Step 2: Train critic
results = pipeline.step2_train_critic(
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    early_stopping_patience=10
)

# Step 3: Initialize LoRA
pipeline.step3_initialize_lora(lora_r=16, lora_alpha=32)

# Step 4: PPO training
pipeline.step4_ppo_training(num_epochs=20, steps_per_epoch=50)
```

---

## Critic Training Configuration

### Key Hyperparameters

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| `num_epochs` | 50 | Maximum training epochs | 30-100 typical; early stopping handles overfitting |
| `batch_size` | 32 | Samples per batch | 16-128; larger = faster, smaller = more stochastic |
| `learning_rate` | 1e-3 | Optimizer learning rate | 1e-3 to 1e-4; lower if unstable |
| `hidden_dims` | [512, 256] | MLP hidden layer sizes | Larger for complex patterns, smaller for limited data |
| `early_stopping_patience` | 10 | Epochs without improvement | 5-20; higher for noisy data |

### Data Split Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_ratio` | 0.70 | Training data fraction |
| `dev_ratio` | 0.15 | Validation/dev data fraction |
| `test_ratio` | 0.15 | Held-out test data fraction |

### Example Configurations

**Quick Experiment (Limited Data)**
```python
results = pipeline.step2_train_critic(
    hidden_dims=[256, 128],
    num_epochs=30,
    batch_size=16,
    learning_rate=1e-3,
    early_stopping_patience=5,
    train_ratio=0.80,
    dev_ratio=0.10,
    test_ratio=0.10
)
```

**Production Training (Full Dataset)**
```python
results = pipeline.step2_train_critic(
    hidden_dims=[512, 256, 128],
    num_epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    early_stopping_patience=15,
    train_ratio=0.70,
    dev_ratio=0.15,
    test_ratio=0.15
)
```

---

## PPO Training Configuration

### Key Hyperparameters

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| `learning_rate` | 1e-5 | Policy learning rate | 1e-5 to 1e-6; very important for stability |
| `batch_size` | 4 | Samples per PPO update | 4-16; larger = more stable |
| `ppo_epochs` | 4 | Updates per batch | 4-8; too high may overfit |
| `cliprange` | 0.2 | PPO clipping parameter | 0.1-0.3; lower = conservative |
| `target_kl` | 6.0 | KL divergence target | 0.01-0.1 for stability |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 10-20 | PPO training epochs |
| `steps_per_epoch` | 50-100 | Training steps per epoch |
| `early_stopping_patience` | 5 | Epochs without reward improvement |

### Example Configuration

```python
from src.ppo_training.ppo_trainer import PPOTrainingConfig

config = PPOTrainingConfig(
    learning_rate=5e-6,      # Slightly lower for stability
    batch_size=8,            # Larger batch for stable gradients
    ppo_epochs=4,            # Standard value
    cliprange=0.2,           # Standard value
    target_kl=0.1            # Conservative KL target
)

# In pipeline
results = ppo_trainer.train(
    prompts=prompts,
    target_efficiencies=targets,
    amino_acid_sequences=aa_seqs,
    num_epochs=20,
    steps_per_epoch=50,
    early_stopping_patience=5,
    verbose=True
)
```

---

## Data Splitting

### Using the Training Utilities

```python
from src.training import create_data_loaders_with_test, DataSplitConfig

# Configure splits
config = DataSplitConfig(
    train_ratio=0.70,  # 70% for training
    dev_ratio=0.15,    # 15% for validation during training
    test_ratio=0.15,   # 15% for final evaluation
    batch_size=32,
    random_seed=42     # For reproducibility
)

# Create data loaders
train_loader, dev_loader, test_loader = create_data_loaders_with_test(
    embeddings,
    target_values={'translation_efficiency': te_values},
    cell_line_indices=cell_indices,
    config=config
)
```

### Split Recommendations

| Dataset Size | Recommended Split |
|--------------|-------------------|
| < 1,000 | 80/10/10 or 5-fold CV |
| 1,000 - 10,000 | 70/15/15 (default) |
| > 10,000 | 70/15/15 or 60/20/20 |

---

## Monitoring Training Progress

### Console Output Interpretation

**Critic Training Output:**
```
Epoch  15/ 50 | Train Loss: 0.2341 | Val Loss: 0.2567 | Val R²: 0.8234 ★
```

- **Train Loss ↓**: Should decrease (lower is better)
- **Val Loss ↓**: Should decrease, watch for gap with train loss
- **Val R² ↑**: Should increase toward 1.0 (higher is better)
- **★ symbol**: Indicates new best validation performance

**Warning Signs:**
- Train loss much lower than val loss → Overfitting
- Val R² not improving → May need different architecture
- Early stopping triggered → Model has converged

**PPO Training Output:**
```
Epoch 5 Summary: ★ (best)
  Avg Reward:  2.3456 ± 0.5678
  Range:       [1.2345, 3.4567]
  Eval Reward: 2.4567
```

- **Avg Reward ↑**: Should increase over training
- **Range**: Shows best/worst case outcomes
- **Eval Reward**: Performance on held-out prompts

### Training History Files

Training automatically saves history to JSON files:

```python
# Critic training history
with open('models/critic_training_history.json') as f:
    history = json.load(f)
    
# Access metrics
train_losses = [e['loss'] for e in history['train_history']]
val_r2_scores = [e['avg_r2'] for e in history['val_history']]
test_results = history['test_results']
```

### Plotting Learning Curves

```python
import matplotlib.pyplot as plt

# Get learning curve data
curves = tracker.get_learning_curves()

# Plot losses
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(curves['epochs'], curves['train_loss'], label='Train')
plt.plot(curves['epochs'], curves['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(curves['epochs'], curves['val_r2'])
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('Validation R² Score')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Val loss increasing while train loss decreases | Overfitting | Reduce epochs, add dropout, reduce model size |
| R² stuck near 0 | Learning rate too high/low | Try 1e-3 to 1e-4 range |
| Early stopping too early | Patience too low | Increase `early_stopping_patience` |
| PPO rewards not improving | Learning rate too high | Reduce to 1e-6, check reward signal |
| Out of memory | Batch size too large | Reduce `batch_size` |

### Recommended Workflow

1. **Start small**: Use max_samples=1000 for initial experiments
2. **Quick validation**: Use early_stopping_patience=5 initially
3. **Monitor closely**: Watch for overfitting (train/val gap)
4. **Scale up**: Increase data and epochs once confident in setup
5. **Save checkpoints**: Always save best model for recovery

### Getting Help

If training issues persist:
1. Check training history JSON for patterns
2. Visualize learning curves
3. Try reducing model complexity
4. Verify data quality (check for NaNs, outliers)
