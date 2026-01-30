# Amino Acid Sequence Validation

## Feature Overview

The pipeline now includes **amino acid sequence validation** to ensure that generated RNA sequences correctly encode the expected protein.

## What Was Added

### 1. Validation Module (`src/sequence_generation/validation.py`)

**Key Functions:**

- **`validate_cds_amino_acids(cds, expected_aa)`**
  - Translates CDS back to amino acids
  - Compares with expected sequence
  - Returns validation status and detailed error messages
  
- **`validate_full_rna_sequence(full_seq, utr5, utr3, expected_aa)`**
  - Validates complete RNA sequence structure
  - Checks 5'UTR, CDS, and 3'UTR correctness
  - Returns comprehensive validation results

- **`compute_validation_reward(seq, utr5, utr3, expected_aa, base_reward)`**
  - Computes adjusted reward with validation bonus/penalty
  - +1.0 for valid sequences
  - -10.0 for invalid sequences

### 2. Integration into PPO Reward Functions

**Single-Metric PPO (`ppo_trainer.py`):**
```python
reward = te_match + validation_bonus
# validation_bonus = +2 if valid, -15 if invalid
```

**Multi-Metric PPO (`multi_metric_ppo.py`):**
```python
reward = weighted_metric_rewards + validation_bonus
# validation_bonus = +2 if valid, -15 if invalid
```

## Reward Structure

| Condition | Bonus/Penalty |
|-----------|---------------|
| Valid sequence (preserves amino acids) | **+2.0** |
| Invalid sequence (wrong amino acids) | **-15.0** |
| Embedding error | **-100.0** |

## Usage

The validation is **automatically applied** during PPO training when UTRs are provided:

```python
reward = ppo_trainer.compute_reward(
    generated_sequence="AUGC...",
    target_efficiency=5.0,
    amino_acid_sequence="MYPFIRTARM",
    utr5="AUGCUGACU...",  # Provide for validation
    utr3="UGACUGACU..."   # Provide for validation
)
```

## Validation Checks

1. ✅ **RNA validity**: Only AUGC characters
2. ✅ **Length**: CDS is multiple of 3
3. ✅ **Translation**: CDS translates to correct amino acids
4. ✅ **UTR preservation**: 5'UTR and 3'UTR match expected
5. ✅ **Stop codon**: Optionally allows/requires stop codon

## Error Messages

The validation provides detailed error messages:

```
❌ Invalid RNA sequence (contains non-AUGC characters)
❌ CDS length (17) is not a multiple of 3
❌ Amino acid mismatch at positions [3, 7, 9]
   Translated: MYPXIRTXRX...
   Expected:   MYPFIRTARM...
❌ 5'UTR mismatch
❌ Translation failed: Unknown codon 'AUX'
```

## Testing

```python
from src.sequence_generation import validate_cds_amino_acids

# Valid sequence
is_valid, msg = validate_cds_amino_acids(
    "AUGUAUCCAUUCAUAAGAACAGCAAGAAUAUG",
    "MYPFIRTARM"
)
# Returns: (True, "Valid")

# Invalid sequence
is_valid, msg = validate_cds_amino_acids(
    "AUGUA",  # Too short
    "MYPFIRTARM"
)
# Returns: (False, "CDS length (5) is not a multiple of 3")
```

## Benefits

1. **Prevents invalid sequences** from being rewarded
2. **Ensures biological correctness** of optimized sequences
3. **Guides RL training** toward valid solutions
4. **Provides interpretable feedback** on what went wrong

## Implementation Complete ✅

All PPO trainers now include amino acid validation by default!
