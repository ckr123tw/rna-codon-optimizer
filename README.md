# RNA Codon Optimization Pipeline

**AI-powered RNA sequence optimization using reinforcement learning and foundation models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview

This pipeline optimizes RNA codon sequences for desired properties using:
- **Evo-1-8k** foundation model for RNA sequence embeddings
- **Multi-metric critic** predicting translation efficiency and mRNA half-life
- **LoRA** for parameter-efficient fine-tuning
- **PPO** for reinforcement learning-based optimization

## Features

✅ Multi-metric optimization (translation efficiency + mRNA half-life)  
✅ Cell line and tissue conditioning  
✅ Human codon usage frequencies  
✅ End-to-end pipeline from data to optimized sequences  
✅ Configurable via YAML  
✅ GPU + CPU support

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ckr123tw/rna-codon-optimizer.git
cd rna-codon-optimizer

# Install dependencies
pip install -r requirements.txt

# Download datasets (see data/README.md)

# Run quick test
python example_usage.py --mode quick

# Run full pipeline
python example_usage.py --mode full
```

## Usage

```python
from src.pipeline import RNACodonOptimizationPipeline

# Initialize with multi-metric support
pipeline = RNACodonOptimizationPipeline(
    data_paths={
        'te': 'data/supplementary_table1.xlsx',
        'half_life': 'data/cetnar_half_life.xlsx'
    },
    metrics=['translation_efficiency', 'half_life']
)

# Train
pipeline.train_multi_metric_critic()

# Generate optimized sequence
result = pipeline.generate_optimized_sequence(
    utr5="AUGCUGACU...",
    utr3="UGACUGACU...",
    amino_acid_sequence="MYPFIRTARM",
    targets={
        'translation_efficiency': 5.0,
        'half_life': 4.5
    },
    cell_line="HEK293",
    metric_weights={'translation_efficiency': 0.6, 'half_life': 0.4}
)
```

## Hardware Requirements

**For 1-week training:**
- GPU: NVIDIA A100 (40GB) or A6000
- RAM: 64GB+
- Storage: 500GB SSD

**Cloud options:**  
Lambda Labs 8x A100: ~$12/hour

See [enhancement_plan.md](docs/enhancement_plan.md) for details.

## Datasets

**Translation Efficiency:** Zheng et al. 2025 ([DOI: 10.1038/s41587-025-02712-x](https://doi.org/10.1038/s41587-025-02712-x))  
**mRNA Half-Life:** Cetnar et al. 2024 ([DOI: 10.1038/s41467-024-54059-7](https://doi.org/10.1038/s41467-024-54059-7))

See `data/README.md` for download links.

## Project Structure

```
rna-codon-optimizer/
├── src/
│   ├── sequence_generation/  # Codon selection & Evo embeddings
│   ├── critic/               # Multi-metric prediction models
│   ├── lora_generation/      # LoRA-based generation
│   ├── ppo_training/         # PPO optimization
│   └── pipeline.py           # End-to-end integration
├── data/                     # Datasets
├── models/                   # Checkpoints
├── configs/                  # YAML configuration
└── tests/                    # Unit tests
```

## Citation

If you use this pipeline, please cite:

```bibtex
@article{zheng2025predicting,
  title={Predicting the translation efficiency of messenger RNA in mammalian cells},
  author={Zheng, Dinghai and Persyn, Logan and Wang, Jun and Liu, Yue and Ulloa-Montoya, Fernando and Cenik, Can and Agarwal, Vikram},
  journal={Nature Biotechnology},
  year={2025},
  doi={10.1038/s41587-025-02712-x}
}

@article{cetnar2024predicting,
  title={Predicting synthetic mRNA stability using massively parallel kinetic measurements, biophysical modeling, and machine learning},
  author={Cetnar, Daniel P and Hossain, Ashraful and Vezeau, Gina E and Salis, Howard M},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={9601},
  year={2024},
  doi={10.1038/s41467-024-54059-7}
}
```

## License

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)

This work is licensed for **academic and non-commercial use only**. 

**You may:**
- ✅ Use for academic research
- ✅ Use for educational purposes
- ✅ Modify and distribute (with attribution)
- ✅ Use in non-profit organizations

**You may NOT:**
- ❌ Use in commercial products or services
- ❌ Use for commercial profit
- ❌ Use in for-profit companies without permission

**For commercial licensing**, please contact [your contact information].

See [LICENSE](LICENSE) file for full terms.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Acknowledgments

- Evo genomic foundation model by TogetherComputer
- Datasets from Zheng et al. 2025 and Cetnar et al. 2024
