# RNA Codon Optimization Pipeline - Multi-Metric

## Datasets

### Translation Efficiency

Download Supplementary Table 1 from Zheng et al. 2025:
https://pmc.ncbi.nlm.nih.gov/articles/instance/12323635/bin/NIHMS2101086-supplement-Supplementary_Table1.xlsx

Save as: `supplementary_table1.xlsx`

### mRNA Half-Life

Download supplementary data from Cetnar et al. 2024:

**Paper:** Cetnar DP, Hossain A, Vezeau GE, Salis HM. Predicting synthetic mRNA stability using massively parallel kinetic measurements, biophysical modeling, and machine learning. Nat Commun. 2024 Nov 6;15(1):9601. doi: 10.1038/s41467-024-54059-7

**PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11541907/

**Data availability:** Check the paper's supplementary materials for:
- mRNA sequences
- Half-life measurements (hours)
- Experimental conditions

Save as: `cetnar_half_life.xlsx` (or appropriate format)

## Usage

Both datasets will be automatically loaded and merged by the pipeline when available.
