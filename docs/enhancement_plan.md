# RNA Codon Optimizer: Enhancement Plan

This document outlines the roadmap for future development and scaling of the RNA Codon Optimization pipeline.

## 1. Advanced Conditioning

### Cell Line and Tissue Specificity
**Goal:** Optimize RNA sequences for specific cellular environments (e.g., HEK293, HeLa, Liver, Muscle).
- **Implementation:**
    - Train separate critic heads or conditional embeddings for different cell lines using cell-specific TE datasets.
    - Integrate tissue-specific codon usage tables as prior priors in the generation step.
    - **Data Source:** Expand dataset collection to include cell-specific ribosome profiling data (e.g., from disparate tissues in Zheng et al. 2025).

### 5' and 3' UTR Co-Optimization
**Goal:** Jointly optimize UTRs and CDS for maximum stability and translation.
- **Implementation:**
    - Extend the Evo-1-8k context window usage to include full UTR sequences during PPO training.
    - Use the critic to score the entire mRNA molecule (5'UTR + CDS + 3'UTR) rather than just the CDS.

## 2. Model & Algorithm Improvements

### Scaling PPO (Proximal Policy Optimization)
- **Goal:** Improve training stability and exploration.
- **Implementation:**
    - **KL Penalty Scheduling:** Dynamically adjust KL divergence penalties to prevent the model from straying too far from biological "naturalness".
    - **Rejection Sampling:** Implement "Best-of-N" sampling as a stronger baseline or alternative to PPO for rapid inference.

### Critic Model Architecture
- **Goal:** Increase prediction accuracy for translation efficiency and half-life.
- **Implementation:**
    - Move from simple MLP heads to **Attention-based Pooling** layers on top of Evo embeddings.
    - Experiment with **Graph Neural Networks (GNNs)** if secondary structure features are explicitly integrated.

## 3. Deployment & Compute

### Distributed Training Support
- **Goal:** Reduce training time from days to hours.
- **Implementation:**
    - Integrate `accelerate` fully for multi-GPU training.
    - Implement DeepSpeed ZERO-3 offloading to train 7B+ models on consumer hardware (e.g., 24GB GPUs).

### Web Interface
- **Goal:** Make the tool accessible to non-computational biologists.
- **Implementation:**
    - Build a Streamlit or Gradio interface.
    - Allow users to input Amino Acid sequences and selecting target cell lines via dropdowns.

## 4. Experimental Validation (Wet Lab Loop)

**Goal:** Verify computational predictions in vitro/in vivo.
1.  **Design:** Generate 10-20 optimized variants for a reporter gene (e.g., eGFP, Firefly Luciferase).
2.  **Synthesis:** Order mRNA synthesis or DNA plasmids.
3.  **Transfection:** Transfect into target cell lines.
4.  **Measurement:** Measure protein production (Fluorescence/Luminescence) and mRNA decay (qPCR).
5.  **Feedback:** Re-train Critic models with these high-confidence ground truth data points.

---

*Last Updated: 2026-01-30*
