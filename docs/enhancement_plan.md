# RNA Codon Optimizer: Enhancement Plan

This document outlines the roadmap for future development and scaling of the RNA Codon Optimization pipeline.

## 1. Advanced Conditioning

### Cell Line Specificity ✅ IMPLEMENTED
**Status:** Implemented and tested.
- **What's done:**
    - Cell line embeddings added to `MultiMetricCritic`
    - `format_conditional_prompt` includes cell line in generation prompts
    - `load_zheng_data` maps cell line names to indices
    - End-to-end training tested with HEK293, HeLa, HepG2 cell lines

### Tissue Specificity (Future)
**Goal:** Optimize RNA sequences for specific tissues (Liver, Muscle, Brain, etc.).
- **Not yet implemented.** Requires:
    - Tissue-specific codon usage tables as generation priors
    - Tissue-specific ribosome profiling datasets (beyond cell lines)
    - Tissue embedding layer in critic (similar to cell line implementation)

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
