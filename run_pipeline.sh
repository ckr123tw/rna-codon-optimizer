#!/bin/bash

# Training Pipeline Wrapper
# Usage: ./run_pipeline.sh --data_path data/dataset.csv [args]

# Setup environment (optional: for Conda users)
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

echo "Starting RNA Codon Optimization Pipeline Training..."
python scripts/training/run_pipeline.py "$@"
