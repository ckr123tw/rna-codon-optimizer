#!/bin/bash

# Sequence Generation Wrapper
# Usage: ./generate_sequences.sh --input input_seqs.csv --output results.csv [args]

# Setup environment (optional: for Conda users)
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

echo "Generating Optimized Sequences..."
python scripts/generation/generate_sequences.py "$@"
