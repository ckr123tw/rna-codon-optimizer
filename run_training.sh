#!/bin/bash

# Run the training script with proper environment setup
# For Conda environments, ensure LD_LIBRARY_PATH includes conda lib path
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# Run the training script
# Pass all arguments to the python script
python scripts/train_ppo.py "$@"
