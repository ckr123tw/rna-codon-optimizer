#!/bin/bash

# Ensure we use the Miniforge environment libraries
export LD_LIBRARY_PATH=/home/ckr/miniforge3/lib:$LD_LIBRARY_PATH

# Run the training script
# Pass all arguments to the python script
/home/ckr/miniforge3/bin/python scripts/train_ppo.py "$@"
