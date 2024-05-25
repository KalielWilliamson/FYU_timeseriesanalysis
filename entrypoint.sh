#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' /workspace/.env | xargs)

# Log in to Weights & Biases
wandb login $WANDB_API_KEY

# Start Jupyter notebook
exec jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
