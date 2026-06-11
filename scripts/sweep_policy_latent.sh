#!/bin/bash

GPU=${1:-0}   # default to GPU 0 if no argument provided
export CUDA_VISIBLE_DEVICES=$GPU

python scripts/run.py --repo policy_latent --sigma 0.3
python scripts/run.py --repo policy_latent --sigma 0.5
python scripts/run.py --repo policy_latent --sigma 0.7