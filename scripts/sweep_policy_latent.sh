#!/bin/bash

GPU=${1:-0}   # default to GPU 0 if no argument provided
export CUDA_VISIBLE_DEVICES=$GPU

python scripts/run.py --repo policy_latent --sigma 0.3 --seed 2
python scripts/run.py --repo policy_latent --sigma 0.5 --seed 2
python scripts/run.py --repo policy_latent --sigma 0.7 --seed 2