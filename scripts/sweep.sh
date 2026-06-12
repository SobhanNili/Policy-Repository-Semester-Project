#!/bin/bash

GPU=${1:-0}   # default to GPU 0 if no argument provided
export CUDA_VISIBLE_DEVICES=$GPU

python scripts/run.py --repo dino --sigma 0.3
python scripts/run.py --repo dino --sigma 0.5
python scripts/run.py --repo dino --sigma 0.7

python scripts/run.py --repo morphology --sigma 0.15
python scripts/run.py --repo morphology --sigma 0.3
python scripts/run.py --repo morphology --sigma 0.5

python scripts/run.py --repo policy_latent --sigma 0.3
python scripts/run.py --repo policy_latent --sigma 0.5
python scripts/run.py --repo policy_latent --sigma 0.7

python scripts/run.py --repo performance --mode rank
python scripts/run.py --repo performance --mode solve_rate
python scripts/run.py --repo performance --mode hybrid

python scripts/run.py --repo random