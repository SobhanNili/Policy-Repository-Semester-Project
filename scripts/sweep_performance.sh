#!/bin/bash

GPU=${1:-0}   # default to GPU 0 if no argument provided
export CUDA_VISIBLE_DEVICES=$GPU

python scripts/run.py --repo performance --mode return_rank
python scripts/run.py --repo performance --mode solve_rate
python scripts/run.py --repo performance --mode hybrid