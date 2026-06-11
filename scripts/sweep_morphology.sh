#!/bin/bash

GPU=${1:-0}   # default to GPU 0 if no argument provided
export CUDA_VISIBLE_DEVICES=$GPU

python scripts/run.py --repo morphology --sigma 0.15
python scripts/run.py --repo morphology --sigma 0.3
python scripts/run.py --repo morphology --sigma 0.5