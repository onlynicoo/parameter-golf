#!/usr/bin/env bash
# run_interactive.sh — run Parameter Golf training inside an interactive srun session.
#
# Usage (from an already-allocated interactive node):
#   bash run_interactive.sh
#   RUN_ID=v2_smoke ITERATIONS=200 bash run_interactive.sh

set -euo pipefail

CLUSTER_REPO="/cluster/home/nlorenzon/github/parameter-golf"
CONDA_ENV="/cluster/scratch/nlorenzon/envs/param-golf"

cd "$CLUSTER_REPO"

module load eth_proxy

if [[ -f ".env" ]]; then
  set -o allexport; source .env; set +o allexport
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Download data if not present
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

export RUN_ID="${RUN_ID:-v2_$(date +%Y%m%d_%H%M%S)}"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "=== Training: RUN_ID=${RUN_ID} GPUs=${NUM_GPUS} ==="

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train_gpt.py

echo "=== Done: RUN_ID=${RUN_ID} ==="
