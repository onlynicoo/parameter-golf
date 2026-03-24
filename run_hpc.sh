#!/bin/bash
#SBATCH --job-name=param-golf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# =============================================================================
# Parameter Golf — HPC training script
# Adjust --gpus-per-node, --partition, and module loads for your cluster.
# =============================================================================

set -euo pipefail

# --- Cluster-specific setup (edit for your HPC) ---
# module load cuda/12.4
# module load python/3.11
# source /path/to/your/venv/bin/activate

NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"
mkdir -p logs

# --- Step 1: Download data (only needs to run once) ---
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

# --- Step 2: Install dependencies ---
pip install -q torch sentencepiece numpy huggingface_hub datasets tqdm zstandard 2>/dev/null || true

# --- Step 3: Train ---
# Baseline run (train_gpt.py, CUDA)
export RUN_ID="${RUN_ID:-hpc_v2_$(date +%Y%m%d_%H%M%S)}"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024

# You can override any hyperparameter via environment variables, e.g.:
# export ITERATIONS=200       # short smoke test
# export NUM_LAYERS=10
# export MLP_MULT=3

echo "=== Starting training: RUN_ID=${RUN_ID} GPUs=${NUM_GPUS} ==="

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    train_gpt.py

echo "=== Training complete ==="
echo "Logs: logs/${RUN_ID}.txt"
