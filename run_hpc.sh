#!/usr/bin/env bash
# run_hpc.sh — submit a Parameter Golf training job on ETH Euler.
#
# Usage:
#   sbatch run_hpc.sh
#   sbatch --export=ALL,RUN_ID=v2_smoke,ITERATIONS=200 run_hpc.sh
#   sbatch --export=ALL,GPU_TYPE=a100-pcie-40gb,N_GPUS=4 run_hpc.sh

set -euo pipefail

CLUSTER_REPO="/cluster/home/nlorenzon/github/parameter-golf"
LOG_DIR="/cluster/scratch/nlorenzon/logs/param-golf"
CONDA_ENV="/cluster/scratch/nlorenzon/envs/param-golf"

# Defaults (override via --export or environment)
GPU_TYPE="${GPU_TYPE:-rtx_4090}"
N_GPUS="${N_GPUS:-4}"

mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=param-golf
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8192
#SBATCH --time=01:00:00
#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}
#SBATCH --output=${LOG_DIR}/pg_%j.out
#SBATCH --error=${LOG_DIR}/pg_%j.err

echo "=== Job \$SLURM_JOB_ID started on \$(hostname) at \$(date) ==="
echo "=== GPUs: ${GPU_TYPE} x ${N_GPUS} ==="

cd ${CLUSTER_REPO}

# Internet access for HuggingFace downloads
module load eth_proxy

# Load env vars if present
if [[ -f ".env" ]]; then
  set -o allexport; source .env; set +o allexport
fi

# Activate conda environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# Download data if not present
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

# Training config
export RUN_ID="\${RUN_ID:-v2_\$(date +%Y%m%d_%H%M%S)}"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024

# Detect GPU count from SLURM
NUM_GPUS=\$(echo \$CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "=== Training: RUN_ID=\${RUN_ID} GPUs=\${NUM_GPUS} ==="

torchrun \\
    --standalone \\
    --nproc_per_node="\${NUM_GPUS}" \\
    train_gpt.py

echo "=== Job \$SLURM_JOB_ID finished at \$(date) ==="
echo "=== Log: logs/\${RUN_ID}.txt ==="
EOF

echo "Job submitted. Logs will appear in: ${LOG_DIR}/"
