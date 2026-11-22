#!/bin/bash
# Pre-training validation on a GPU (gpu-debug, < 1 h):
#   1. preflight.py checks with the spec architecture (4 full sessions of data/small)
#   2. smoke run of train.py on data/small (spec architecture, 3 epochs = 87 steps):
#      uninterrupted reference run, plus a run stopped after epoch 1 and resumed
#   3. timing: ~7 min of train.py on data/full (step-0 full-val eval + steps/s)
# Submit from the repo root:  sbatch training/jobs/smoke_gpu.sbatch
#SBATCH -J poker-smoke
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH -o training/logs/smoke-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11-5e66
source "${VENV:-/oscar/home/msamiull/projects/cs2470/.venv_fp}/bin/activate"  # set VENV to use another environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY="python -u -W ignore::DeprecationWarning"
S=training/checkpoints/smoke_${SLURM_JOB_ID}
echo "host $(hostname) | job $SLURM_JOB_ID | start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== 1. preflight"
$PY training/preflight.py --data data/small --out training/results/preflight_gpu.json

echo "=== 2a. smoke: uninterrupted reference"
$PY training/train.py --data data/small --out_dir $S/ref --seed 0 --eval_interval 20
echo "exit code $?"
echo "=== 2b. smoke: stop after epoch 1, then resume"
$PY training/train.py --data data/small --out_dir $S/resume --seed 0 --eval_interval 20 --stop_after_epoch 1
$PY training/train.py --data data/small --out_dir $S/resume --seed 0 --eval_interval 20 --resume
echo "exit code $?"

echo "=== 3. timing on data/full (killed after 7 min)"
timeout 420 $PY training/train.py --data data/full --out_dir $S/timing --seed 0 --log_interval 20 --eval_interval 1000000
echo "end $(date)"
