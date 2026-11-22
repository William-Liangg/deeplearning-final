#!/bin/bash
# Oracle CONTROL model (DECISIONS.md D36): same settings as the main model, but every session
# starts with its true-style token. 1 epoch, seed 0 -> training/checkpoints/oracle/seed0/.
# Submit from the repo root, from a committed state:  sbatch training/jobs/train_oracle.sbatch
#SBATCH -J train-oracle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o training/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11-5e66
source "${VENV:-/oscar/home/msamiull/projects/cs2470/.venv_fp}/bin/activate"  # set VENV to use another environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RESUME=""
if compgen -G "training/checkpoints/oracle/seed0/epoch_*.pt" > /dev/null; then RESUME="--resume"; fi
echo "host $(hostname) | job $SLURM_JOB_ID | git $(git rev-parse --short HEAD) | start $(date)"
git status --porcelain | sed 's/^/  dirty: /'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -u -W ignore::DeprecationWarning training/train.py --data data/full --out_dir training/checkpoints/oracle \
    --seed 0 --epochs 1 --oracle_style $RESUME
echo "end $(date)"
