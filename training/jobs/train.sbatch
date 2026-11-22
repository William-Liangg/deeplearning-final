#!/bin/bash
# Train one seed per training-specification.md on data/full -> training/checkpoints/seed<SEED>/.
# Auto-resumes from the latest epoch checkpoint if one exists (e.g. after a timeout).
#
# Submit from the repo root, from a COMMITTED state (train.py records the git commit and
# a dirty flag; the code that runs is whatever is on disk when the job starts):
#   mkdir -p training/logs
#   sbatch -J train-s0 training/jobs/train.sbatch 0
#   sbatch -J train-s1 --dependency=afterok:<seed0 jobid> --kill-on-invalid-dep=yes training/jobs/train.sbatch 1
#SBATCH -J poker-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o training/logs/%x-%j.out

set -euo pipefail
SEED=${1:?usage: sbatch training/jobs/train.sbatch <seed>}
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11-5e66
source "${VENV:-/oscar/home/msamiull/projects/cs2470/.venv_fp}/bin/activate"  # set VENV to use another environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_DIR=training/checkpoints/seed${SEED}
RESUME=""
if compgen -G "${RUN_DIR}/epoch_*.pt" > /dev/null; then
    RESUME="--resume"
    echo "found epoch checkpoint(s) in ${RUN_DIR}: resuming"
fi

echo "host $(hostname) | job $SLURM_JOB_ID | seed $SEED | git $(git rev-parse --short HEAD) | start $(date)"
git status --porcelain | sed 's/^/  dirty: /'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# All training settings are train.py's defaults (= the spec): 6x256, 8 heads x 32, MLP 1024,
# AdamW lr 3e-4 constant, wd 0.01, batch 8 x accum 4, clip 1.0, 3 epochs.
python -u -W ignore::DeprecationWarning training/train.py --data data/full --out_dir training/checkpoints \
    --seed "$SEED" $RESUME

echo "end $(date)"
