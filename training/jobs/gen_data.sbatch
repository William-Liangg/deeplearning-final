#!/bin/bash
# Generate the full 100k-session dataset (~9 min, single-threaded) and tokenize it.
# Submit from the repo root:  sbatch training/jobs/gen_data.sbatch
#SBATCH -J poker-gen-data
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o training/logs/gen_data-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.11-5e66
source "${VENV:-/oscar/home/msamiull/projects/cs2470/.venv_fp}/bin/activate"  # set VENV to use another environment

echo "host $(hostname) | start $(date)"
python -u data/simulate.py --sessions 100000 --out data/sessions_full.jsonl
python -u data/prepare.py --inp data/sessions_full.jsonl --out data/full
echo "end $(date)"
echo "If 'truncated sessions' above is not near 0, rerun prepare.py with a larger --seq_len"
echo "(training/train.py reads n_ctx from the array width, so the model will match automatically)."
