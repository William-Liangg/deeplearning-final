# Training

This folder trains the model that the probing work studies: a small transformer that only
ever learns to predict poker actions. It never sees opponent style labels. The question is
whether it ends up representing the opponent's style on its own. The data pipeline is in
`data/` at the repo root.

For a plain-language, step-by-step account of how the training was carried out, including
changes from the original plan, see [`PROCESS.md`](PROCESS.md).

| File | What it is |
|---|---|
| `train.py` | Model, training loop, `load_checkpoint()` |
| `preflight.py` | The spec's 4 pre-training checks |
| `baselines.py` | All baselines and accuracy by hand number (definitions in its docstring) |
| `jobs/` | Slurm scripts used on Oscar (set `$VENV` to your environment) |
| `results/` | Baseline and preflight results (JSON) |
| `PROCESS.md` | How the training was carried out, step by step |
| `training-specification.md` | Training specification |

All commands run from the repo root.

---

## Quick start for probing

1. **Get the checkpoints.** Download the zip from the team Google Drive (link in the team
   chat) and unzip it **at the repo root**. It adds `training/checkpoints/` and the val
   data in `data/full/`. Both are git-ignored.
2. **Install:** `pip install -r requirements.txt -r training/requirements.txt`
   (TransformerLens must be 3.9.0).
3. **Load a model and cache activations:**

```python
import sys; sys.path.insert(0, "training")
import numpy as np, torch
from train import load_checkpoint

model, ckpt = load_checkpoint("training/checkpoints/seed0/step_008439.pt")
tokens = torch.from_numpy(np.load("data/full/val_tokens.npy")[:4].astype(np.int64))
logits, cache = model.run_with_cache(tokens.to(model.cfg.device))
resid = cache["blocks.3.hook_resid_post"]     # [4, 1024, 256]
styles = np.load("data/full/val_styles.npy")  # labels; names are in vocab.json "styles"
```

Notes:

- Tokens are stored as uint16, so cast them to int64. Rows are right-padded with 0
  (`[PAD]`): ignore activations at PAD positions. Attention is causal, so no mask is needed.
- Each `[HAND]` token starts a new hand, which gives you "hands seen" for each position.
- `data/full/val_knobs.npy` holds each opponent's `play_rate`, `aggression` and
  `bluff_rate` (NaN for random opponents).
- **Controls:**
  - `step_000000.pt` in each seed folder is the untrained model.
  - The oracle model is `training/checkpoints/oracle/seed0/step_002813.pt`. It expects the
    style token at the start of each session:
    `from train import add_style_tokens; add_style_tokens(arr, styles, 68)`.

## Checkpoints

Each seed has a folder, `training/checkpoints/seed0/`, `seed1/` and `seed2/`, containing:

| File | Contents |
|---|---|
| `step_000000.pt` … `step_008439.pt` | Weights at 0, 0.1, 0.3, 1, 3, 10, 30, 60 and 100 % of training: steps 0, 8, 25, 84, 253, 844, 2532, 5063, 8439. Step 0 is untrained. |
| `epoch_1.pt`, `epoch_2.pt`, `epoch_3.pt` | Full resumable state (optimizer, RNG, metric history) |
| `metrics.jsonl` | Every logged metric |
| `run_meta.json` | Config, parameter count, seed, package versions, git commit, exact command, data fingerprints |

Use the final checkpoint, `step_008439.pt`, unless you are studying how representations
form during training.

## What was trained

| | |
|---|---|
| **Data** | 100,000 heads-up limit hold'em sessions (RLCard). Each session is 50 hands against one opponent: tight, loose, bluffer or random, with slightly randomized knobs. 90/10 train/val split by session. 68-token vocab. The longest session is 953 tokens, so none are truncated at 1024. |
| **Model** | TransformerLens 3.9.0 `HookedTransformer`, randomly initialized. 6 layers, d_model 256, 8 heads × 32, d_mlp 1024, GELU, LayerNorm, learned positions, n_ctx 1024, no BOS token. 5,036,100 parameters. |
| **Objective** | Next-token cross-entropy, counted only where the target is one of the 8 action tokens (`H:`/`O:` fold, check, call, raise). Cards and outcomes stay visible as input. Style labels are never used. |
| **Training** | AdamW, constant lr 3e-4, weight decay 0.01. 8 sessions × 4 accumulation steps = 32 sessions per step. Gradient clipping 1.0, fp32, 3 epochs (8,439 steps). Batches are whole sessions, never mixed. |
| **Runs** | Seeds 0, 1 and 2 on Oscar GPUs, 33–52 minutes each. All 4 preflight checks passed (`results/preflight_gpu.json`). |
| **Oracle control** | Same model and settings, but each session starts with a token giving the true style (vocab 72). Trained for 1 epoch with seed 0. Used only as a ceiling. |

## Results

**Headline metric:** restricted opponent accuracy. That is how often the model predicts the
opponent's next action correctly, choosing among the 4 opponent actions. It is measured on
the val set (969,972 predictions).

| Predictor | Accuracy |
|---|---|
| Always the most common action | 0.318 |
| Running count of this opponent's actions | 0.416 |
| Overall frequencies, legal actions only | 0.513 |
| Trained model, current hand only (`pos0`) | 0.544 |
| Running count, legal actions only (best without labels) | 0.603 |
| **Trained model, 3 seeds** | **0.6129 ± 0.0009** |
| Ceiling: frequencies given the true style | 0.612 |
| Ceiling: oracle model | 0.618 |

The untrained model scores 0.283 (3-seed mean).

**Accuracy by hand number:**

| Hand | 1 | 2–5 | 6–10 | 11–20 | 21–30 | 31–40 | 41–50 |
|---|---|---|---|---|---|---|---|
| Trained model (3 seeds) | .537 | .588 | .606 | .617 | .618 | .618 | .620 |
| Oracle model | .616 | .616 | .616 | .620 | .618 | .618 | .619 |
| Running count, legal only | .516 | .573 | .594 | .606 | .609 | .611 | .613 |
| Current hand only, `pos0` (seed 0) | .536 | .544 | .544 | .544 | .542 | .543 | .543 |

**Key findings**

- **The three seeds agree closely:** 0.6138, 0.6131 and 0.6119 (sd 0.0009).
- **The model uses the session history.** It starts at .537 on hand 1 and climbs to .620.
  The same model limited to the current hand stays at about .544.
- **By hands 11–20 it matches the oracle model,** which is told the style up front. Its
  only gap is in the first few hands, before it has seen the opponent play.

**Caveats**

- **The ceiling is low** (about .618), because opponents act on hole cards nobody can see.
  The whole range from the best counting rule to the oracle is about 1.5 points.
- **The legal running count also gets close** (.613 by the end). So behavior alone can't
  tell "represents style" apart from "tracks action counts". That is the question for
  probing and patching.
- **The oracle trained for 1 epoch** against 3 for the main models. The main models barely
  improved after epoch 1.
- **Current-hand-only baseline:** `pos0` (the hand restarted at position 0) is the
  recommended version, pending team confirmation. The other version (`origpos`) has a positional
  artifact and drops as hand number increases.
- **Full-vocab accuracy** (0.584) is lower than restricted accuracy because it also scores
  guessing who acts next, which is random at the start of each hand.

Full numbers are in `results/`.

## Reproducing

```bash
pip install -r requirements.txt -r training/requirements.txt

# Data. Seeds are fixed, so everyone gets identical data; the sha256 is in run_meta.json.
python data/simulate.py --sessions 100000 --out data/sessions_full.jsonl
python data/prepare.py --inp data/sessions_full.jsonl --out data/full
python data/simulate.py --sessions 1000 --out data/sessions_small.jsonl    # for preflight
python data/prepare.py --inp data/sessions_small.jsonl --out data/small

python training/preflight.py --data data/small
python training/train.py --seed 0          # repeat for seeds 1 and 2; --resume continues a run
python training/train.py --seed 0 --epochs 1 --oracle_style --out_dir training/checkpoints/oracle

python training/baselines.py --ckpt training/checkpoints/seed0/step_008439.pt \
    --init_ckpt training/checkpoints/seed0/step_000000.pt --out training/results/baselines_seed0.json
python training/baselines.py --oracle_ckpt training/checkpoints/oracle/seed0/step_002813.pt \
    --out training/results/baselines_oracle.json
```
