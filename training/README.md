# Training: the model under test

**Research question.** Does a transformer trained *only* on next-token prediction over
poker transcripts learn to represent its opponent's play style, without ever being told
it? The model is the test subject. The results come from probing its activations
(linear probes for style, accuracy vs. hands seen) and activation patching.

**Who did what**

- **William:** data pipeline (`data/simulate.py`, `data/prepare.py` at the repo root).
- **Gunner:** training specification (`training-specification.md`) and probing.
- **Muhammad:** training. That is the code, runs, checkpoints and results in this folder.

`training-specification.md` is the source of truth for training. Every interpretation of
it, every judgment call and every deviation is logged in [`DECISIONS.md`](DECISIONS.md)
(D01–D38). Read it before relying on any detail below.

---

## Contents

| Path | What it is |
|---|---|
| `train.py` | Builds the model, trains it, logs metrics, saves checkpoints. `load_checkpoint()` reloads any checkpoint. |
| `preflight.py` | The spec's 4 pre-training checks. |
| `baselines.py` | Frequency, legality-aware, style-told and model-based baselines; accuracy by hand number. |
| `jobs/*.sbatch` | Slurm scripts used on Brown's Oscar cluster. |
| `results/*.json` | Baseline and preflight results for every run. |
| `DECISIONS.md` | Decision log. |
| `training-specification.md` | Gunner's spec. |
| `requirements.txt` | Python packages for this folder. |
| `checkpoints/` | Not in git. Comes from the Google Drive zip (see [Checkpoints](#checkpoints)). |

All commands below run **from the repo root**. Default paths do not depend on the
working directory: `--data` defaults to `data/full` at the repo root (`data/small` for
preflight), and `--out_dir` defaults to `training/checkpoints`.

---

## How the training was carried out

### Data (William's pipeline, unchanged)

- 100,000 simulated heads-up limit hold'em sessions (RLCard). One session is 50 hands
  against **one** opponent.
- Opponent styles: tight, loose, bluffer or random, with jittered knobs (`play_rate`,
  `aggression`, `bluff_rate`).
- Split by session, 90/10: 90,000 train and 10,000 val sessions.
- Vocab of 68 tokens: 8 special, 8 actions (`H:`/`O:` × fold/check/call/raise), 52 cards.
  It is read from `data/full/vocab.json`.
- One row is one session: `(sessions, 1024)` uint16, right-padded with `0 = [PAD]`.
  Sessions are at most 953 tokens long (mean about 452), so **0 sessions are truncated**.
- Style labels and knobs are never given to the main model; they are for probing only.

### Model

- `transformer_lens.HookedTransformer` (TransformerLens **3.9.0**) built from a
  `HookedTransformerConfig`, randomly initialized, not `from_pretrained`.
- 6 layers, d_model 256, 8 heads × d_head 32, d_mlp 1024, GELU, LayerNorm, learned
  positional embeddings, causal attention.
- d_vocab 68, n_ctx 1024, no BOS token.
- **5,036,100 parameters.**

### Loss

- Our own next-token cross-entropy over the full 68-token softmax.
- It is averaged **only over positions whose target is one of the 8 action tokens**.
- Cards, `O:won`/`O:lost`, structure tokens and PAD are never targets, but they stay
  visible as inputs.
- With gradient accumulation, the loss is normalized by the total number of action
  targets across the 4 micro-batches (D13).

### Training settings

- AdamW: lr 3e-4 **constant** (no warmup or decay, D12), weight decay 0.01,
  betas 0.9/0.999.
- Batch 8 sessions × gradient accumulation 4 = 32 sessions per optimizer step.
- Gradient-norm clipping 1.0, fp32, 3 epochs.
- Steps: 2,813 per epoch, 8,439 in total. The last step of each epoch has 2 micro-batches
  instead of 4 (D14).
- Each batch is whole rows: rows are never concatenated or windowed. Data order is a
  per-epoch permutation from `np.random.default_rng([seed, epoch])`.
- Logged: train/val action loss, hero and opponent accuracy (restricted and full-vocab),
  LR and grad norm.
  - A fixed val subset (the first 1,000 rows) is evaluated frequently.
  - The full val set is evaluated at step 0 and at every epoch boundary.

### Runs

- **3 seeds (0, 1, 2)** on Oscar, one GPU each. Seeds 0 and 1 ran on NVIDIA L40S and
  seed 2 on an RTX A5000. Each took 33–52 minutes; the oracle took 12 minutes.
- All three finished with exit code 0, from a clean git tree.
- Each run directory has a `run_meta.json` with:
  - architecture, parameter count, vocab fingerprint and action-token ids;
  - seed, package versions, GPU, exact command, and git commit with dirty flag;
  - dataset sizes and sha256 fingerprints;
  - the checkpoint schedule and results.

### Checkpoint schedule (per seed, in `training/checkpoints/seed<N>/`)

| File | When |
|---|---|
| `step_000000.pt` … `step_008439.pt` | Weights only, at 0, 0.1, 0.3, 1, 3, 10, 30, 60 and 100 % of optimizer steps: steps 0, 8, 25, 84, 253, 844, 2532, 5063, 8439. Step 0 is before any update. |
| `epoch_1.pt`, `epoch_2.pt`, `epoch_3.pt` | Full resumable state: weights, optimizer, RNG states, step, metric history. |
| `metrics.jsonl` | Every logged metric. `kind` is `train`, `val_subset` or `val_full`. |
| `run_meta.json` | Run metadata (above). |

Every `.pt` file holds `config`, `state_dict`, `step` and `meta`, so it is self-describing.

### Preflight checks (`preflight.py`, all passed on GPU: `results/preflight_gpu.json`)

1. **Overfit:** the model overfits 4 sessions down to the achievable loss floor. The floor
   is above 0 because some sessions share identical prefixes (D26).
2. **Reload:** a saved checkpoint reloads with `load_checkpoint` and reproduces the logits.
3. **`run_with_cache`:** exposes the residual stream, per-head attention and MLP
   activations (81 hooks, plus per-head results).
4. **`run_with_hooks`:** patching an activation changes the logits downstream, and only at
   later positions (attention is causal). Ablating a head also changes the logits.

### Baselines (`baselines.py`; definitions in its docstring and in D09, D27, D28, D33, D35, D37)

All scores are for the opponent's next action on the 10,000 val sessions (969,972 targets).

**Frequency baselines**
- **majority:** always predict the most common opponent action.
- **freq_overall / freq_running:** action frequencies over the whole training set, or
  counted so far in the current session (smoothing α chosen on train).
- **freq_overall_legal / freq_running_legal:** the same, but per *situation* and only over
  legal actions. The situation is reconstructed from the tokens with RLCard's rules:
  facing a bet or not, and whether the raise cap is reached. Every action in the data is
  legal under this reconstruction.

**Model-based baselines**
- **current_hand_only:** the trained model sees only the current hand, with no session
  history. There are two variants:
  - `pos0`: the hand restarts at position 0;
  - `origpos`: the hand keeps its original positions.
- **init:** the step-0 (untrained) checkpoint.

**Ceilings, which use the true style for evaluation only**
- **oracle_style_freq_legal:** legality-aware frequencies per true style and situation.
- **oracle model:** a separate control model (`train.py --oracle_style`, D36). Each session
  starts with a token for its true style: `[STYLE:tight|loose|bluffer|random]`, ids 68–71,
  vocab 72, 5,038,152 parameters. Otherwise it uses the same architecture and settings,
  trained for **1 epoch** with seed 0. The main models never see style labels.

---

## Results

**Headline metric: restricted opponent accuracy** (D34). This is the argmax among the 4
`O:` action tokens. Full-vocab accuracy (argmax over all 68 tokens) is reported too.
`loss_r` is the cross-entropy after renormalizing over the 4 opponent actions.

### All seeds (full val set, final checkpoint `step_008439.pt`)

| | Seed 0 | Seed 1 | Seed 2 | Mean ± sd |
|---|---|---|---|---|
| Val action loss, step 0 → final | 4.30 → 0.9585 | 4.04 → 0.9588 | 4.79 → 0.9591 | 0.9588 ± 0.0003 |
| **Restricted opponent accuracy** | 0.6138 | 0.6131 | 0.6119 | **0.6129 ± 0.0009** |
| Restricted opponent loss (`loss_r`) | 0.7696 | 0.7693 | 0.7695 | 0.7695 |
| Full-vocab opponent accuracy | 0.5816 | 0.5835 | 0.5884 | 0.5845 |
| Restricted hero accuracy | 0.6189 | 0.6188 | 0.6188 | 0.6188 |
| Current hand only (`pos0`) | 0.5430 | 0.5438 | 0.5440 | 0.5436 |
| Untrained, step 0 | 0.252 | 0.311 | 0.285 | 0.283 |

The val loss was 0.963, 0.962 and 0.961 after epoch 1: almost all the learning happens in
the first epoch.

### Baselines and ceilings (restricted opponent accuracy / `loss_r`)

| Predictor | Accuracy | `loss_r` |
|---|---|---|
| majority | 0.3181 | — |
| freq_overall | 0.3181 | 1.362 |
| freq_running (α = 8) | 0.4163 | 1.244 |
| freq_overall_legal | 0.5130 | 0.951 |
| current_hand_only `pos0` (3-seed mean) | 0.5436 | 0.921 |
| freq_running_legal (α = 4), best without labels | 0.6032 | 0.809 |
| **main model (3-seed mean)** | **0.6129** | **0.770** |
| *ceiling:* oracle_style_freq_legal | 0.6118 | 0.782 |
| *ceiling:* oracle model (1 epoch) | 0.6183 | 0.758 |

The oracle model's full-vocab opponent accuracy is 0.5971. Its val action loss went from
4.366 to 0.9546 after its single epoch.

### Accuracy by hand number (restricted opponent accuracy)

| Hands into the session | 1 | 2–5 | 6–10 | 11–20 | 21–30 | 31–40 | 41–50 |
|---|---|---|---|---|---|---|---|
| main model (3-seed mean) | .537 | .588 | .606 | .617 | .618 | .618 | .620 |
| oracle model | .616 | .616 | .616 | .620 | .618 | .618 | .619 |
| oracle_style_freq_legal | .608 | .612 | .614 | .611 | .612 | .611 | .612 |
| freq_running_legal | .516 | .573 | .594 | .606 | .609 | .611 | .613 |
| freq_overall_legal (no history) | .517 | .513 | .512 | .515 | .512 | .513 | .512 |
| current hand only, `pos0` (seed 0) | .536 | .544 | .544 | .544 | .542 | .543 | .543 |
| current hand only, `origpos` (seed 0) | .536 | .533 | .536 | .531 | .510 | .507 | .483 |

### Highlights

- The three seeds agree closely: restricted opponent accuracy is 0.6129 ± 0.0009.
- At hand 1 the model knows nothing about the opponent and scores like the
  history-free baselines (.537). It improves as hands accumulate.
- **By hands 11–20 it matches the oracle model**, which is told the style from the start.
  All of its remaining gap to the oracle is in the first few hands. This is the pattern
  expected if the model infers the opponent from play.
- Session history is worth about 7 points: full context scores 0.613, the current hand
  alone 0.544.

### Caveats

- **Low ceiling.** Even knowing the true style only reaches about 0.618, because
  opponents act on hole cards nobody can see. From the best counting baseline to the
  oracle is only about 1.5 points, so no model could beat counting by a large margin here.
- **Counting nearly reaches the ceiling.** freq_running_legal gets to 0.613 by the last
  hands. Behavior alone therefore cannot separate "the model represents the opponent's
  style" from "the model tracks action counts". That is exactly the question for probing
  and patching.
- **The oracle model trained for 1 epoch**, the main models for 3. The main models barely
  improved after epoch 1 (above), so this should matter little.
- **The current-hand-only baseline choice is `PENDING` Gunner** (D09, D27, D37).
  - Only the `origpos` variant drops with hand number. Its late absolute positions with no
    history before them never occur in training.
  - The `pos0` variant and the history-free frequency baseline stay flat.
  - Recommendation: `pos0` as the primary baseline, `origpos` as a diagnostic.
- **Restricted vs full-vocab accuracy.** Full-vocab accuracy also scores whether the model
  guesses *which player* acts next. Right after `[HAND]` that is random, so full-vocab
  opponent and hero accuracies trade off across seeds and epochs (a see-saw).
  - Restricted accuracy removes that coin flip, so it is the headline (D34).
  - Full-vocab numbers are still in `metrics.jsonl` and `results/`.

---

## Checkpoints

Checkpoints and the val data are **not** in git (too large). They are in one zip on the
team's Google Drive; the link is in our team chat.

**Unzip it at the repo root.** It contains `training/checkpoints/...` and `data/full/...`,
so every file lands where the code expects it:

```
training/checkpoints/seed0/  seed1/  seed2/   9 weight + 3 epoch checkpoints each, metrics.jsonl, run_meta.json
training/checkpoints/oracle/seed0/            oracle control model (9 weight + 1 epoch checkpoint, metadata)
training/checkpoints/DRIVE_README.txt         list of contents
data/full/val_tokens.npy, val_styles.npy, val_knobs.npy, vocab.json
```

```bash
cd <repo root>
unzip /path/to/poker_probe_checkpoints_2026-11-21.zip
```

On Windows, use "Extract All" with the repo root as the destination. Remove the folder
name that Windows adds by default, or move `training/` and `data/` up afterwards.

Both folders are git-ignored (`training/.gitignore`, and `data/full/` in the root
`.gitignore`). The training tokens are not in the zip: regenerate them (below) if you need
to retrain or rerun `preflight.py` / `baselines.py`.

## Reproducing

```bash
pip install -r requirements.txt            # William's data pipeline (rlcard, numpy)
pip install -r training/requirements.txt   # torch, transformer_lens==3.9.0, ...
```

**1. Data.** The seeds are fixed, so everyone gets identical data.

```bash
python data/simulate.py --sessions 100000 --out data/sessions_full.jsonl   # ~9 min
python data/prepare.py --inp data/sessions_full.jsonl --out data/full      # prints "truncated sessions: 0"
python data/simulate.py --sessions 1000 --out data/sessions_small.jsonl    # small set for preflight
python data/prepare.py --inp data/sessions_small.jsonl --out data/small
```

The sha256 of `train_tokens.npy` / `val_tokens.npy` should match `data.train_tokens_sha256`
/ `data.val_tokens_sha256` in any `run_meta.json`. `train.py` refuses to resume on
different data.

**2. Preflight**

```bash
python training/preflight.py --data data/small
```

**3. Train each seed** (about an hour each on one modern GPU)

```bash
python training/train.py --seed 0              # -> training/checkpoints/seed0/
python training/train.py --seed 0 --resume     # continue from the latest epoch_E.pt
python training/train.py --seed 1
python training/train.py --seed 2
```

On Slurm (Oscar), the scripts load the module and activate `$VENV`, a Python environment
with `requirements.txt` installed. Edit the `module load` line for other clusters.

```bash
mkdir -p training/logs
export VENV=/path/to/your/venv
sbatch -J train-s0 training/jobs/train.sbatch 0    # auto-resumes if an epoch checkpoint exists
sbatch -J train-s1 --dependency=afterok:<seed0 jobid> --kill-on-invalid-dep=yes training/jobs/train.sbatch 1
sbatch -J train-s2 --dependency=afterok:<seed0 jobid> --kill-on-invalid-dep=yes training/jobs/train.sbatch 2
```

**4. Oracle control model** (1 epoch, seed 0)

```bash
python training/train.py --seed 0 --epochs 1 --oracle_style --out_dir training/checkpoints/oracle
# or: sbatch training/jobs/train_oracle.sbatch          -> training/checkpoints/oracle/seed0/
```

**5. Baselines.** Each command also computes the data baselines and the style-told
ceiling. They need the train and val tokens.

```bash
python training/baselines.py --out training/results/baselines_data.json
for s in 0 1 2; do
  python training/baselines.py --ckpt training/checkpoints/seed$s/step_008439.pt \
      --init_ckpt training/checkpoints/seed$s/step_000000.pt --out training/results/baselines_seed$s.json
done
python training/baselines.py --oracle_ckpt training/checkpoints/oracle/seed0/step_002813.pt \
    --out training/results/baselines_oracle.json
```

**6. Load a checkpoint** (e.g. for probing). Run this from the repo root; it needs only the
zip's files plus this folder.

```python
import sys; sys.path.insert(0, "training")
import numpy as np, torch
from train import load_checkpoint

model, ckpt = load_checkpoint("training/checkpoints/seed0/step_008439.pt")   # GPU if available
tokens = torch.from_numpy(np.load("data/full/val_tokens.npy")[:4].astype(np.int64))  # cast uint16 -> int64
logits, cache = model.run_with_cache(tokens.to(model.cfg.device))
resid = cache["blocks.3.hook_resid_post"]     # [4, 1024, 256]
styles = np.load("data/full/val_styles.npy")  # probing labels (index into vocab.json "styles")
```

- Any `step_*.pt` or `epoch_*.pt` file loads the same way.
- For the oracle model, first prepend the style token:
  `from train import add_style_tokens; add_style_tokens(arr, styles, 68)`.
- Rows are right-padded and attention is causal, so no attention mask is needed.
  Activations at PAD positions are meaningless; ignore them.

## Note on code versions

- The three main runs recorded local commit **`e08982f`**, and the oracle run recorded
  local commit **`aa1eae2`**. Neither commit is on GitHub.
- The code here trains the main model exactly as `e08982f` did. Since that commit it
  gained only:
  - the restricted-accuracy metric;
  - the `--oracle_style` flag;
  - the legality-aware and style-told baselines;
  - the folder move into `training/`, which changed only paths and default locations
    (D38).
- Rerunning a seed with this code performs the same training computation. Results should match up to GPU floating-point nondeterminism.
