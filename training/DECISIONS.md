# Decision log

A record of how the project actually ran: every judgment call, interpretation of
`training-specification.md` ("the spec") or the proposal, and every deviation from them.
**Rule:** any change that involves a judgment call adds an entry here in the same commit
(see the project notes). Never deviate from the spec without logging it.

Each entry: **Date** · **Decision** · **Alternatives** · **Why** · **Decided by**
(Muhammad, Gunner, or William) · **Deviation** (from the spec or the original
proposal) · **Probing impact** (could it affect probing results?).
Status tags: `PENDING` = waiting for a teammate's confirmation.

---

## Backfilled (before 2026-11-21)

### D01. Train our own small model from scratch, not fine-tune TexasHoldEm-Llama-3.2-1B
- **Date:** after the 2026-10-28 proposal (exact date not recorded)
- **Decision:** randomly initialized small transformer trained only on our simulated transcripts.
- **Alternatives:** fine-tune TexasHoldEm-Llama-3.2-1B (the proposal's plan).
- **Why:** no outside pretraining that could confound whether style representations emerge from next-action prediction.
- **Decided by:** Gunner
- **Deviation:** yes, from the proposal (now required by the spec).
- **Probing impact:** yes, by design: it removes the pretraining confound.

### D02. Opponent styles: tight / loose / bluffer / random with jittered knobs, heads-up limit hold'em
- **Date:** ≤ 2026-11-11 (simulator commit)
- **Decision:** William's simulator (`data/simulate.py`, RLCard `limit-holdem`): 4 styles; each non-random session jitters `play_rate`, `aggression`, `bluff_rate` (std 0.08); hero is a fixed bot.
- **Alternatives:** the proposal's GTO / tight / aggressive bots.
- **Why:** simple, controllable knobs; no GTO solver needed.
- **Decided by:** William
- **Deviation:** yes, from the proposal. **The report must describe the actual styles.**
- **Probing impact:** yes: it defines the labels and knobs that probes target.

### D03. TransformerLens `HookedTransformer` 3.9.0, not `TransformerBridge` 4.0
- **Date:** ≤ 2026-11-20
- **Decision:** build `HookedTransformer` from `HookedTransformerConfig`; pin `transformer_lens==3.9.0`.
- **Alternatives:** TransformerLens 4.0 `TransformerBridge`.
- **Why:** stable hooks for probing and patching, no conversion step. 4.0 removed `HookedTransformer`, hence the pin.
- **Decided by:** Gunner (in the spec)
- **Deviation:** no
- **Probing impact:** no (it fixes the hook names probing code will use).

### D04. Separate venv `.venv_fp` (Oscar)
- **Date:** ≤ 2026-11-20
- **Decision:** project env `.venv_fp` (Python 3.11): torch 2.10.0 with the CUDA 12.8 wheel, transformer_lens 3.9.0, rlcard 1.2.0. torch left unpinned in `requirements.txt` for teammates.
- **Alternatives:** install into the course env `.venv311`.
- **Why:** installing into `.venv311` would upgrade numpy and protobuf and break TensorFlow 2.15 for the homeworks. CUDA 12.8 build for compatibility with Oscar's driver (torch ≥ 2.11 is CUDA 13).
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D05. Keep `seq_len` = `n_ctx` = 1024
- **Date:** 2026-11-20
- **Decision:** keep 1024 positions.
- **Alternatives:** regenerate with a longer `--seq_len`.
- **Why:** 0 truncated sessions in the full dataset (longest is 953 tokens).
- **Decided by:** Muhammad
- **Deviation:** no (matches the spec)
- **Probing impact:** no

### D06. Data and checkpoints are gitignored
- **Date:** ≤ 2026-11-20
- **Decision:** `data/*.jsonl`, `data/small/`, `data/full/`, `checkpoints/`, `logs/` are not committed. Teammates regenerate data from fixed seeds; `run_meta.json` records sha256 fingerprints of the token arrays so regenerated data can be checked.
- **Alternatives:** commit data and checkpoints (too large) or use Git LFS.
- **Why:** size.
- **Decided by:** William (data, original `.gitignore`); Muhammad (checkpoints, logs)
- **Deviation:** no
- **Probing impact:** no. **OPEN:** how checkpoints get shared with Gunner is still to be decided.

### D07. Trim trailing PAD columns per batch
- **Date:** 2026-11-20
- **Decision:** before each forward pass, drop right-end columns that are PAD in every row of the batch.
- **Alternatives:** always run at 1024 positions.
- **Why:** faster and less memory. With right padding and causal attention, real tokens never see later PADs. Verified that real-token logits are unchanged within 4e-6.
- **Decided by:** Muhammad
- **Deviation:** no (rows stay whole; no windows or concatenation)
- **Probing impact:** no (float noise only)

### D08. RLCard post-flop action order left as-is
- **Date:** ≤ 2026-11-20
- **Decision:** keep RLCard's post-flop acting order, which differs from standard heads-up poker. This is upstream behavior, not our code.
- **Alternatives:** patch the environment's acting order.
- **Why:** it is consistent across all sessions and styles; patching RLCard risks new bugs.
- **Decided by:** not recorded (flagged for Gunner)
- **Deviation:** no
- **Probing impact:** unlikely (the same for every style), but the report should mention it.

### D09. Baseline definitions `PENDING` Gunner
- **Date:** ≤ 2026-11-21
- **Decision:** "current-hand-only" = the trained model with its input cut to the current hand. "Action frequencies" = implemented both ways: overall frequencies across the training data, and a running per-opponent count within the session. Details in D27–D29.
- **Alternatives:** a separate model trained on single hands; a single frequency definition.
- **Why:** Gunner hasn't confirmed the definitions; the model-input version is Muhammad's preference.
- **Decided by:** Muhammad (pending Gunner)
- **Deviation:** interpretation of the spec's "current-hand-only history" and "action frequencies".
- **Probing impact:** no (baselines only), but it affects how results are framed.

### D10. Seed schedule: seed 0, then seeds 1 and 2 automatically
- **Date:** 2026-11-21
- **Decision:** submit seed 0; submit seeds 1 and 2 with a Slurm `afterok` dependency on seed 0, so they start together automatically if seed 0 succeeds (exit code 0). They are not held for a manual review of seed 0. Seeds 1 and 2 can be cancelled if seed 0's results look wrong.
- **Alternatives:** manual review of seed 0 before submitting 1 and 2 (the spec's "pilot seed 0, then seeds 1 and 2").
- **Why:** saves queue time; train.py exits non-zero on clear failures (D17), which blocks the dependency.
- **Decided by:** Muhammad
- **Deviation:** minor (automatic rather than reviewed hand-off).
- **Probing impact:** no

---

## 2026-11-21: implementing the spec (train.py, preflight.py, baselines.py)

### D11. The spec is the source of truth for training
- **Decision:** where `training-specification.md` and the project notes conflict, the spec wins; the notes now say so and point to it. Conflicting rules in the notes (loss on all non-PAD targets, periodic checkpoints) were rewritten to match the spec.
- **Alternatives:** —
- **Why:** Gunner defined training so the checkpoints suit probing.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D12. Constant learning rate 3e-4, no warmup or decay
- **Decision:** constant LR 3e-4 for all 3 epochs. The old train.py's warmup + cosine schedule was removed.
- **Alternatives:** linear warmup + cosine decay (old train.py); warmup only.
- **Why:** interpretation of the spec, which gives only "3e-4" and no schedule. To be revisited if seed 0's loss is unstable.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec)
- **Probing impact:** possibly: the LR schedule affects training dynamics and the late checkpoints.

### D13. Gradient-accumulation loss normalization
- **Decision:** each optimizer step's loss = (sum of action-token cross-entropies over all 4 micro-batches) / (total action-token targets across those 4 micro-batches). Every action counts equally, however many each micro-batch contains, which matches a single batch of 32.
- **Alternatives:** average of the 4 per-micro-batch means (weights actions in short micro-batches more).
- **Why:** correct per-action weighting.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec)
- **Probing impact:** negligible

### D14. Last partial accumulation group of each epoch is kept
- **Decision:** 90,000 sessions / 8 = 11,250 micro-batches per epoch; 11,250 / 4 = 2,812.5. The last optimizer step of each epoch uses the remaining 2 micro-batches (16 sessions), normalized by its own action count (D13). So **2,813 optimizer steps per epoch, 8,439 total** for 3 epochs.
- **Alternatives:** drop the partial group (2,812 steps per epoch; 16 sessions skipped per epoch).
- **Why:** every training session is seen exactly once per epoch.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec)
- **Probing impact:** no

### D15. Checkpoint fractions are measured in optimizer steps
- **Decision:** checkpoint step = round-half-up(fraction × total optimizer steps); step 0 = before any update. For 8,439 steps: 0% → 0, 0.1% → 8, 0.3% → 25, 1% → 84, 3% → 253, 10% → 844, 30% → 2,532, 60% → 5,063, 100% → 8,439. If two fractions round to the same step (only in tiny test runs), one file is saved and both fractions are recorded. These are weights-only files (`step_XXXXXX.pt`). Epoch boundaries (2,813 / 5,626 / 8,439) also get full resumable `epoch_E.pt` files.
- **Alternatives:** fractions of micro-batches, of sessions, or of tokens seen.
- **Why:** optimizer steps are the unit in which weights change; identical across seeds.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec)
- **Probing impact:** defines which training moments can be probed.

### D16. Validation: fixed subset for frequent logs, full set at epoch boundaries
- **Decision:** val metrics on a fixed subset (the first 1,000 val sessions; the same for every seed, randomly ordered by `prepare.py`) every 250 optimizer steps and at every fraction checkpoint. Full val set (10,000 sessions) at step 0 and at each epoch boundary. All metrics go to `metrics.jsonl`.
- **Alternatives:** full val at every eval (slow); a random subset each time (noisy, not comparable).
- **Why:** cheap and frequent, and comparable across steps and seeds.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D17. Exit with an error when training clearly failed
- **Decision:** train.py exits non-zero if (a) any optimizer step has a non-finite loss or gradient norm (it stops immediately), or (b) the final full-val action loss is not below the step-0 full-val action loss. `run_meta.json` records the failure. A non-zero exit blocks seeds 1 and 2 (`afterok`).
- **Alternatives:** always exit 0 and review by hand; stricter thresholds (e.g. must beat a baseline).
- **Why:** catches broken runs without second-guessing borderline ones.
- **Decided by:** Muhammad
- **Deviation:** no (addition)
- **Probing impact:** no

### D18. Loss and accuracy definitions
- **Decision:** loss = cross-entropy over the full 68-token softmax, averaged over action-token targets only. Accuracy = argmax over the full vocab == target. Hero and opponent accuracy are split by the target's owner. Hero and opponent losses are also logged separately (extra). baselines.py also reports "restricted" accuracy and loss (argmax / softmax over only the 4 actions of the target's player), which is comparable to frequency baselines that only predict actions.
- **Alternatives:** restrict the softmax to the 8 action tokens during training.
- **Why:** the spec says "next-token cross-entropy"; full-vocab is the standard reading. Restricted numbers matter mainly for the untrained step-0 model, whose full-vocab argmax is usually a non-action token.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec)
- **Probing impact:** no for probing; it affects how step-0 accuracy reads (report restricted accuracy there).

### D19. AdamW details
- **Decision:** `torch.optim.AdamW` with default betas (0.9, 0.999) and eps 1e-8. Weight decay 0.01 applies to all parameters (embeddings, LayerNorm, biases included).
- **Alternatives:** exclude biases, LayerNorm and embeddings from decay (common in GPT training).
- **Why:** the spec gives no exclusions; this matches the TransformerLens Grokking demo.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec)
- **Probing impact:** negligible at wd 0.01

### D20. fp32 training, no mixed precision
- **Decision:** plain fp32 (no autocast, no gradient scaler; the epoch checkpoints store `scaler: None`). TF32 matmuls left at the PyTorch default (off). No `torch.compile`, folding, pruning or quantization.
- **Alternatives:** bf16/fp16 autocast with a GradScaler (faster).
- **Why:** simplest, and it gives exactly the plain weights probing loads. The measured fp32 speed fits the job time limit (see the job notes).
- **Decided by:** Muhammad
- **Deviation:** no (the spec only requires saving scaler state "if used")
- **Probing impact:** no

### D21. Data order and seeding
- **Decision:** each epoch visits every training session once, in the order `numpy.random.default_rng([seed, epoch]).permutation(n)`. Model init uses TransformerLens's default (`init_mode="gpt2"`), seeded through `HookedTransformerConfig(seed=seed)`, which seeds python, numpy and torch. RNG states (python, numpy, torch CPU, CUDA) are still saved in epoch checkpoints and restored on resume.
- **Alternatives:** one continuous torch generator (its state would need to be saved mid-epoch).
- **Why:** the data order is a pure function of (seed, epoch), so resuming reproduces the same order. Smoke test on an A40 (spec architecture, data/small): a run stopped after epoch 1 and resumed was bit-identical to an uninterrupted run (weight diff 0.0 at every checkpoint, identical val losses). On CPU the two matched to ~1e-7 in val loss; that small difference is multithreaded float noise and also appears before the stop.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D22. Model config details
- **Decision:** `positional_embedding_type="standard"` (learned), `attention_dir="causal"`, `act_fn="gelu"`, `normalization_type="LN"`, `default_prepend_bos=False`, `d_vocab=68` from `vocab.json` (asserted), `n_ctx=1024`. Tokens are fed as int64 tensors, so no tokenizer or BOS is ever involved. No attention mask (right padding + causal).
- **Alternatives:** —
- **Why:** the spec's architecture, stated explicitly.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D23. Logged gradient norm = pre-clipping norm
- **Decision:** the logged gradient norm is the total L2 norm returned by `clip_grad_norm_` (before clipping), as mean and max over each 10-step log window.
- **Alternatives:** post-clip norm (always ≤ 1, uninformative).
- **Why:** shows instability and how often clipping is active.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D24. Checkpoint files and resume safety
- **Decision:** one directory per seed (`checkpoints/seed<N>/`). Every checkpoint embeds its config plus core metadata (seed, parameter count, vocab sha256, action-token ids), so each file is self-describing. All epoch checkpoints are kept, not only the latest. Files are written atomically (temp file + rename). `--resume` refuses if the config, run-defining args or training-data hash differ. A fresh start refuses if epoch checkpoints already exist. `--stop_after_epoch` exists only for testing resume. The eval batch size (16) has no effect on results.
- **Alternatives:** keep only the latest epoch checkpoint (saves about 40 MB per epoch).
- **Why:** the spec wants resumable epoch checkpoints; keeping all of them also gives full optimizer states at each epoch for later analysis.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D25. Recorded git state
- **Decision:** `run_meta.json` records the commit and a dirty flag. Untracked, non-ignored files count as dirty, and the dirty paths are listed. Each launch (including resumes) records its own command, git state, packages, host and Slurm job id. Training is submitted only from a committed state.
- **Alternatives:** ignore untracked files.
- **Why:** stricter, so the recorded commit is meaningful.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D26. Preflight overfit criterion is relative to the achievable floor
- **Decision:** check 1 passes when the action loss is within 0.02 nats of the minimum achievable loss and accuracy equals the best achievable, both computed exactly from the sample.
- **Alternatives:** require loss ≈ 0 and accuracy = 100%.
- **Why:** different sessions share identical prefixes (always the first `[HAND]`, sometimes whole opening hands) followed by different actions, so a perfect score is impossible.
- **Decided by:** Muhammad
- **Deviation:** no (interpretation of the spec's "can be overfit")
- **Probing impact:** no

### D27. Current-hand-only baseline: two position variants `PENDING` Gunner
- **Decision:** each val hand (from its `[HAND]` up to the next `[HAND]`) is fed to the trained model alone. `origpos` keeps the hand's original absolute positions (positional embeddings replaced through a `hook_pos_embed` hook), so only the history is removed. `pos0` restarts the hand at position 0. Both are reported.
- **Alternatives:** only one variant; a separately trained current-hand model.
- **Why:** each variant has a confound: `origpos` puts a late position with no history, `pos0` makes every hand look like hand 1. Reporting both brackets the effect of history.
- **Decided by:** Muhammad
- **Deviation:** interpretation of the spec
- **Probing impact:** no (it frames the "does the model use history" comparison).

### D28. Running per-opponent frequency baseline `PENDING` Gunner
- **Decision:** p(a) = (this opponent's earlier count of a in the session + α·p_overall(a)) / (n_earlier + α). Earlier actions include earlier streets of the current hand. α is picked from {0.25 … 32} by lowest loss on the training split. "Overall frequencies" predicts the training-set distribution (same accuracy as majority, plus a loss). Neither uses legality of actions; legality-aware versions were added later (D33).
- **Alternatives:** no prior (undefined before the first action); counts only from completed hands; legality-aware frequencies.
- **Why:** the simplest well-defined version of each candidate definition.
- **Decided by:** Muhammad
- **Deviation:** interpretation of the spec
- **Probing impact:** no

### D29. Hand-number breakdown
- **Decision:** an action target's hand number = the number of `[HAND]` tokens up to and including the predicting position (1–50). Reported for every baseline and model evaluation.
- **Alternatives:** —
- **Why:** shows whether accuracy grows with the hands seen against this opponent.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

### D30. Compute platform and job layout
- **Decision:** train on Oscar GPUs (Slurm `gpu` partition) instead of Colab. `jobs/train.sbatch <seed>` auto-resumes when an epoch checkpoint exists. Seeds 1 and 2 use `--dependency=afterok:<seed0>` with `--kill-on-invalid-dep=yes`, so they are removed if seed 0 fails rather than sitting in the queue.
- **Alternatives:** Colab (the spec's assumption for batch size).
- **Why:** available compute; the physical batch of 8 fits.
- **Decided by:** Muhammad
- **Deviation:** no (the spec mentions Colab only as a memory assumption)
- **Probing impact:** no

### D31. Small result files are committed
- **Decision:** `results/` (preflight and baseline JSON) is committed; checkpoints and logs are not.
- **Alternatives:** gitignore results too.
- **Why:** small, and teammates need the numbers.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no

---

## 2026-11-21: git history

### D32. Training-run commit e08982f rewritten to 21eb224 (identical code)
- **Date:** 2026-11-21
- **Decision:** the training runs for seeds 0, 1 and 2 (Slurm jobs 9463643, 9463644, 9463645) record commit `e08982f4164bbdb5af26dc46354a9d5851760caf` in `run_meta.json`. Before the first push, that commit was rewritten to `21eb22458ee200d34394a6aa10dec80ae66614c2` (message and author changed only). The tree is identical (`0f491ec7c76ae12ec3e36ccaeb2f32a927ac7fba`), so the code the runs used is exactly the code in 21eb224. The original commit is kept in a local-only tag `run-e08982f` (never pushed); `git diff run-e08982f 21eb224` is empty.
- **Alternatives:** push the original commit unchanged; rewrite before training (that would have delayed the runs).
- **Why:** project policy on commit authorship.
- **Decided by:** Muhammad
- **Deviation:** no
- **Probing impact:** no (identical tree). When matching a checkpoint to code, treat e08982f and 21eb224 as the same.

---

## 2026-11-21: legality-aware baselines

### D33. Legality-aware frequency baselines `PENDING` Gunner
- **Date:** 2026-11-21
- **Decision:** added `freq_overall_legal` and `freq_running_legal` to `baselines.py`, keeping the original versions. The opponent's situation at each action is reconstructed from the tokens with RLCard's limit hold'em rules (`round.get_legal_actions`):
  - facing a bet = the previous token is a raise, or `[HAND]` (preflop first actor faces the big blind);
  - can raise = fewer than 4 raises so far in the street (blinds don't count);
  - legal set = fold always; check iff not facing; call iff facing; raise iff can raise.

  situation = (facing, can raise). `freq_overall_legal` = p(action | situation) from the training data (+0.5 pseudo-count per legal action); its argmax is also the legal-majority baseline. `freq_running_legal` = this opponent's earlier counts in the same situation within the session, with prior α·p(action | situation) and α chosen on the training split. Both put zero probability on illegal actions.
- **Fix (2026-11-21):** the situation names in `train_opp_freqs_by_situation` were first printed in the wrong order; only those labels were wrong, not the baseline numbers. Fixed in code and relabelled in the already-written JSON files.
- **Validation:** on data/small, 0 of ~194k real hero and opponent actions fall outside the reconstructed legal set. `baselines.py` reports this count for every run (`illegal_actions_under_reconstruction`); the vectorized running counts were checked against a brute-force loop.
- **Alternatives:** condition on more context (street, previous action, board), which would turn it into a richer context model; mask the unconditioned running counts to the legal set instead of conditioning the counts on the situation.
- **Why:** the original frequency baselines also lose points by predicting illegal actions, which the model can rule out from context. This isolates "knows the opponent's tendencies" from "knows the rules". Requested by Muhammad; conditioning on (facing, can raise) follows his example "facing a bet or not".
- **Decided by:** Muhammad
- **Deviation:** interpretation of the spec's "action frequencies" (addition)
- **Probing impact:** no for probing. It raises the bar the model is compared against: on data/small, the legal running count reaches 0.596 opponent accuracy vs 0.410 for the original running count.

---

## 2026-11-21: metrics, ceilings and the current-hand-only check

### D34. Restricted accuracy is the headline metric
- **Date:** 2026-11-21
- **Decision:** headline = **restricted accuracy**: argmax among the 4 action tokens of the target's player (for opponent targets: O:fold / check / call / raise). Full-vocab accuracy (argmax over all 68 tokens) is still reported everywhere. `train.py` now also logs `hero_acc_r` / `opp_acc_r` (training itself is unchanged; a tiny CPU run reproduced earlier losses exactly). `baselines.py` prints restricted accuracy and loss first.
- **Alternatives:** keep full-vocab accuracy as the headline (D18).
- **Why:** who acts first after each `[HAND]` is random (blinds are assigned at random), so full-vocab accuracy partly scores an unpredictable "hero or opponent?" coin flip. That also causes the hero/opponent accuracy see-saw across epochs. Restricted accuracy is comparable to the frequency baselines, which only predict actions.
- **Decided by:** Muhammad
- **Deviation:** no (the spec doesn't define accuracy; refines D18)
- **Probing impact:** no (reporting only). Seeds 0–2 have only full-vocab accuracy in `metrics.jsonl`; restricted numbers for them come from `baselines.py`.

### D35. Cheap ceiling: style-told legal frequency counter
- **Date:** 2026-11-21
- **Decision:** `oracle_style_freq_legal` in `baselines.py`: p(action | TRUE style, situation) from the training data (+0.5 pseudo-count per legal action), legal actions only, evaluated on val. Style labels are used for evaluation only.
- **Alternatives:** condition on the true knobs (continuous; needs binning or the simulator's policy); combine style with the running count.
- **Why:** shows how much knowing the style buys a frequency counter. It is not a strict ceiling: it ignores each session's knob jitter (which a running count can pick up) and everything about the board and hand.
- **Decided by:** Muhammad
- **Deviation:** addition (an evaluation-only use of labels; the model never sees them)
- **Probing impact:** no

### D36. Oracle control model: true style as the first token
- **Date:** 2026-11-21
- **Decision:** a separate control model (the main models are untouched): `train.py --oracle_style` prepends one of 4 new tokens `[STYLE:tight|loose|bluffer|random]` (ids 68–71, vocab 72) to every session. Everything else matches the main model: architecture, action-only loss, AdamW 3e-4 constant, wd 0.01, batch 8 × 4, clip 1.0, fp32. It trains for **1 epoch with seed 0** (2,813 steps) into `checkpoints/oracle_style/seed0/` and is evaluated with the same restricted metrics and hand-number table (`baselines.py --oracle_ckpt`). Style labels are loaded only under this flag. The style token is never a target (it is at position 0).
- **Alternatives:** 3 epochs like the main model (more compute; the main model plateaued after about 1 epoch); give the knobs instead of the style.
- **Why:** it was in the original plan. It estimates how well this architecture predicts actions when told the style from hand 1, which bounds how much the main model could gain from inferring the style.
- **Decided by:** Muhammad
- **Deviation:** addition. It intentionally breaks the "never give style labels" rule, which applies to the main model only.
- **Probing impact:** no for the main model. It is a reference for interpreting it; note that the oracle gets 1 epoch vs 3.

### D37. Current-hand-only: why the original-positions variant drops with hand number
- **Date:** 2026-11-21
- **Finding:** seed 0, restricted accuracy by hand number:

  | hands | 1 | 2–5 | 6–10 | 11–20 | 21–30 | 31–40 | 41–50 |
  |---|---|---|---|---|---|---|---|
  | legal freq, no history | .517 | .513 | .512 | .515 | .512 | .513 | .512 |
  | current hand, restarted at position 0 | .536 | .544 | .544 | .544 | .542 | .543 | .543 |
  | current hand, original positions | .536 | .533 | .536 | .531 | .510 | .507 | .483 |
  | full context | .536 | .587 | .607 | .619 | .618 | .619 | .621 |

  The history-free baseline and the restart-at-0 variant are flat, as expected for freshly dealt hands. Only the original-positions variant drops. The two current-hand variants differ only in their positional embeddings, so the drop comes from position, not from the hands. The original-positions variant shows the model a late absolute position (up to ~950) with no history before it, a combination never seen in training, where late positions always follow hundreds of history tokens. The mismatch grows with hand number.
- **Decision:** recommend `current_hand_only_pos0` as the primary current-hand-only baseline, with `_origpos` kept as a diagnostic. `PENDING` Gunner (D09, D27).
- **Decided by:** Muhammad (pending Gunner)
- **Deviation:** no
- **Probing impact:** possibly relevant: it suggests late positional embeddings carry history-dependent assumptions. Worth keeping in mind for patching experiments that move activations across positions.


### D38. Release layout: training code moved into `training/` for GitHub
- **Date:** 2026-11-21
- **Decision:** the training code, job scripts, spec, results (JSON) and this log are published in a `training/` folder on top of William's commit 3b25fc6. Data stays in `data/` at the repo root. Commands run from the repo root (`python training/train.py ...`). Default paths no longer depend on the working directory: `--data` defaults to `<repo>/data/full` (`data/small` for preflight), `--out_dir` to `training/checkpoints`, logs go to `training/logs/`, and the oracle control model's run dir is `training/checkpoints/oracle/seed0/` (was `checkpoints/oracle_style/seed0/`). Job scripts accept a `VENV` variable for the Python environment. Checkpoints and the val data are shared as a zip on the team Google Drive, laid out to unzip at the repo root. Earlier entries in this log use the old paths (`train.py`, `checkpoints/...`, `results/...`), which are now under `training/`.
- **Alternatives:** keep the files at the repo root (mixes with the data pipeline's files); a separate repo.
- **Why:** keeps the training work in one folder without touching William's files.
- **Decided by:** Muhammad
- **Deviation:** no. Training logic is unchanged. The runs recorded local commit e08982f, which is not on GitHub. Since then the code gained only the restricted-accuracy metric (D34), the oracle control flag (D36) and the legality-aware and style-told baselines (D33, D35), none of which changes how the main model trains, plus the path changes listed here.
- **Probing impact:** no.
