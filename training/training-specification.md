# Poker Transformer Training Specification

## Scientific objective

Train a transformer only to predict poker actions, then test whether opponent behavior becomes internally represented without style supervision.

## Required choices and rationale

| Choice | Requirement | Rationale |
|---|---|---|
| Model | Randomly initialized TransformerLens `HookedTransformer` 3.9.0 | Training from scratch as opposed to using a prebuilt model prevents any exterior training confounding representation emergence. TransformerLens exposes stable hooks for later observation (as opposed to using TransformerBridge 4.0). |
| Architecture (open to changing) | 6 blocks; residual width 256; 8 heads of width 32; MLP width 1,024; causal attention; learned positional embeddings; GELU; LayerNorm | I think 5M parameters will be small enough for repeated causal analysis and multiple seeds but also provides several layers and heads to localize representations. Can be changed if this turns out not to be performant. |
| Sequence and vocabulary | Maximum 1,024 positions; use the ordered 68-token vocabulary from `vocab.json`; insert no language-model tokenizer or beginning token | This matches the prepared session arrays and preserves poker events as atomic tokens. |
| Session boundary | Each array row remains one complete session; never concatenate rows or mix windows across sessions | Opponent identity is persistent within a row and changes between rows. Crossing rows would create false histories. |
| Inputs | Cards, actions, outcomes, and structural markers remain visible | Cards and game state are needed to distinguish style from rational responses to hand strength. |
| Loss | Next-token cross-entropy only where the target is `H:` or `O:` followed by `fold`, `check`, `call`, or `raise` | Actions express opponent behavior. Cards are randomly dealt, and predicting them would add irreducible noise. `O:won` and `O:lost` are outcomes, not actions. |
| Hidden labels | Never load style labels or behavioral knobs during training | Otherwise opponent recognition would be explicitly supervised rather than emergent. |

## Pilot training settings and rationale

| Setting | Pilot value | Rationale |
|---|---:|---|
| Optimizer | AdamW | Standard stable optimizer for decoder transformers. |
| Learning rate | `3e-4` | Conventional starting point for a small randomly initialized transformer; it must be validated by stable pilot learning. |
| Weight decay | `0.01` | Mild regularization without making regularization a major experimental variable. |
| Physical batch | 8 sessions | Expected to fit a Colab GPU at 1,024 positions. |
| Gradient accumulation | 4 | Gives an effective batch of 32 while preserving complete sessions. |
| Gradient clipping | `1.0` | Limits unstable updates during the pilot. |
| Training budget | Initially 3 epochs | A bounded pilot budget; retain it only if validation behavior supports it. |
| Seeds | Pilot seed 0, then seeds 1 and 2 | One seed validates the pipeline; three full runs provide a minimum reproducibility check within course-scale compute. Report every seed. |

None of these effect the probing process though, so they can be changed.

## Checkpoints and rationale

Save weights at training fractions `0%, 0.1%, 0.3%, 1%, 3%, 10%, 30%, 60%, 100%` and at each epoch boundary. We want to save these in case we want to do analysis to see at what part of the training process the representations emmerge.

- Step zero is the untrained control.
- Logarithmic early spacing captures rapid representation formation without excessive storage.
- Identical fractions across seeds make training trajectories comparable.

Weights-only checkpoints are sufficient for probing. Epoch checkpoints must also contain optimizer state, gradient-scaler state, random-number states, epoch, optimizer step, and metric history so interrupted runs can resume.

Each run must record architecture dimensions, actual parameter count, ordered-vocabulary fingerprint, action-token IDs, seed, package versions, Git commit, exact command, dataset sizes, epoch, and optimizer step. Save ordinary unquantized model weights; do not prune, fold, or compile the probing checkpoint.

## Validation and handoff

Before full training, verify:

1. a tiny sample can be overfit;
2. checkpoint reload reproduces logits;
3. `run_with_cache` exposes residual, attention-head, and MLP activations;
4. `run_with_hooks` can alter an activation and downstream logits.

Log training and validation action loss, hero-action accuracy, opponent-action accuracy, learning rate, and gradient norm. Compare opponent prediction against the initialization checkpoint, majority action, action frequencies, and current-hand-only history.

Provide the model-construction and loading code, initialization checkpoint, intermediate weights, resumable epoch checkpoints, metric logs, evaluation results, exact commands, seeds, package versions, and Git commit.

## References

- [TransformerLens 3.9](https://github.com/TransformerLensOrg/TransformerLens/tree/v3.9.0)
- [Grokking training demo](https://github.com/TransformerLensOrg/TransformerLens/blob/v3.9.0/demos/Grokking_Demo.ipynb)
- [Othello-GPT latent game-state example](https://arxiv.org/abs/2210.13382)
- [Training-dynamics checkpoint analysis](https://arxiv.org/abs/2301.05217)
