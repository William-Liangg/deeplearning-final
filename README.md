# Poker Opponent Probe

**Goal:** Does a transformer trained to predict the next poker action learn opponent play styles without being explicitly told them?

## 1. Setup

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

## 2. Generate the Dataset

Run these commands from the repo root. Seeds are fixed, so everyone generates identical data.

**Small dataset (~5 seconds):** For debugging and testing.

```bash
python data/simulate.py --sessions 1000 --out data/sessions_small.jsonl
python data/prepare.py --inp data/sessions_small.jsonl --out data/small
```

**Full dataset (~9 minutes):** For actual training.

```bash
python data/simulate.py --sessions 100000 --out data/sessions_full.jsonl
python data/prepare.py --inp data/sessions_full.jsonl --out data/full
```

The two scripts run in order:
1. `simulate.py` generates poker sessions (50 hands against one opponent per session).
2. `prepare.py` converts those sessions into tokenized NumPy arrays and splits them into training and validation sets.

## 3. Understanding the Data

After preparation, `data/small/` or `data/full/` contains:

| File | Shape | Purpose |
|---|---|---|
| `train_tokens.npy`, `val_tokens.npy` | `(sessions, 1024)` | Tokenized poker sessions used as model input |
| `train_styles.npy`, `val_styles.npy` | `(sessions,)` | Opponent style labels, **probing only** |
| `train_knobs.npy`, `val_knobs.npy` | `(sessions, 3)` | Opponent behavior parameters, **probing only** |
| `vocab.json` | — | Token vocabulary, style names, and knob names |

**Important details:**
- Each row represents one session of 50 hands against the same opponent.
- Each session is padded to 1024 tokens (`0 = [PAD]`). Ignore padding when computing loss.
- Train on complete sessions. **Never combine windows across rows**, since different rows represent different opponents.
- Style labels and knobs are only for probing, **not transformer training**.
- The three knobs are `play_rate`, `aggression`, and `bluff_rate`. Random bots have `NaN` knob values.

## 4. Understanding the Tokens

Each session is a sequence of tokens representing poker actions and outcomes.

| Token | Meaning |
|---|---|
| `[HAND]` | Start of a new hand |
| `[FLOP]`, `[TURN]`, `[RIVER]` | Betting stages, followed by board cards |
| `H:<action>` | Hero's action |
| `O:<action>` | Opponent's action |
| `[SHOW]` | Showdown, followed by opponent's cards |
| `O:won`, `O:lost` | End-of-hand outcome |

The transformer sees the sequence of actions, board cards, and outcomes. It **never receives opponent style labels or hidden cards**, except cards revealed at showdown.

## 5. Training and Probing

The intended workflow is:

1. **Train:** Use the token arrays to train a transformer to predict the next action.
2. **Extract:** Collect the transformer's learned representations.
3. **Probe:** Use the held-out style labels and behavior knobs to test whether those representations encode opponent play style.

The key question is whether style information emerges from next-action prediction alone, without explicit style supervision.