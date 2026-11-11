"""Turn sessions.jsonl into arrays ready for training and probing.

Splits by session (never by hand), maps tokens to ids with a fixed vocab,
and pads every session to the same length.

Usage:
    python data/prepare.py --inp data/sessions_small.jsonl --out data/small

Outputs in --out:
    vocab.json                     token <-> id
    train_tokens.npy, val_tokens.npy   (n_sessions, seq_len) uint16, padded with [PAD]
    train_styles.npy, val_styles.npy   (n_sessions,) style id (for probing only)
    train_knobs.npy,  val_knobs.npy    (n_sessions, 3) play_rate, aggression, bluff_rate (NaN for random)
"""

import argparse
import json
import os
import random

import numpy as np

STYLE_NAMES = ["tight", "loose", "bluffer", "random"]
KNOB_NAMES = ["play_rate", "aggression", "bluff_rate"]


def build_vocab():
    """Fixed vocab, so every teammate gets identical token ids."""
    vocab = ["[PAD]", "[HAND]", "[FLOP]", "[TURN]", "[RIVER]", "[SHOW]", "O:won", "O:lost"]
    for who in ["H", "O"]:
        for action in ["fold", "check", "call", "raise"]:
            vocab.append(f"{who}:{action}")
    for rank in "23456789TJQKA":
        for suit in "cdhs":
            vocab.append(rank + suit)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default="data/sessions_small.jsonl")
    parser.add_argument("--out", default="data/small")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    vocab = build_vocab()
    stoi = {tok: i for i, tok in enumerate(vocab)}

    sessions = [json.loads(line) for line in open(args.inp)]
    random.seed(0)
    random.shuffle(sessions)
    n_val = int(len(sessions) * args.val_frac)
    splits = {"val": sessions[:n_val], "train": sessions[n_val:]}

    os.makedirs(args.out, exist_ok=True)
    n_truncated = 0
    for name, rows in splits.items():
        tokens = np.zeros((len(rows), args.seq_len), dtype=np.uint16)  # 0 = [PAD]
        styles = np.zeros(len(rows), dtype=np.int64)
        knobs = np.full((len(rows), 3), np.nan, dtype=np.float32)

        for i, row in enumerate(rows):
            ids = [stoi[t] for t in row["tokens"]][: args.seq_len]
            n_truncated += len(row["tokens"]) > args.seq_len
            tokens[i, : len(ids)] = ids
            styles[i] = STYLE_NAMES.index(row["style"])
            if row["knobs"]:
                knobs[i] = [row["knobs"][k] for k in KNOB_NAMES]

        np.save(f"{args.out}/{name}_tokens.npy", tokens)
        np.save(f"{args.out}/{name}_styles.npy", styles)
        np.save(f"{args.out}/{name}_knobs.npy", knobs)
        print(f"{name}: {len(rows)} sessions")

    with open(f"{args.out}/vocab.json", "w") as f:
        json.dump({"vocab": vocab, "styles": STYLE_NAMES, "knobs": KNOB_NAMES}, f, indent=1)

    print(f"vocab size {len(vocab)}, seq_len {args.seq_len}, truncated sessions: {n_truncated}")


if __name__ == "__main__":
    main()