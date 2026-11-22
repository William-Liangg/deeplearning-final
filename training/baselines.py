"""Baselines for opponent-action prediction (training-specification.md, "Validation and handoff").

All numbers are on the validation split (whole sessions), at every position whose TARGET
is an opponent action token (O:fold / O:check / O:call / O:raise). Statistics that a
baseline needs are estimated on the training split only. Style labels and knobs are never
loaded. Every result is also broken down by hand number (1-50): the hand that contains the
predicting position (= number of [HAND] tokens up to and including it).

Two definitions below are still pending team confirmation (see PROCESS.md), so both
candidate versions of "action frequencies" are reported.

Data-only baselines (no model; run any time):
  majority           Always predict the single most frequent opponent action in the
                     training data.
  freq_overall       Predict the overall opponent-action distribution of the training data
                     at every target. Same accuracy as `majority` (same argmax); it adds a
                     cross-entropy number. Loss is over the 4 opponent actions.
  freq_running       Per-opponent running count within the session: at each target,
                     p(a) = (count of a among THIS opponent's earlier actions in the
                     session + alpha * p_overall(a)) / (n_earlier + alpha). The prior
                     strength alpha is chosen on the training split (lowest loss over a
                     small grid). Prediction = argmax; ties fall back to the overall
                     frequencies through the prior.
  Note: the three baselines above ignore which actions are legal at that point (e.g. no
  "check" when facing a raise), which the model can read from context. The *_legal
  versions below fix that.

Legality-aware data-only baselines (only ever predict legal actions):
  The opponent's situation at each action is reconstructed from the tokens with RLCard's
  limit hold'em rules (rlcard/games/limitholdem/round.py, get_legal_actions):
    facing_bet = the previous token is a raise, or is [HAND] (the first actor preflop
                 faces the big blind). After a call/check, or at the start of a post-flop
                 street (previous token is a board card), the player is not facing a bet.
    can_raise  = fewer than 4 raises so far in this street (blinds don't count).
    legal set  = fold always; check iff not facing_bet; call iff facing_bet;
                 raise iff can_raise.
  situation = (facing_bet, can_raise), 4 possible values. The output reports how many real
  opponent / hero actions fall outside the reconstructed legal set (expected: 0).
  freq_overall_legal  p(a | situation) from the training data (+0.5 pseudo-count per
                      legal action), zero for illegal actions. Prediction = argmax, so
                      this is also the "legal majority" baseline.
  freq_running_legal  Per-opponent running count within the session, conditioned on the
                      situation: p(a) = (this opponent's earlier count of a in the SAME
                      situation + alpha * p(a | situation)) / (n_earlier_same + alpha),
                      over legal actions only. alpha chosen on the training split.

Ceilings (use the TRUE style labels; evaluation only, never used to train the main model):
  oracle_style_freq_legal  p(a | true style, situation) from the training data (+0.5
                           pseudo-count per legal action), legal actions only. "What a
                           frequency counter could do if it were told the style."
  oracle_model             A separate control model (train.py --oracle_style): same
                           architecture and settings, but every session starts with a token
                           for its true style. Evaluated like `model` (full context).

Model-based baselines (need checkpoints; run after training):
  model              The trained checkpoint (--ckpt) with full session context.
  init               The untrained step-0 checkpoint (--init_ckpt), full context.
  current_hand_only  The trained checkpoint, but its input is cut to just the current hand:
                     each hand (from its [HAND] token up to the next [HAND]) is fed as its
                     own sequence, so nothing from earlier hands is visible. Two variants:
                       *_origpos  the hand keeps its original absolute positions (the
                                  positional embeddings are replaced via a hook on
                                  hook_pos_embed), so only the history is removed;
                       *_pos0     the hand restarts at position 0, i.e. the input is
                                  literally just the hand.
  HEADLINE METRIC: `acc_restricted` = argmax among the 4 opponent-action tokens only
  (comparable to the data-only baselines, which only ever predict actions). Also reported:
  `acc` = argmax over the full vocab == target (as in train.py); it additionally penalises
  guessing the wrong player, e.g. who acts first after [HAND], which is random.
  `loss` = full-vocab cross-entropy (as in training), `loss_restricted` = cross-entropy
  after renormalising over the 4 opponent-action tokens (comparable to freq_* losses).

Usage:
    python training/baselines.py --data data/full --out training/results/baselines_data.json
    python training/baselines.py --data data/full --ckpt training/checkpoints/seed0/step_008439.pt \\
        --init_ckpt training/checkpoints/seed0/step_000000.pt --out training/results/baselines_seed0.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from train import (REPO_DIR, OPP_ACTIONS, HERO_ACTIONS, add_style_tokens, get_device, load_checkpoint,
                   load_tokens, load_vocab)

SHOW_HANDS = (1, 2, 3, 5, 10, 20, 30, 40, 50)
ALPHA_GRID = (0.25, 0.5, 1, 2, 4, 8, 16, 32)


# ----------------------------------------------------------------- target positions

def opp_targets(arr, opp_ids, hand_id):
    """For every opponent-action target in `arr` [N, L]:
    row, target position j (predicted from position j-1), action index 0-3, hand number."""
    lut = np.full(int(arr.max()) + 1, -1, dtype=np.int64)
    lut[opp_ids] = np.arange(len(opp_ids))
    act = lut[arr]  # [N, L], -1 where not an opponent action
    hand_no = np.cumsum(arr == hand_id, axis=1)  # hands started up to and incl. each pos
    rows, pos = np.nonzero(act[:, 1:] >= 0)
    pos = pos + 1
    return rows, pos, act[rows, pos], hand_no[rows, pos - 1]


def summarize(correct, loss, hand_no, n_hands=50):
    out = {"n": int(len(correct)), "acc": float(np.mean(correct))}
    if loss is not None:
        out["loss"] = float(np.mean(loss))
    by_hand = []
    for h in range(1, n_hands + 1):
        m = hand_no == h
        by_hand.append(float(np.mean(correct[m])) if m.any() else None)
    out["acc_by_hand"] = by_hand
    return out


# ------------------------------------------------------------------- data baselines

def overall_freqs(arr, ids):
    counts = np.array([(arr == i).sum() for i in ids], dtype=np.float64)
    return counts / counts.sum(), counts


def running_counts(arr, opp_ids, chunk=10000):
    """Counts of each opponent action strictly before every position, chunked."""
    for s in range(0, len(arr), chunk):
        a = arr[s: s + chunk]
        onehot = np.stack([(a == i) for i in opp_ids], axis=-1).astype(np.int32)  # [n, L, 4]
        before = np.cumsum(onehot, axis=1) - onehot
        yield s, a, before


def running_eval(arr, opp_ids, hand_id, p_overall, alphas):
    """Per-alpha (correct, loss, hand_no) arrays of the running-count baseline."""
    res = {a: ([], [], []) for a in alphas}
    for _, a, before in running_counts(arr, opp_ids):
        rows, pos, act, hand = opp_targets(a, opp_ids, hand_id)
        c = before[rows, pos].astype(np.float64)  # [T, 4] this opponent's earlier counts
        n = c.sum(1, keepdims=True)
        for alpha in alphas:
            p = (c + alpha * p_overall) / (n + alpha)
            res[alpha][0].append(p.argmax(1) == act)
            res[alpha][1].append(-np.log(p[np.arange(len(act)), act]))
            res[alpha][2].append(hand)
    return {a: tuple(np.concatenate(x) for x in v) for a, v in res.items()}


RAISE_CAP = 4  # rlcard limitholdem Game.allowed_raise_num


def situation(arr, vocab):
    """(facing_bet, can_raise) for the player whose action would be token j, for all j."""
    v = {t: i for i, t in enumerate(vocab)}
    raise_ids = [v["H:raise"], v["O:raise"]]
    is_raise = np.isin(arr, raise_ids)
    prev = np.zeros_like(arr)
    prev[:, 1:] = arr[:, :-1]
    facing = np.isin(prev, raise_ids + [v["[HAND]"]])
    street_start = np.isin(arr, [v["[HAND]"], v["[FLOP]"], v["[TURN]"], v["[RIVER]"]])
    raises_before = np.cumsum(is_raise, axis=1, dtype=np.int32) - is_raise  # strictly before j
    idx = np.where(street_start, np.arange(arr.shape[1], dtype=np.int32)[None], 0)
    last_start = np.maximum.accumulate(idx, axis=1)
    in_street = raises_before - np.take_along_axis(raises_before, last_start, axis=1)
    return facing, in_street < RAISE_CAP


def legal_mask(facing, can_raise):
    """Boolean [..., 4] over (fold, check, call, raise)."""
    return np.stack([np.ones_like(facing), ~facing, facing, can_raise], axis=-1)


# legal set per situation key (key = facing_bet * 2 + can_raise)
LEGAL_BY_KEY = legal_mask(np.array([False, False, True, True]), np.array([False, True, False, True]))


def legal_targets(arr, opp_ids, hand_id, vocab):
    """opp_targets() plus situation key (0-3) and legal mask [T, 4] per target."""
    rows, pos, act, hand = opp_targets(arr, opp_ids, hand_id)
    facing, can_raise = situation(arr, vocab)
    f, c = facing[rows, pos], can_raise[rows, pos]
    return rows, pos, act, hand, f.astype(np.int64) * 2 + c, legal_mask(f, c)


def count_illegal(arr, vocab):
    """Real actions outside the reconstructed legal set, per player (should be 0)."""
    facing, can_raise = situation(arr, vocab)
    mask = legal_mask(facing, can_raise)
    out = {}
    for who in ("H", "O"):
        ids = [vocab.index(f"{who}:{a}") for a in ("fold", "check", "call", "raise")]
        lut = np.full(len(vocab), -1)
        lut[ids] = np.arange(4)
        a = lut[arr]
        r, p = np.nonzero(a >= 0)
        out[who] = int((~mask[r, p, a[r, p]]).sum())
    return out


def situation_freqs(arr, opp_ids, hand_id, vocab, chunk=10000):
    """p(a | situation) over legal actions, from all opponent actions in `arr`."""
    counts = np.zeros((4, 4))
    for s in range(0, len(arr), chunk):
        _, _, act, _, key, _ = legal_targets(arr[s: s + chunk], opp_ids, hand_id, vocab)
        np.add.at(counts, (key, act), 1)
    counts = (counts + 0.5) * LEGAL_BY_KEY
    return counts / counts.sum(1, keepdims=True), LEGAL_BY_KEY


def running_legal_eval(arr, opp_ids, hand_id, vocab, p_sit, alphas, chunk=10000):
    """Per-alpha (correct, loss, hand_no) of the situation-conditioned running count."""
    res = {a: ([], [], []) for a in alphas}
    for s in range(0, len(arr), chunk):
        a_ = arr[s: s + chunk]
        rows, pos, act, hand, key, legal = legal_targets(a_, opp_ids, hand_id, vocab)
        # Earlier counts of each action by this opponent in the same situation.
        # Targets from np.nonzero are already ordered by (session, position); a stable sort
        # by (session, situation) keeps time order inside each group.
        group = rows * 4 + key
        idx = np.argsort(group, kind="stable")
        onehot = np.eye(4)[act[idx]]
        excl = np.cumsum(onehot, axis=0) - onehot  # exclusive running sum over all groups
        first = np.r_[0, np.nonzero(np.diff(group[idx]))[0] + 1]
        start_of = np.repeat(first, np.diff(np.r_[first, len(idx)]))
        before = np.empty((len(act), 4))
        before[idx] = excl - excl[start_of]  # restart the count at each group's start
        n = before.sum(1, keepdims=True)
        prior = p_sit[key]
        for alpha in alphas:
            p = (before + alpha * prior) / (n + alpha) * legal
            p = p / p.sum(1, keepdims=True)
            res[alpha][0].append(np.where(legal, p, -1).argmax(1) == act)
            res[alpha][1].append(-np.log(p[np.arange(len(act)), act]))
            res[alpha][2].append(hand)
    return {a: tuple(np.concatenate(x) for x in v) for a, v in res.items()}


def legal_baselines(train, val, vocab):
    opp_ids = [vocab.index(t) for t in OPP_ACTIONS]
    hand_id = vocab.index("[HAND]")
    out = {"illegal_actions_under_reconstruction": {"train": count_illegal(train, vocab),
                                                     "val": count_illegal(val, vocab)}}
    p_sit, _ = situation_freqs(train, opp_ids, hand_id, vocab)
    # index = key = facing_bet * 2 + can_raise
    names = ["not facing, raise capped", "not facing, can raise",
             "facing bet, raise capped", "facing bet, can raise"]
    out["train_opp_freqs_by_situation"] = {
        n: dict(zip(OPP_ACTIONS, p.round(4).tolist())) for n, p in zip(names, p_sit)}
    _, _, act, hand, key, legal = legal_targets(val, opp_ids, hand_id, vocab)
    pred = np.where(legal, p_sit[key], -1).argmax(1)
    out["freq_overall_legal"] = summarize(pred == act, -np.log(p_sit[key, act]), hand)
    train_res = running_legal_eval(train, opp_ids, hand_id, vocab, p_sit, ALPHA_GRID)
    train_loss = {a: float(np.mean(v[1])) for a, v in train_res.items()}
    best = min(train_loss, key=train_loss.get)
    c, l, h = running_legal_eval(val, opp_ids, hand_id, vocab, p_sit, [best])[best]
    out["freq_running_legal"] = {"alpha": best, "alpha_train_loss": train_loss, **summarize(c, l, h)}
    return out


def style_oracle_legal(train, val, train_styles, val_styles, vocab, chunk=10000):
    """CEILING (uses true style labels, evaluation only): p(a | true style, situation)."""
    opp_ids = [vocab.index(t) for t in OPP_ACTIONS]
    hand_id = vocab.index("[HAND]")
    n_styles = int(max(train_styles.max(), val_styles.max())) + 1
    counts = np.zeros((n_styles, 4, 4))
    for s in range(0, len(train), chunk):
        rows, _, act, _, key, _ = legal_targets(train[s: s + chunk], opp_ids, hand_id, vocab)
        np.add.at(counts, (train_styles[s + rows], key, act), 1)
    p = (counts + 0.5) * LEGAL_BY_KEY[None]
    p = p / p.sum(-1, keepdims=True)
    rows, _, act, hand, key, legal = legal_targets(val, opp_ids, hand_id, vocab)
    pp = p[val_styles[rows], key]
    pred = np.where(legal, pp, -1).argmax(1)
    return summarize(pred == act, -np.log(pp[np.arange(len(act)), act]), hand)


def data_baselines(train, val, vocab):
    opp_ids = [vocab.index(t) for t in OPP_ACTIONS]
    hero_ids = [vocab.index(t) for t in HERO_ACTIONS]
    hand_id = vocab.index("[HAND]")
    p_opp, c_opp = overall_freqs(train, opp_ids)
    p_hero, _ = overall_freqs(train, hero_ids)
    rows, pos, act, hand = opp_targets(val, opp_ids, hand_id)
    out = {"train_opp_action_freqs": dict(zip(OPP_ACTIONS, p_opp.tolist())),
           "train_hero_action_freqs": dict(zip(HERO_ACTIONS, p_hero.tolist())),
           "val_opp_targets": int(len(act))}

    maj = int(p_opp.argmax())
    out["majority"] = {"predicts": OPP_ACTIONS[maj], **summarize(act == maj, None, hand)}
    out["freq_overall"] = summarize(act == maj, -np.log(p_opp[act]), hand)

    # choose alpha on the training split, then evaluate on val
    train_res = running_eval(train, opp_ids, hand_id, p_opp, ALPHA_GRID)
    train_loss = {a: float(np.mean(v[1])) for a, v in train_res.items()}
    best = min(train_loss, key=train_loss.get)
    c, l, h = running_eval(val, opp_ids, hand_id, p_opp, [best])[best]
    out["freq_running"] = {"alpha": best, "alpha_train_loss": train_loss, **summarize(c, l, h)}

    # hero majority, for reference
    hero_lut = {i: k for k, i in enumerate(hero_ids)}
    hv = val[:, 1:][np.isin(val[:, 1:], hero_ids)]
    out["hero_majority"] = {"predicts": HERO_ACTIONS[int(p_hero.argmax())],
                            "acc": float(np.mean([hero_lut[t] == p_hero.argmax() for t in hv]))}
    return out


# ------------------------------------------------------------------ model baselines

@torch.no_grad()
def model_full_context(model, val, vocab, device, batch_size=16):
    """Opponent/hero action metrics with the full session as context."""
    opp = torch.tensor([vocab.index(t) for t in OPP_ACTIONS], device=device)
    hero = torch.tensor([vocab.index(t) for t in HERO_ACTIONS], device=device)
    hand_id, pad_id = vocab.index("[HAND]"), vocab.index("[PAD]")
    acc = {k: [] for k in ("opp_c", "opp_cr", "opp_l", "opp_lr", "opp_h", "hero_c", "hero_cr")}
    for s in range(0, len(val), batch_size):
        b = torch.from_numpy(val[s: s + batch_size].astype(np.int64))
        length = int((b != pad_id).any(0).nonzero().max()) + 1
        b = b[:, :length].to(device)
        logits = model(b, return_type="logits").float()
        _collect(logits[:, :-1], b[:, 1:], torch.cumsum(b == hand_id, 1)[:, :-1], opp, hero, acc)
    return _finish(acc)


def _collect(logits, target, hand_no, opp, hero, acc):
    """Append per-target results for opponent and hero action targets."""
    for who, ids in (("opp", opp), ("hero", hero)):
        m = torch.isin(target, ids)
        lg, tg = logits[m], target[m]
        acc[f"{who}_c"].append((lg.argmax(-1) == tg).cpu().numpy())
        sub = lg[:, ids]
        tgt_idx = (tg[:, None] == ids[None]).float().argmax(1)
        acc[f"{who}_cr"].append((sub.argmax(-1) == tgt_idx).cpu().numpy())
        if who == "opp":
            acc["opp_l"].append(F.cross_entropy(lg, tg, reduction="none").cpu().numpy())
            acc["opp_lr"].append(F.cross_entropy(sub, tgt_idx, reduction="none").cpu().numpy())
            acc["opp_h"].append(hand_no[m].cpu().numpy())


def _finish(acc):
    a = {k: np.concatenate(v) for k, v in acc.items()}
    out = summarize(a["opp_c"], a["opp_l"], a["opp_h"])
    r = summarize(a["opp_cr"], a["opp_lr"], a["opp_h"])
    out.update({"acc_restricted": r["acc"], "loss_restricted": r["loss"],
                "acc_restricted_by_hand": r["acc_by_hand"],
                "hero_acc": float(np.mean(a["hero_c"])),
                "hero_acc_restricted": float(np.mean(a["hero_cr"]))})
    return out


def hand_segments(val, hand_id, pad_id):
    """(row, start, tokens) for every hand: from its [HAND] up to the next [HAND]."""
    segs = []
    for r, row in enumerate(val):
        n = int((row != pad_id).sum())
        starts = np.nonzero(row[:n] == hand_id)[0].tolist() + [n]
        for k in range(len(starts) - 1):
            segs.append((r, starts[k], row[starts[k]: starts[k + 1]].astype(np.int64)))
    return segs


@torch.no_grad()
def model_current_hand(model, val, vocab, device, keep_positions, batch_size=512):
    """Each hand fed alone. keep_positions=True keeps the original absolute positions."""
    opp = torch.tensor([vocab.index(t) for t in OPP_ACTIONS], device=device)
    hero = torch.tensor([vocab.index(t) for t in HERO_ACTIONS], device=device)
    hand_id, pad_id = vocab.index("[HAND]"), vocab.index("[PAD]")
    segs = hand_segments(val, hand_id, pad_id)
    hand_no_of = {}
    for r, start, _ in segs:  # hand number = order of the hand within its session
        hand_no_of[(r, start)] = hand_no_of.get(("n", r), 0) + 1
        hand_no_of[("n", r)] = hand_no_of[(r, start)]
    acc = {k: [] for k in ("opp_c", "opp_cr", "opp_l", "opp_lr", "opp_h", "hero_c", "hero_cr")}
    W_pos = model.W_pos
    for s in range(0, len(segs), batch_size):
        chunk = segs[s: s + batch_size]
        L = max(len(t) for _, _, t in chunk)
        b = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        for i, (_, _, t) in enumerate(chunk):
            b[i, : len(t)] = torch.from_numpy(t)
        b = b.to(device)
        hooks = []
        if keep_positions:
            starts = torch.tensor([st for _, st, _ in chunk], device=device)
            pos_idx = (starts[:, None] + torch.arange(L, device=device)[None]).clamp(max=W_pos.shape[0] - 1)

            def set_pos(pos_embed, hook):
                return W_pos[pos_idx]

            hooks = [("hook_pos_embed", set_pos)]
        logits = model.run_with_hooks(b, return_type="logits", fwd_hooks=hooks).float()
        hand_no = torch.tensor([hand_no_of[(r, st)] for r, st, _ in chunk], device=device)
        _collect(logits[:, :-1], b[:, 1:], hand_no[:, None].expand(-1, L - 1), opp, hero, acc)
    return _finish(acc)


# --------------------------------------------------------------------------- report

def print_table(results):
    rows = [(k, v) for k, v in results.items() if isinstance(v, dict) and "acc_by_hand" in v]
    head = f"{'baseline':28s} {'acc_r':>6s} {'loss_r':>6s} {'acc':>6s} {'loss':>6s} | " + \
        " ".join(f"h{h:<4d}" for h in SHOW_HANDS)
    print("\nOpponent-action prediction on val. HEADLINE = acc_r / loss_r (restricted to the 4 O: "
          "actions); acc / loss = full vocab")
    print(head)
    print("-" * len(head))
    for name, v in rows:
        by = v.get("acc_restricted_by_hand", v["acc_by_hand"])
        fmt = lambda x: f"{x:6.4f}" if isinstance(x, float) else f"{'-':>6s}"
        acc_r = v.get("acc_restricted", v["acc"])
        loss_r = v.get("loss_restricted", v.get("loss"))
        print(f"{name:28s} {fmt(acc_r)} {fmt(loss_r)} {fmt(v['acc'])} {fmt(v.get('loss'))} | " +
              " ".join(f"{by[h - 1]:.3f}" if by[h - 1] is not None else "  -  " for h in SHOW_HANDS))
    print("(by-hand columns: restricted accuracy for model rows, plain accuracy otherwise)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO_DIR, "data", "full"))
    p.add_argument("--ckpt", default=None, help="trained checkpoint (enables model baselines)")
    p.add_argument("--init_ckpt", default=None, help="step-0 checkpoint")
    p.add_argument("--oracle_ckpt", default=None, help="oracle control model (train.py --oracle_style)")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_val", type=int, default=None, help="use only the first N val sessions")
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args()

    vocab = load_vocab(args.data)
    train = load_tokens(args.data, "train")
    val = load_tokens(args.data, "val")
    # style labels: evaluation-only ceilings (never given to the main model)
    train_styles = np.load(os.path.join(args.data, "train_styles.npy"))
    val_styles = np.load(os.path.join(args.data, "val_styles.npy"))
    if args.max_val:
        val, val_styles = val[: args.max_val], val_styles[: args.max_val]
    results = {"data": args.data, "val_sessions": int(len(val)), "definitions": "see baselines.py docstring"}
    results.update(data_baselines(train, val, vocab))
    results.update(legal_baselines(train, val, vocab))
    results["oracle_style_freq_legal"] = style_oracle_legal(train, val, train_styles, val_styles, vocab)

    device = get_device(args.device)
    if args.init_ckpt:
        model, ck = load_checkpoint(args.init_ckpt, device)
        results["init"] = {"ckpt": args.init_ckpt, "step": ck["step"],
                           **model_full_context(model, val, vocab, device)}
    if args.oracle_ckpt:
        model, ck = load_checkpoint(args.oracle_ckpt, device)
        val_o = add_style_tokens(val, val_styles, len(vocab))
        results["oracle_model"] = {"ckpt": args.oracle_ckpt, "step": ck["step"],
                                   **model_full_context(model, val_o, vocab, device)}
    if args.ckpt:
        model, ck = load_checkpoint(args.ckpt, device)
        results["model"] = {"ckpt": args.ckpt, "step": ck["step"],
                            **model_full_context(model, val, vocab, device)}
        for keep, tag in ((True, "origpos"), (False, "pos0")):
            results[f"current_hand_only_{tag}"] = {
                "ckpt": args.ckpt, **model_current_hand(model, val, vocab, device, keep)}

    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, dict) or "acc_by_hand" not in v},
                     indent=1))
    print_table(results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
