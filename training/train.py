"""Train a randomly initialized HookedTransformer on poker sessions (action prediction).

Source of truth: training-specification.md. How the training was carried out is described
in PROCESS.md.

What this script does:
  - Model: transformer_lens 3.9.0 HookedTransformer built from HookedTransformerConfig
    (random init, no from_pretrained). Default architecture = the spec: 6 blocks,
    d_model 256, 8 heads x 32, d_mlp 1024, GELU, LayerNorm, learned positional
    embeddings, causal attention, n_ctx 1024. The ordered 68-token vocab comes from
    vocab.json; no tokenizer and no BOS token.
  - Data: every batch is a set of whole rows (sessions). Rows are never concatenated or
    windowed. Trailing PAD columns shared by the whole micro-batch are trimmed, which
    does not change any real token's output (right padding + causal attention).
    Style labels and knobs are never loaded.
  - Loss: our own next-token cross-entropy (full 68-way softmax), averaged ONLY over
    positions whose target is one of the 8 action tokens (H:/O: fold/check/call/raise).
    Cards, O:won/O:lost, structure tokens and PAD are never targets but stay visible as
    inputs.
  - Optimisation: AdamW, lr 3e-4 (constant), weight decay 0.01, physical batch 8,
    gradient accumulation 4 (effective batch 32), grad-norm clipping 1.0, 3 epochs, fp32.
  - Logging: train/val action loss, hero-action accuracy, opponent-action accuracy,
    learning rate and gradient norm. Frequent val logs use a fixed val subset (first
    --val_subset rows); the full val set is used at step 0 and at each epoch boundary.
    Everything is appended to metrics.jsonl in the run dir.
  - Checkpoints (in <out_dir>/seed<N>/):
      step_XXXXXX.pt  weights only, at training fractions 0, .1, .3, 1, 3, 10, 30, 60,
                      100 % of optimizer steps (step 0 = before any update)
      epoch_E.pt      full resumable state at each epoch boundary (weights, optimizer,
                      scaler state (None: fp32), RNG states, epoch, step, metric history)
      run_meta.json   architecture, parameter count, vocab fingerprint, action ids,
                      seed, package versions, git commit + dirty flag, exact command(s),
                      dataset sizes and fingerprints, checkpoint schedule, progress
  - Fails loudly: exits non-zero on a non-finite loss / grad norm, or if the final
    full-val action loss is not below the step-0 value.
  - Plain fp32 weights: no torch.compile, weight folding, pruning or quantization.

Usage:
    python training/train.py --data data/full --seed 0            # -> training/checkpoints/seed0/
    python training/train.py --data data/full --seed 0 --resume   # continue from latest epoch_E.pt

Reload any checkpoint (e.g. for probing):
    import sys; sys.path.insert(0, "training")   # from the repo root
    from train import load_checkpoint
    model, ckpt = load_checkpoint("training/checkpoints/seed0/step_008439.pt")
    logits, cache = model.run_with_cache(tokens)   # tokens: int64 [batch, pos]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

ACTIONS = ("fold", "check", "call", "raise")
HERO_ACTIONS = tuple(f"H:{a}" for a in ACTIONS)
OPP_ACTIONS = tuple(f"O:{a}" for a in ACTIONS)
CKPT_FRACTIONS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
EXPECTED_VOCAB_SIZE = 68
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))  # <repo>/training
REPO_DIR = os.path.dirname(TRAINING_DIR)                   # <repo> (data/ lives here)

# Args that define the training run. A resumed run must match them exactly.
RUN_DEFINING_ARGS = ("seed", "epochs", "batch_size", "grad_accum", "lr", "weight_decay",
                     "grad_clip", "val_subset", "oracle_style")


# ----------------------------------------------------------------------------- data

def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vocab(folder):
    with open(os.path.join(folder, "vocab.json")) as f:
        vocab = json.load(f)["vocab"]
    if len(vocab) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"spec requires a {EXPECTED_VOCAB_SIZE}-token vocab, got {len(vocab)}")
    return vocab


def vocab_fingerprint(vocab):
    """sha256 of the ordered token list (order matters: it defines the token ids)."""
    return hashlib.sha256(json.dumps(vocab).encode()).hexdigest()


def action_token_ids(vocab):
    """{token: id} for the 8 action tokens, plus id tensors for hero / opponent actions."""
    ids = {tok: vocab.index(tok) for tok in HERO_ACTIONS + OPP_ACTIONS}
    hero = torch.tensor([ids[t] for t in HERO_ACTIONS])
    opp = torch.tensor([ids[t] for t in OPP_ACTIONS])
    return ids, hero, opp


def load_tokens(folder, split):
    """Token array as stored (uint16). Cast to int64 per batch with to_batch()."""
    return np.load(os.path.join(folder, f"{split}_tokens.npy"))


def style_tokens(folder):
    """Names of the 4 extra tokens used ONLY by the --oracle_style control model."""
    with open(os.path.join(folder, "vocab.json")) as f:
        return [f"[STYLE:{s}]" for s in json.load(f)["styles"]]


def add_style_tokens(arr, styles, n_base_vocab):
    """Oracle control only: prepend each session's true-style token (id n_base_vocab + style).

    The row shifts right by one; the dropped last column must be PAD (sessions are at most
    953 tokens long, so there is room at n_ctx 1024). Never used for the main model.
    """
    if (arr[:, -1] != 0).any():
        raise ValueError("no room to prepend a style token (last column is not PAD)")
    out = np.zeros_like(arr)
    out[:, 0] = n_base_vocab + styles.astype(arr.dtype)
    out[:, 1:] = arr[:, :-1]
    return out


def array_sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def to_batch(arr, idx, pad_id, device):
    """Whole rows `idx` -> int64 tensor on device, trailing all-PAD columns trimmed."""
    batch = torch.from_numpy(arr[idx].astype(np.int64))
    return trim_padding(batch, pad_id).to(device)


def trim_padding(batch, pad_id):
    """Drop PAD columns at the right end that every row in the batch shares."""
    nonpad = (batch != pad_id).any(dim=0).nonzero()
    length = int(nonpad.max()) + 1 if len(nonpad) else 1
    return batch[:, :length]


def epoch_permutation(n, seed, epoch):
    """Data order for one epoch, a pure function of (seed, epoch) so resume is exact."""
    return np.random.default_rng([seed, epoch]).permutation(n)


# ---------------------------------------------------------------------------- model

def build_config(d_vocab, n_ctx, seed, device, n_layers=6, d_model=256, n_heads=8,
                 d_head=32, d_mlp=1024):
    """Plain dict of HookedTransformerConfig kwargs (saved in every checkpoint)."""
    return dict(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_mlp=d_mlp,
        act_fn="gelu",
        normalization_type="LN",
        positional_embedding_type="standard",  # learned absolute positions
        attention_dir="causal",
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        default_prepend_bos=False,  # no BOS token anywhere (we feed token tensors)
        seed=seed,  # HookedTransformerConfig seeds python/numpy/torch before init
        device=str(device),
    )


def build_model(cfg_dict, device=None):
    cfg = dict(cfg_dict)
    if device is not None:
        cfg["device"] = str(device)
    return HookedTransformer(HookedTransformerConfig(**cfg))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device="auto"):
    """Rebuild a HookedTransformer from any checkpoint saved by this script.

    Works for weights-only step_XXXXXX.pt and full epoch_E.pt files. Returns
    (model in eval mode, checkpoint dict). Building the config re-seeds the global RNGs
    (TransformerLens behaviour); the weights are then overwritten by the checkpoint.
    """
    device = get_device(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# -------------------------------------------------------------------------- metrics

def action_stats(logits, tokens, hero_ids, opp_ids):
    """Sums over positions whose target is an action token.

    logits: [batch, pos, vocab], tokens: [batch, pos]. The logit at position t predicts
    tokens[t + 1]. Returns `loss_sum` with grad (sum of full-vocab CE over action
    targets) plus detached counts. Two accuracies:
      restricted (headline): argmax among the 4 action tokens of the target's player;
      full-vocab: argmax over all tokens == target (also penalises guessing the wrong
      player, e.g. who acts first after [HAND], which is random).
    """
    pred = logits[:, :-1]
    target = tokens[:, 1:]
    is_hero = torch.isin(target, hero_ids)
    is_opp = torch.isin(target, opp_ids)
    is_act = is_hero | is_opp
    act_logits = pred[is_act].float()
    act_target = target[is_act]
    ce = F.cross_entropy(act_logits, act_target, reduction="none")
    correct = act_logits.argmax(-1) == act_target
    opp_sel = is_opp[is_act]
    hero_sel = ~opp_sel
    with torch.no_grad():
        own_ids = torch.where(opp_sel[:, None], opp_ids[None], hero_ids[None])  # [n, 4]
        own_logits = act_logits.gather(1, own_ids)
        correct_r = own_ids.gather(1, own_logits.argmax(1, keepdim=True)).squeeze(1) == act_target
        stats = {
            "n_act": int(is_act.sum()),
            "n_hero": int(hero_sel.sum()),
            "n_opp": int(opp_sel.sum()),
            "hero_correct": int(correct[hero_sel].sum()),
            "opp_correct": int(correct[opp_sel].sum()),
            "hero_correct_r": int(correct_r[hero_sel].sum()),
            "opp_correct_r": int(correct_r[opp_sel].sum()),
            "hero_loss_sum": float(ce[hero_sel].sum()),
            "opp_loss_sum": float(ce[opp_sel].sum()),
        }
    stats["loss_sum"] = ce.sum()
    return stats


def count_action_targets(batch, action_ids):
    return int(torch.isin(batch[:, 1:], action_ids).sum())


class MetricSums:
    """Accumulates action_stats() outputs and turns them into means."""

    KEYS = ("n_act", "n_hero", "n_opp", "hero_correct", "opp_correct",
            "hero_correct_r", "opp_correct_r", "hero_loss_sum", "opp_loss_sum", "loss_sum")

    def __init__(self):
        self.s = {k: 0.0 for k in self.KEYS}

    def add(self, stats):
        for k in self.KEYS:
            v = stats[k]
            self.s[k] += float(v.detach()) if torch.is_tensor(v) else v

    def means(self):
        s = self.s
        return {
            "action_loss": s["loss_sum"] / max(s["n_act"], 1),
            "hero_loss": s["hero_loss_sum"] / max(s["n_hero"], 1),
            "opp_loss": s["opp_loss_sum"] / max(s["n_opp"], 1),
            "hero_acc_r": s["hero_correct_r"] / max(s["n_hero"], 1),
            "opp_acc_r": s["opp_correct_r"] / max(s["n_opp"], 1),
            "hero_acc": s["hero_correct"] / max(s["n_hero"], 1),
            "opp_acc": s["opp_correct"] / max(s["n_opp"], 1),
            "n_hero": int(s["n_hero"]),
            "n_opp": int(s["n_opp"]),
        }


@torch.no_grad()
def evaluate(model, arr, pad_id, hero_ids, opp_ids, batch_size, device):
    """Action-only loss / accuracies over all rows of `arr` (whole sessions)."""
    was_training = model.training
    model.eval()
    sums = MetricSums()
    for i in range(0, len(arr), batch_size):
        batch = to_batch(arr, np.arange(i, min(i + batch_size, len(arr))), pad_id, device)
        logits = model(batch, return_type="logits")
        sums.add(action_stats(logits, batch, hero_ids, opp_ids))
    model.train(was_training)
    return sums.means()


# ------------------------------------------------------------------------- metadata

def git_info():
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")  # includes untracked, non-ignored files
        return {"commit": commit, "dirty": bool(status), "dirty_files": status.splitlines()}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return {"commit": None, "dirty": None, "error": str(e)}


def package_versions():
    import transformer_lens
    try:
        from importlib.metadata import version
        tl_version = version("transformer_lens")
    except Exception:
        tl_version = getattr(transformer_lens, "__version__", None)
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "transformer_lens": tl_version,
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def launch_record(args, resumed_from):
    return {
        "command": shlex.join([sys.executable] + sys.argv),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "resumed_from": resumed_from,
        "git": git_info(),
        "packages": package_versions(),
        "args": vars(args),
    }


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def atomic_save(obj, path):
    """torch.save to a temp file then rename, so a kill never leaves a corrupt file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------------ RNG state

def get_rng_state():
    np_state = np.random.get_state(legacy=False)
    state = {
        "python": random.getstate(),
        "numpy": {"bit_generator": np_state["bit_generator"],
                  "key": np_state["state"]["key"].tolist(),
                  "pos": int(np_state["state"]["pos"]),
                  "has_gauss": int(np_state["has_gauss"]),
                  "gauss": float(np_state["gauss"])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state({"bit_generator": n["bit_generator"],
                         "state": {"key": np.array(n["key"], dtype=np.uint32), "pos": n["pos"]},
                         "has_gauss": n["has_gauss"], "gauss": n["gauss"]})
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


# --------------------------------------------------------------------- checkpointing

def checkpoint_schedule(total_steps):
    """{step: [fractions]} with step = round(fraction * total optimizer steps)."""
    sched = {}
    for f in CKPT_FRACTIONS:
        step = int(math.floor(f * total_steps + 0.5))
        sched.setdefault(step, []).append(f)
    return sched


def latest_epoch_checkpoint(run_dir):
    found = []
    for path in glob.glob(os.path.join(run_dir, "epoch_*.pt")):
        m = re.fullmatch(r"epoch_(\d+)\.pt", os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return max(found)[1] if found else None


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO_DIR, "data", "full"), help="folder made by data/prepare.py")
    p.add_argument("--out_dir", default=os.path.join(TRAINING_DIR, "checkpoints"), help="run dir = <out_dir>/seed<N>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from the latest epoch_E.pt")
    # architecture (spec defaults)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--d_mlp", type=int, default=1024)
    p.add_argument("--n_ctx", type=int, default=1024)
    # optimisation (spec pilot values)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8, help="physical batch (whole sessions)")
    p.add_argument("--grad_accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging / eval
    p.add_argument("--log_interval", type=int, default=10, help="optimizer steps between train logs")
    p.add_argument("--eval_interval", type=int, default=250, help="optimizer steps between val-subset evals")
    p.add_argument("--val_subset", type=int, default=1000, help="first N val rows used for frequent evals")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--oracle_style", action="store_true",
                   help="CONTROL MODEL ONLY: prepend each session's true-style token (loads style "
                        "labels). Never use for the main model.")
    p.add_argument("--stop_after_epoch", type=int, default=None,
                   help="testing aid: exit cleanly after saving this epoch's checkpoint")
    return p.parse_args(argv)


class TrainingFailed(Exception):
    pass


def main(argv=None):
    args = parse_args(argv)
    device = get_device(args.device)
    run_dir = os.path.join(args.out_dir, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, "run_meta.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # ---- data (tokens + vocab only; style labels / knobs are never loaded)
    vocab = load_vocab(args.data)
    pad_id = vocab.index("[PAD]")
    action_ids_map, hero_ids, opp_ids = action_token_ids(vocab)
    hero_ids, opp_ids = hero_ids.to(device), opp_ids.to(device)
    all_action_ids = torch.cat([hero_ids, opp_ids])
    train_arr = load_tokens(args.data, "train")
    val_arr = load_tokens(args.data, "val")
    raw_sha = {"train": array_sha256(train_arr), "val": array_sha256(val_arr)}
    if args.oracle_style:  # control model only: style labels become an input token
        extra = style_tokens(args.data)
        train_arr = add_style_tokens(train_arr, np.load(os.path.join(args.data, "train_styles.npy")), len(vocab))
        val_arr = add_style_tokens(val_arr, np.load(os.path.join(args.data, "val_styles.npy")), len(vocab))
        vocab = vocab + extra
        print(f"ORACLE CONTROL: prepended true-style tokens {extra} (ids {len(vocab) - 4}-{len(vocab) - 1})")
    val_sub = val_arr[: args.val_subset]
    if train_arr.shape[1] > args.n_ctx:
        raise ValueError(f"data seq_len {train_arr.shape[1]} > n_ctx {args.n_ctx}")
    cpu_action_ids = all_action_ids.cpu()

    n_train = len(train_arr)
    micro_per_epoch = math.ceil(n_train / args.batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    schedule = checkpoint_schedule(total_steps)

    # ---- model + optimizer (fp32, no compile)
    cfg_dict = build_config(len(vocab), args.n_ctx, args.seed, device, args.n_layers,
                            args.d_model, args.n_heads, args.d_head, args.d_mlp)
    model = build_model(cfg_dict)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = count_params(model)
    cfg_saved = {k: v for k, v in cfg_dict.items() if k != "device"}

    data_info = {
        "dir": args.data,
        "train_sessions": int(n_train),
        "val_sessions": int(len(val_arr)),
        "val_subset_sessions": int(len(val_sub)),
        "seq_len": int(train_arr.shape[1]),
        "train_tokens_sha256": raw_sha["train"],  # of the stored arrays (before any style token)
        "val_tokens_sha256": raw_sha["val"],
        "oracle_style_token_prepended": bool(args.oracle_style),
    }
    core_meta = {  # embedded in every checkpoint so each file is self-describing
        "seed": args.seed,
        "oracle_style": bool(args.oracle_style),
        "n_params": n_params,
        "vocab_sha256": vocab_fingerprint(vocab),
        "action_token_ids": action_ids_map,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
    }

    # ---- fresh start or resume
    history = []
    start_epoch, step = 0, 0
    latest = latest_epoch_checkpoint(run_dir)
    if args.resume:
        if latest is None:
            raise SystemExit(f"--resume given but no epoch_*.pt in {run_dir}")
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        if ckpt["config"] != cfg_saved:
            raise SystemExit(f"config mismatch with {latest}:\n{ckpt['config']}\nvs\n{cfg_saved}")
        defaults = vars(parse_args([]))
        for k in RUN_DEFINING_ARGS:
            saved = ckpt["args"].get(k, defaults[k])  # args added later default for old runs
            if saved != getattr(args, k):
                raise SystemExit(f"arg {k} differs from {latest}: {saved} vs {getattr(args, k)}")
        if ckpt["meta"]["train_tokens_sha256"] != data_info["train_tokens_sha256"]:
            raise SystemExit("training data differs from the checkpoint's data")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng"])
        start_epoch, step, history = ckpt["epoch"], ckpt["step"], ckpt["history"]
        with open(meta_path) as f:
            meta = json.load(f)
        meta["launches"].append(launch_record(args, latest))
        with open(metrics_path, "w") as f:  # drop entries logged after this checkpoint
            for h in history:
                f.write(json.dumps(h) + "\n")
        print(f"resumed from {latest}: epoch {start_epoch}, step {step}", flush=True)
    else:
        if latest is not None:
            raise SystemExit(f"{run_dir} already has {os.path.basename(latest)}; "
                             "pass --resume, or delete the run dir to start over")
        launch = launch_record(args, None)
        meta = {
            "spec": "training-specification.md",
            "status": "running",
            "seed": args.seed,
            "architecture": cfg_saved,
            "n_params": n_params,
            "oracle_style": bool(args.oracle_style),
            "vocab": {"size": len(vocab), "sha256": core_meta["vocab_sha256"], "tokens": vocab,
                      "path": os.path.join(args.data, "vocab.json")},
            "action_token_ids": action_ids_map,
            "data": data_info,
            "training": {
                "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8, decay on all params)",
                "lr": args.lr, "lr_schedule": "constant", "weight_decay": args.weight_decay,
                "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                "effective_batch": args.batch_size * args.grad_accum,
                "grad_clip": args.grad_clip, "epochs": args.epochs,
                "micro_batches_per_epoch": micro_per_epoch,
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                "precision": "fp32", "loss": "full-vocab CE, mean over action-token targets",
            },
            "checkpoint_schedule": [
                {"step": s, "fractions": fr, "file": f"step_{s:06d}.pt"}
                for s, fr in sorted(schedule.items())
            ],
            "epoch_checkpoints": [],
            "git": launch["git"],
            "packages": launch["packages"],
            "command": launch["command"],
            "launches": [launch],
            "progress": {"epoch": 0, "step": 0},
            "results": {},
        }
        open(metrics_path, "w").close()

    def log(entry):
        history.append(entry)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_meta(**updates):
        meta.update(updates)
        meta["progress"] = {"epoch": step / steps_per_epoch, "completed_epochs": epoch_done,
                            "step": step, "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        write_json(meta_path, meta)

    def eval_and_log(kind, arr):
        t = time.time()
        m = evaluate(model, arr, pad_id, hero_ids, opp_ids, args.eval_batch_size, device)
        log({"kind": kind, "step": step, "epoch": step / steps_per_epoch, **m,
             "eval_seconds": round(time.time() - t, 1)})
        print(f"[{kind}] step {step} | action_loss {m['action_loss']:.4f} | "
              f"opp_acc_r {m['opp_acc_r']:.4f} | hero_acc_r {m['hero_acc_r']:.4f} | "
              f"hero_acc {m['hero_acc']:.4f} | opp_acc {m['opp_acc']:.4f}", flush=True)
        return m

    def save_weights_if_scheduled():
        if step not in schedule:
            return
        path = os.path.join(run_dir, f"step_{step:06d}.pt")
        atomic_save({"kind": "weights", "state_dict": model.state_dict(), "config": cfg_saved,
                     "step": step, "epoch": step / steps_per_epoch,
                     "fractions": schedule[step], "meta": core_meta}, path)
        print(f"saved {path} (fractions {schedule[step]})", flush=True)

    epoch_done = start_epoch
    print(f"device {device} | params {n_params:,} | train {train_arr.shape} val {val_arr.shape}")
    print(f"steps/epoch {steps_per_epoch} | total steps {total_steps} | "
          f"checkpoint steps {sorted(schedule)} | run dir {run_dir}", flush=True)
    save_meta()

    try:
        if step == 0:
            m0 = eval_and_log("val_full", val_arr)
            eval_and_log("val_subset", val_sub)
            meta["results"]["step0_val_full"] = m0
            save_weights_if_scheduled()

        t0, step_t0 = time.time(), step
        window, gnorms = MetricSums(), []
        for epoch in range(start_epoch, args.epochs):
            perm = epoch_permutation(n_train, args.seed, epoch)
            micro = [perm[i: i + args.batch_size] for i in range(0, n_train, args.batch_size)]
            for g in range(0, len(micro), args.grad_accum):
                batches = [to_batch(train_arr, idx, pad_id, "cpu") for idx in micro[g: g + args.grad_accum]]
                # Normalise by the action-target count of the whole effective batch, so the
                # step's loss is the mean over all its action targets (as one batch of 32).
                n_act_total = sum(count_action_targets(b, cpu_action_ids) for b in batches)
                optimizer.zero_grad(set_to_none=True)
                step_sums = MetricSums()
                for b in batches:
                    b = b.to(device)
                    logits = model(b, return_type="logits")
                    stats = action_stats(logits, b, hero_ids, opp_ids)
                    (stats["loss_sum"] / max(n_act_total, 1)).backward()
                    step_sums.add(stats)
                step_loss = step_sums.means()["action_loss"]
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                if not (math.isfinite(step_loss) and math.isfinite(gnorm)):
                    raise TrainingFailed(f"non-finite loss {step_loss} / grad norm {gnorm} at step {step + 1}")
                optimizer.step()
                step += 1
                window.add(step_sums.s)
                gnorms.append(gnorm)

                if step % args.log_interval == 0 or step == total_steps:
                    m = window.means()
                    elapsed = time.time() - t0
                    rate = (step - step_t0) / max(elapsed, 1e-9)
                    log({"kind": "train", "step": step, "epoch": step / steps_per_epoch,
                         "lr": optimizer.param_groups[0]["lr"],
                         "grad_norm_mean": float(np.mean(gnorms)), "grad_norm_max": float(np.max(gnorms)),
                         **m, "steps_per_sec": rate})
                    eta = (total_steps - step) / max(rate, 1e-9) / 60
                    mem = (f" | max mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
                           if device.type == "cuda" else "")
                    print(f"step {step}/{total_steps} | loss {m['action_loss']:.4f} | opp_r {m['opp_acc_r']:.3f} "
                          f"| hero_r {m['hero_acc_r']:.3f} | lr {optimizer.param_groups[0]['lr']:.1e} "
                          f"| gnorm {np.mean(gnorms):.2f} | {rate:.2f} steps/s | eta {eta:.0f} min{mem}",
                          flush=True)
                    window, gnorms = MetricSums(), []

                if step % args.eval_interval == 0 or step in schedule:
                    eval_and_log("val_subset", val_sub)
                save_weights_if_scheduled()

            # ---- epoch boundary: full val + full resumable checkpoint
            epoch_done = epoch + 1
            mf = eval_and_log("val_full", val_arr)
            meta["results"][f"epoch{epoch_done}_val_full"] = mf
            path = os.path.join(run_dir, f"epoch_{epoch_done}.pt")
            atomic_save({"kind": "epoch", "state_dict": model.state_dict(), "config": cfg_saved,
                         "optimizer": optimizer.state_dict(), "scaler": None,  # fp32: no scaler
                         "rng": get_rng_state(), "epoch": epoch_done, "step": step,
                         "history": history, "args": vars(args),
                         "meta": {**core_meta, "train_tokens_sha256": data_info["train_tokens_sha256"]}},
                        path)
            meta["epoch_checkpoints"] = [e for e in meta["epoch_checkpoints"] if e["epoch"] != epoch_done]
            meta["epoch_checkpoints"].append({"epoch": epoch_done, "step": step, "file": os.path.basename(path)})
            save_meta()
            print(f"saved {path}", flush=True)
            if args.stop_after_epoch is not None and epoch_done >= args.stop_after_epoch and epoch_done < args.epochs:
                save_meta(status=f"stopped after epoch {epoch_done} (--stop_after_epoch)")
                print("stopping early (--stop_after_epoch); continue with --resume")
                return 0

        # ---- final sanity check (full val, final vs step 0)
        epoch_done = args.epochs
        v0 = next(h for h in history if h["kind"] == "val_full" and h["step"] == 0)["action_loss"]
        vf = [h for h in history if h["kind"] == "val_full"][-1]["action_loss"]
        if not vf < v0:
            raise TrainingFailed(f"final val action loss {vf:.4f} is not below step-0 value {v0:.4f}")
        save_meta(status="completed")
        print(f"done: val action loss {v0:.4f} (step 0) -> {vf:.4f} (step {step})", flush=True)
        return 0
    except TrainingFailed as e:
        save_meta(status=f"failed: {e}")
        print(f"TRAINING FAILED: {e}", file=sys.stderr, flush=True)
        return 1
    except BaseException as e:  # OOM, Ctrl-C, ...: record it, then re-raise
        save_meta(status=f"crashed: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
