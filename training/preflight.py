"""Pre-training checks required by training-specification.md ("Validation and handoff").

  1. A tiny sample can be overfit (train.py's model builder and action-only loss).
  2. Checkpoint reload reproduces logits (train.load_checkpoint on a saved file).
  3. run_with_cache exposes residual, per-head attention and MLP activations.
  4. run_with_hooks can alter an activation and change downstream logits (and only
     downstream positions, since attention is causal).

Uses the spec architecture by default. Exits non-zero if any check fails.

Usage:
    python training/preflight.py --data data/small                 # GPU if available
    python training/preflight.py --data data/small --max_len 256   # quicker on CPU
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

from train import (REPO_DIR, action_stats, action_token_ids, build_config, build_model, count_params,
                   get_device, load_checkpoint, load_tokens, load_vocab, to_batch)


def achievable_floor(batch, ids):
    """Best possible (loss, accuracy) when memorising `batch`.

    Different sessions can share an identical prefix (always the first [HAND], sometimes
    whole opening hands) followed by different actions. No model can separate those, so
    the floor is the empirical conditional entropy of the action given its exact prefix,
    and the best accuracy is the majority share within each identical-prefix group.
    """
    action_ids = set(torch.cat(ids).tolist())
    groups = {}
    for row in batch.tolist():
        for t in range(len(row) - 1):
            if row[t + 1] in action_ids:
                groups.setdefault(tuple(row[: t + 1]), []).append(row[t + 1])
    n = sum(len(g) for g in groups.values())
    loss, correct = 0.0, 0
    for g in groups.values():
        counts = np.bincount(g)
        counts = counts[counts > 0]
        loss -= float((counts * np.log(counts / counts.sum())).sum())
        correct += int(counts.max())
    return loss / n, correct / n


def check_overfit(args, cfg, data, ids, device):
    """Train on a few whole sessions until action loss and accuracy reach the floor."""
    floor_loss, best_acc = achievable_floor(data["overfit"], ids)
    target_loss, target_acc = floor_loss + args.overfit_loss, best_acc - 1e-9
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.overfit_lr, weight_decay=0.0)
    batch = data["overfit"]
    hist = []
    for it in range(1, args.overfit_steps + 1):
        logits = model(batch, return_type="logits")
        s = action_stats(logits, batch, *ids)
        loss = s["loss_sum"] / s["n_act"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc = (s["hero_correct"] + s["opp_correct"]) / s["n_act"]
        lv = float(loss.detach())
        if it == 1 or it % 50 == 0:
            hist.append({"step": it, "action_loss": lv, "action_acc": acc})
        if lv < target_loss and acc >= target_acc:
            hist.append({"step": it, "action_loss": lv, "action_acc": acc})
            break
    with torch.no_grad():
        s = action_stats(model(batch, return_type="logits"), batch, *ids)
    final_loss = float(s["loss_sum"]) / s["n_act"]
    final_acc = (s["hero_correct"] + s["opp_correct"]) / s["n_act"]
    ok = final_loss < target_loss and final_acc >= target_acc
    return model, ok, {"sessions": int(batch.shape[0]), "seq_len": int(batch.shape[1]),
                       "action_targets": s["n_act"], "steps_run": it,
                       "final_action_loss": final_loss, "final_action_acc": final_acc,
                       "floor_action_loss": floor_loss, "best_possible_acc": best_acc,
                       "criterion": f"loss < floor + {args.overfit_loss} and acc = best possible",
                       "trace": hist}


def check_reload(model, cfg, data, device):
    """Save in train.py's weights-only format, reload with load_checkpoint, compare logits."""
    model.eval()
    batch = data["val"]
    with torch.no_grad():
        before = model(batch, return_type="logits")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save({"kind": "weights", "state_dict": model.state_dict(), "config": cfg, "step": -1}, path)
        reloaded, _ = load_checkpoint(path, device)
    with torch.no_grad():
        after = reloaded(batch, return_type="logits")
    diff = (before - after).abs().max().item()
    return diff == 0.0, {"max_abs_logit_diff": diff, "compared_shape": list(before.shape)}


def check_cache(model, cfg, data):
    """Residual, per-head attention and MLP activations present with the right shapes."""
    batch = data["val"]
    B, P = batch.shape
    L, H, dh, dm, dmlp = cfg["n_layers"], cfg["n_heads"], cfg["d_head"], cfg["d_model"], cfg["d_mlp"]
    with torch.no_grad():
        logits, cache = model.run_with_cache(batch)
        plain = model(batch, return_type="logits")
    expected = {"hook_embed": (B, P, dm), "hook_pos_embed": (B, P, dm), "ln_final.hook_normalized": (B, P, dm)}
    for l in range(L):
        expected.update({
            f"blocks.{l}.hook_resid_pre": (B, P, dm),
            f"blocks.{l}.hook_resid_mid": (B, P, dm),
            f"blocks.{l}.hook_resid_post": (B, P, dm),
            f"blocks.{l}.attn.hook_q": (B, P, H, dh),
            f"blocks.{l}.attn.hook_k": (B, P, H, dh),
            f"blocks.{l}.attn.hook_v": (B, P, H, dh),
            f"blocks.{l}.attn.hook_z": (B, P, H, dh),
            f"blocks.{l}.attn.hook_attn_scores": (B, H, P, P),
            f"blocks.{l}.attn.hook_pattern": (B, H, P, P),
            f"blocks.{l}.hook_attn_out": (B, P, dm),
            f"blocks.{l}.mlp.hook_pre": (B, P, dmlp),
            f"blocks.{l}.mlp.hook_post": (B, P, dmlp),
            f"blocks.{l}.hook_mlp_out": (B, P, dm),
        })
    problems = []
    for name, shape in expected.items():
        if name not in cache.cache_dict:
            problems.append(f"missing {name}")
        elif tuple(cache[name].shape) != shape:
            problems.append(f"{name} has shape {tuple(cache[name].shape)}, expected {shape}")
    # per-head contributions to the residual stream (from hook_z and W_O)
    if "blocks.0.attn.hook_result" not in cache.cache_dict:
        cache.compute_head_results()
    head_results = cache.stack_head_results(layer=-1)  # [L*H, B, P, d_model]
    per_layer = head_results.reshape(L, H, B, P, dm).sum(1)
    attn_out = torch.stack([cache[f"blocks.{l}.hook_attn_out"] for l in range(L)])
    b_O = torch.stack([model.blocks[l].attn.b_O for l in range(L)])[:, None, None, :]
    head_sum_err = (per_layer + b_O - attn_out).abs().max().item()
    if tuple(head_results.shape) != (L * H, B, P, dm):
        problems.append(f"stack_head_results shape {tuple(head_results.shape)}")
    if head_sum_err > 1e-4:
        problems.append(f"sum of head results + b_O != attn_out (err {head_sum_err:.2e})")
    logit_diff = (logits - plain).abs().max().item()
    if logit_diff > 0:
        problems.append(f"run_with_cache logits differ from forward ({logit_diff:.2e})")
    return not problems, {"hooks_checked": len(expected), "n_cached": len(cache.cache_dict),
                          "head_results_shape": list(head_results.shape),
                          "head_sum_vs_attn_out_err": head_sum_err,
                          "cache_vs_forward_logit_diff": logit_diff, "problems": problems}


def check_hooks(model, cfg, data):
    """Patch one activation at position t0; downstream logits change, earlier ones don't."""
    batch = data["val"]
    P = batch.shape[1]
    t0 = P // 2
    layer = min(2, cfg["n_layers"] - 1)
    hook_name = f"blocks.{layer}.hook_resid_post"
    calls = []

    def scramble(resid, hook):
        calls.append(hook.name)
        resid = resid.clone()
        resid[:, t0] = -resid[:, t0]  # flip the residual stream at one position
        return resid

    with torch.no_grad():
        base = model(batch, return_type="logits")
        hooked = model.run_with_hooks(batch, return_type="logits", fwd_hooks=[(hook_name, scramble)])
        after = model(batch, return_type="logits")  # hooks must be removed again
    before_t0 = (hooked[:, :t0] - base[:, :t0]).abs().max().item()
    from_t0 = (hooked[:, t0:] - base[:, t0:]).abs().max().item()
    at_t0 = (hooked[:, t0] - base[:, t0]).abs().max().item()
    later = (hooked[:, t0 + 1:] - base[:, t0 + 1:]).abs().max().item() if t0 + 1 < P else float("nan")
    leftover = (after - base).abs().max().item()
    # also: ablating one attention head changes the logits
    head_hook = f"blocks.{layer}.attn.hook_z"

    def kill_head(z, hook):
        z = z.clone()
        z[:, :, 0] = 0.0
        return z

    with torch.no_grad():
        ablated = model.run_with_hooks(batch, return_type="logits", fwd_hooks=[(head_hook, kill_head)])
    head_effect = (ablated - base).abs().max().item()
    ok = (len(calls) == 1 and from_t0 > 1e-3 and later > 1e-3 and before_t0 == 0.0
          and leftover == 0.0 and head_effect > 1e-4)
    return ok, {"hook": hook_name, "patched_position": t0, "hook_calls": len(calls),
                "max_diff_before_t0": before_t0, "max_diff_at_t0": at_t0,
                "max_diff_after_t0": later, "max_diff_after_hooks_removed": leftover,
                "head_ablation_hook": f"{head_hook}[head 0]", "head_ablation_max_diff": head_effect}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO_DIR, "data", "small"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--d_mlp", type=int, default=1024)
    p.add_argument("--overfit_sessions", type=int, default=4)
    p.add_argument("--overfit_steps", type=int, default=1500)
    p.add_argument("--overfit_lr", type=float, default=1e-3)
    p.add_argument("--overfit_loss", type=float, default=0.02, help="allowed gap above the floor")
    p.add_argument("--max_len", type=int, default=None, help="truncate test rows (speed on CPU)")
    p.add_argument("--out", default=None, help="optional JSON file for the results")
    args = p.parse_args()

    device = get_device(args.device)
    vocab = load_vocab(args.data)
    pad_id = vocab.index("[PAD]")
    _, hero, opp = action_token_ids(vocab)
    ids = (hero.to(device), opp.to(device))
    train, val = load_tokens(args.data, "train"), load_tokens(args.data, "val")
    cut = (lambda b: b[:, : args.max_len]) if args.max_len else (lambda b: b)
    data = {"overfit": cut(to_batch(train, np.arange(args.overfit_sessions), pad_id, device)),
            "val": cut(to_batch(val, np.arange(2), pad_id, device))}
    cfg = build_config(len(vocab), 1024, args.seed, device, args.n_layers, args.d_model,
                       args.n_heads, args.d_head, args.d_mlp)
    cfg = {k: v for k, v in cfg.items() if k != "device"}

    print(f"device {device} | data {args.data} | arch L{args.n_layers} d{args.d_model} "
          f"h{args.n_heads}x{args.d_head} mlp{args.d_mlp}", flush=True)
    results = {"device": str(device), "data": args.data, "config": cfg}
    t = time.time()
    model, ok1, r1 = check_overfit(args, cfg, data, ids, device)
    results["params"] = count_params(model)
    checks = [("1_overfit_tiny_sample", ok1, r1)]
    checks.append(("2_reload_reproduces_logits", *check_reload(model, cfg, data, device)))
    checks.append(("3_run_with_cache_activations", *check_cache(model, cfg, data)))
    checks.append(("4_run_with_hooks_changes_logits", *check_hooks(model, cfg, data)))
    results["seconds"] = round(time.time() - t, 1)
    all_ok = True
    for name, ok, r in checks:
        all_ok &= ok
        results[name] = {"pass": ok, **r}
        shown = {k: v for k, v in r.items() if k != "trace"}
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {json.dumps(shown)}", flush=True)
    print(f"overfit trace: {r1['trace']}")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED", f"({results['seconds']} s)")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
