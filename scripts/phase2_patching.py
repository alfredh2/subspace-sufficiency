"""Phase 2 Stage 3 (patching): counterfactual patching of head OUTPUTS z_h (full-z only; A-only / V-only patching and routing/content are cancelled because
the attention-recompute gate failed). Aligned pairs only for all-position patches (docs/phase2_plan.md section 4). Reports direct (immediate) and downstream
(after recomputation) changes of the verdict coordinate separately. Usage: python scripts/phase2_patching.py"""
import json
import time
import zlib

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.heads.ablation import capture_z, head_z_patch
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time()
OUT = ROOT / "results/phase2/patching"; OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
GAIN = {l: head_gain(model, l, d_c) for l in range(idx_hook)}
NS = torch.load(ROOT / "results/phase2/neutral_head_stats.pt")
MAG = np.stack([NS["mag"][l].numpy() for l in range(idx_hook)]); GN = np.stack([GAIN[l].view(H, HD).norm(dim=-1).cpu().numpy() for l in range(idx_hook)])
A1 = np.load(ROOT / "results/phase2/stage1_arrays.npz")
order_abs = np.argsort(-np.abs(A1["delta_logic"]).reshape(-1))
native = lambda k: [(int(i // H), int(i % H)) for i in order_abs[:k]]
splits = get_splits(cfg); contents = build_contents("ordinary")
test_ids = sorted({e.content_id for e in splits["items"]})
fams = TRAIN_FAMILIES + HELDOUT_FAMILIES

def nearest(l, arr, h, rng, used, n_cand=3):
    v = np.log(arr[l] + 1e-9); c = [x for x in np.argsort(np.abs(v - v[h])) if (l, int(x)) not in used][:n_cand]; return int(rng.choice(c))
def ctrl(family, hs, d):
    rng = np.random.default_rng(100_003 * len(hs) + 7919 * d + zlib.crc32(family.encode()) % 1000); nat = set(hs); used = set(hs); out = []
    if family == "random":
        for l in sorted({l for l, _ in hs}):
            k = sum(1 for x, _ in hs if x == l); pool = [h for h in range(H) if (l, h) not in nat]; out += [(l, int(h)) for h in rng.choice(pool, min(k, len(pool)), replace=False)]
    elif family == "layer_restricted":
        pool = [(l, h) for l in range(19, 23) for h in range(H) if (l, h) not in nat]; out = [pool[i] for i in rng.choice(len(pool), len(hs), replace=False)]
    else:
        arr = MAG if family == "magnitude_matched" else GN
        for l, h in hs:
            c = nearest(l, arr, h, rng, used); used.add((l, c)); out.append((l, c))
    return out
def spec_of(hs):
    s = {}
    for l, h in hs: s.setdefault(l, []).append(h)
    return s

def head_sets(n_draw):
    sets = {"S8": native(8), "S16": native(16), "L22H5": native(1)}
    for fam in ("random", "layer_restricted", "magnitude_matched"):
        for d in range(n_draw): sets[f"{fam}|d={d}"] = ctrl(fam, native(8), d)
    summary["control_sets"].update({k: v for k, v in sets.items()})
    return sets

@torch.no_grad()
def run_trials(trials, aff, rej, sets, positions, bs=16):
    ia, ir = P5.single_token_id(tok, aff), P5.single_token_id(tok, rej)
    n = len(trials); layers = sorted({l for hs in sets.values() for l, _ in hs})
    res = {k: np.zeros(n) for k in ("m_t", "m_s", "c_t", "c_s")}; res["v_t"] = np.empty(n, "<U5"); res["v_s"] = np.empty(n, "<U5")
    for nm in sets:
        for p in positions:
            for k in ("m", "c", "direct"): res[f"{k}|{nm}|{p}"] = np.full(n, np.nan)
            res[f"v|{nm}|{p}"] = np.full(n, "", "<U5")
    ok_all = np.array([t.get("aligned", True) for t in trials])
    verdict = lambda lg: ["aff" if tok.decode([g])[:1] == aff[0] else "rej" if tok.decode([g])[:1] == rej[0] else "other" for g in lg.argmax(-1).tolist()]
    for s0 in range(0, n, bs):
        b = trials[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev)
        ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
        zs, zt, cs, ct = {}, {}, [], []
        with capture_z(model, layers, zs), capture_residual(model, idx_hook, cs):
            lg_s = last_logits(model, ids_s, m_s, p_s)
        with capture_z(model, layers, zt), capture_residual(model, idx_hook, ct):
            lg_t = last_logits(model, ids_t, m_t, p_t)
        res["m_s"][sl] = (lg_s[:, ia] - lg_s[:, ir]).cpu().numpy(); res["m_t"][sl] = (lg_t[:, ia] - lg_t[:, ir]).cpu().numpy()
        res["v_s"][sl] = verdict(lg_s); res["v_t"][sl] = verdict(lg_t)
        res["c_s"][sl] = (cs[0][:, -1].float() @ d_c).cpu().numpy(); res["c_t"][sl] = (ct[0][:, -1].float() @ d_c).cpu().numpy()
        same_shape = ids_t.shape == ids_s.shape
        for nm, hs in sets.items():
            spec = spec_of(hs)
            for p in positions:
                if p == "all" and (not same_shape or not ok_all[sl].all()):
                    continue
                # immediate (direct) effect of the swap on the verdict coordinate at the last position
                direct = sum(((zs[l].float().view(len(b), -1, H, HD)[:, -1, h] - zt[l].float().view(len(b), -1, H, HD)[:, -1, h]) * GAIN[l].view(H, HD)[h]).sum(-1) for l, h in hs)
                cp = []
                with head_z_patch(model, spec, zs, idx_hook, position=p), capture_residual(model, idx_hook, cp):
                    lg = last_logits(model, ids_t, m_t, p_t)
                res[f"m|{nm}|{p}"][sl] = (lg[:, ia] - lg[:, ir]).cpu().numpy(); res[f"v|{nm}|{p}"][sl] = verdict(lg)
                res[f"c|{nm}|{p}"][sl] = (cp[0][:, -1].float() @ d_c).cpu().numpy(); res[f"direct|{nm}|{p}"][sl] = direct.cpu().numpy()
    return res

def fmt_prompt(fmt, syl): return format_prompt(tok, P5.build_prompt(fmt, syl), False)
summary, raw = {"groups": {}, "control_sets": {}}, {}
def go(group, trials, aff, rej, sets, positions):
    t = time.time(); res = run_trials(trials, aff, rej, sets, positions)
    ta, sa = np.array([x["tgt_aff"] for x in trials]), np.array([x["src_aff"] for x in trials])
    out = {}
    for nm in sets:
        for p in positions:
            out[f"{nm}|{p}"] = P5.summarize_patching(res, ta, sa, nm, p)
            mp, cp_, dr = res[f"m|{nm}|{p}"], res[f"c|{nm}|{p}"], res[f"direct|{nm}|{p}"]
            ok = ~np.isnan(mp); diff = (ta != sa) & ok
            if diff.any():
                sgn = np.where(sa[diff], 1.0, -1.0)
                gc = (sgn * (res["c_s"][diff] - res["c_t"][diff])).mean()
                out[f"{nm}|{p}"].update(c_gap=float(gc), c_shift=float((sgn * (cp_[diff] - res["c_t"][diff])).mean()), c_transfer=float((sgn * (cp_[diff] - res["c_t"][diff])).mean() / gc),
                                        direct_shift=float((sgn * dr[diff]).mean()), direct_transfer=float((sgn * dr[diff]).mean() / gc))
    summary["groups"][group] = dict(n_trials=len(trials), n_aligned=int(sum(x.get("aligned", True) for x in trials)), results=out)
    for k, v in res.items(): raw[f"{group}|{k}"] = v
    raw[f"{group}|tgt_aff"], raw[f"{group}|src_aff"] = ta, sa
    json.dump(summary, open(OUT / "patching.json", "w"), indent=1); np.savez_compressed(OUT / "patching_arrays.npz", **raw)
    r = out[f"S8|{positions[0]}"]
    print(f"{group}: {time.time() - t:.0f}s S8|{positions[0]} transfer={r.get('transfer', float('nan')):+.3f} flip={r.get('flip_to_source', float('nan')):.3f} n_diff={r.get('n_diff')}", flush=True)

# ---- matched pairs, alignment-validated -----------------------------------------------------------------------------------------------------
pairs = P5.build_pairs(contents, test_ids, fams, tok)
align = {}
for fmt in ("num_orig", "num_rev"):
    ok = []
    for p in pairs:
        a_ok, why = A.pair_alignment(tok, fmt_prompt(fmt, p["valid_syl"]), p["valid_syl"].split("\n"), fmt_prompt(fmt, p["invalid_syl"]), p["invalid_syl"].split("\n"))
        ok.append((a_ok, why))
    align[fmt] = ok
    reasons = {}
    for a_ok, why in ok: reasons[why] = reasons.get(why, 0) + 1
    print(f"{fmt}: {sum(a for a, _ in ok)}/{len(pairs)} pairs pass alignment; reasons {reasons}", flush=True)
summary["alignment"] = {fmt: dict(n_pairs=len(pairs), n_valid=int(sum(a for a, _ in ok)), reasons={w: sum(1 for _, x in ok if x == w) for w in {x for _, x in ok}}) for fmt, ok in align.items()}
for fmt, nd in (("num_orig", 30), ("num_rev", 6)):   # num_rev draws reduced from 10 (budget)
    f = P5.FORMATS[fmt]; trials = []
    for p, (a_ok, _) in zip(pairs, align[fmt]):
        v, i = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        trials += [dict(tgt=i, src=v, tgt_aff=False, src_aff=True, aligned=a_ok), dict(tgt=v, src=i, tgt_aff=True, src_aff=False, aligned=a_ok)]
    go(f"match_{fmt}", trials, f["valid"], f["invalid"], head_sets(nd), ("all", "last"))
# same-label lexical controls (S8 + 10 random/magnitude-matched draws)
ctl = P5.lexical_controls(contents, pairs, tok)
f = P5.FORMATS["num_orig"]; ltr = []
for c in ctl:
    b_, o_ = c["base"], c["other"]
    for key, y in (("valid_syl", True), ("invalid_syl", False)):
        a_ok, _ = A.pair_alignment(tok, fmt_prompt("num_orig", b_[key]), b_[key].split("\n"), fmt_prompt("num_orig", o_[key]), o_[key].split("\n"))
        ltr.append(dict(tgt=P5.build_prompt("num_orig", b_[key]), src=P5.build_prompt("num_orig", o_[key]), tgt_aff=y, src_aff=y, aligned=a_ok))
go("lexctrl_num_orig", ltr, f["valid"], f["invalid"], {k: v for k, v in head_sets(10).items() if k in ("S8", "L22H5") or "d=" in k and int(k.split("d=")[1]) < 10}, ("all", "last"))
# non-verdict / other tasks, unmatched pairs, last position only
c5 = build_controls(5, 128, tuple(cfg["controls"]["mmlu_subjects"]), cfg["controls"]["mmlu_per_subject"])
rng = np.random.default_rng(0)
def unmatched(items, n=150):
    a_ = [t for t, g in items if g == "1"]; r_ = [t for t, g in items if g == "2"]; tr = []
    for i, j in zip(rng.choice(len(a_), n), rng.choice(len(r_), n)):
        tr += [dict(tgt=r_[j], src=a_[i], tgt_aff=False, src_aff=True, aligned=False), dict(tgt=a_[i], src=r_[j], tgt_aff=True, src_aff=False, aligned=False)]
    return tr
for name, key in (("language_localizer", "language_localizer"), ("prontoqa", "prontoqa"), ("mmlu_binary", "mmlu_binary"), ("stroop", "stroop"), ("word_recall_48", "word_recall_48")):
    items = [(x[0], x[1]) for x in c5[key]["items"][:128]]
    go(f"unmatched_{name}", unmatched(items), "1", "2", {k: v for k, v in head_sets(10).items()}, ("last",))
lg = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
items = [(P5.build_prompt("num_orig", e.syllogism), "1" if e.valid else "2") for e in lg]
go("unmatched_logic", unmatched(items), "1", "2", head_sets(10), ("last",))
print("done", time.time() - T0)
