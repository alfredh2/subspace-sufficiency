"""Phase 1 Stage 4: causal ablation of top-k native MLP units (mean replacement, all positions) vs random / activation-matched /
alignment-matched / rotated-coordinate controls. Resumable; budget-tracked; early-stop gate per docs/phase1_plan.md section 3.
Everything is stored per example so effects, bootstrap CIs and matched-strength comparisons are computed in the report script.
Usage: python scripts/phase1_stage4_causal.py [--phase A B C D]"""
import argparse
import json
import time

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import build_contents
from causal_subspaces.experiment import get_controls, get_lm_passages, get_splits
from causal_subspaces.interventions.neuron_ablation import AblationRecord, matched_units, random_units, to_spec, unit_ablation
from causal_subspaces.models.hooks import capture_residual, residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.mvs.reference import load_neutral_corpus_ids
from causal_subspaces.neurons import concentration as C
from causal_subspaces.neurons.decompose import layerwise_pass, unit_gain
from causal_subspaces.rotations import haar as H
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--phase", nargs="+", default=["A", "B", "C", "D"])
ap.add_argument("--budget-hours", type=float, default=3.0, help="hard cap for THIS script (pilot ceiling is 5 GPU-hours in total)")
a = ap.parse_args()
T0 = time.time()
OUT = ROOT / "results" / "phase1" / "stage4"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
S1 = json.load(open(ROOT / "results/phase1/stage1.json")); LS = S1["selection"]["layers"]
dirs = {k: v.to(dev) for k, v in torch.load(ROOT / "results/phase1/directions.pt").items()}
V1 = np.load(ROOT / "results/phase1/stage1_vectors.npz")
m, K_GRID = 12288, [4, 16, 64, 256, 512, 2458]
model, tok, rev = load_model(cfg["model"], dev)
splits, controls = get_splits(cfg), get_controls(cfg)
mu = torch.load(next((ROOT / "cache").glob("ref_Qwen*23*.pt")))["mu"].to(dev) if list((ROOT / "cache").glob("ref_Qwen*23*.pt")) else None
GAIN = {dn: {l: unit_gain(model, l, d) for l in LS} for dn, d in dirs.items()}

# ------------------------------------------------ neutral-corpus unit statistics (replacement values, matching features) ----
stat_path = ROOT / "results/phase1/neutral_unit_stats.pt"
if stat_path.exists():
    NS = torch.load(stat_path)
else:
    r = cfg["reference"]
    win = load_neutral_corpus_ids(tok, r["dataset"], r["config"], r["split"], 500_000, r["window"])
    s1 = {l: torch.zeros(m, dtype=torch.float64, device=dev) for l in LS}; sa = {l: torch.zeros(m, dtype=torch.float64, device=dev) for l in LS}
    s2 = {l: torch.zeros(m, dtype=torch.float64, device=dev) for l in LS}
    ntok = [0]
    def mk(l):
        def h(mod, inp):
            x = inp[0].float().reshape(-1, m)
            s1[l] += x.sum(0).double(); sa[l] += x.abs().sum(0).double(); s2[l] += (x ** 2).sum(0).double()
            if l == LS[0]:
                ntok[0] += x.shape[0]
        return h
    hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in LS]
    saved = model.model.layers; model.model.layers = saved[:max(LS) + 1]
    with torch.no_grad():
        for s in range(0, len(win), 16):
            model.model(input_ids=win[s:s + 16].to(dev), use_cache=False)
    model.model.layers = saved
    for h in hs: h.remove()
    NS = {l: dict(mean=(s1[l] / ntok[0]).float().cpu(), abs=(sa[l] / ntok[0]).float().cpu(), sq=(s2[l] / ntok[0]).float().cpu()) for l in LS}
    NS["n_tokens"] = ntok[0]
    torch.save(NS, stat_path)
print(f"neutral stats ready ({time.time() - T0:.0f}s)", flush=True)
ABAR = {l: NS[l]["mean"].to(dev) for l in LS}

# ------------------------------------------------ rankings (training data only) ----------------------------------------------
def delta_pool(task, dname):
    task = {"localizer": "language_localizer"}.get(task, task)   # key names used in stage1_vectors.npz
    out = []
    for l in LS:
        d = torch.tensor(V1[f"{task}/mean_acc"][l] - V1[f"{task}/mean_rej"][l])
        out.append(d * torch.tensor(V1[f"{dname}/g"][l]) if f"{dname}/g" in V1 else d * GAIN[dname][l].cpu())
    return torch.stack(out)                                                     # (len(LS), m)

def top_units(Dpool, k):
    flat = Dpool.abs().reshape(-1)
    pos = flat.topk(k).indices.cpu().numpy()
    return np.array(LS)[pos // m], pos % m

# ------------------------------------------------ evaluation set (held-out subsets) ---------------------------------------------
rng = np.random.default_rng(0)
def strat(sp, n):
    ex = splits[sp]; v = [e for e in ex if e.valid]; i = [e for e in ex if not e.valid]
    return [v[j] for j in rng.choice(len(v), n // 2, replace=False)] + [i[j] for j in rng.choice(len(i), n // 2, replace=False)]
ev = strat("items", 128) + strat("families", 128) + strat("nonwords", 128)
rows, tags = [], []   # (text, ta, tb, accept, group)
def add(group, texts, ta, tb, acc):
    for t, y in zip(texts, acc):
        rows.append((t, ta, tb, bool(y), group))
for fmt in ("num_orig", "num_rev"):
    f = P5.FORMATS[fmt]
    add(f"logic_{fmt}", [P5.build_prompt(fmt, e.syllogism) for e in ev], P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"]), [e.valid for e in ev])
t1, t2 = P5.single_token_id(tok, "1"), P5.single_token_id(tok, "2")
for name, key, n in (("localizer", "language_localizer", 128), ("prontoqa", "prontoqa", 128), ("mmlu_binary", "mmlu_binary", 96), ("stroop", "stroop", 96), ("word_recall_48", "word_recall_48", 96)):
    it = controls[key]["items"][:n]
    add(name, [x[0] for x in it], t1, t2, [x[1] == "1" for x in it])
GROUPS = np.array([r[4] for r in rows]); ACCEPT = np.array([r[3] for r in rows])
PROMPTS = [format_prompt(tok, r[0], False) for r in rows]
TA = torch.tensor([r[1] for r in rows], device=dev); TB = torch.tensor([r[2] for r in rows], device=dev)
ORDER = sorted(range(len(PROMPTS)), key=lambda i: len(PROMPTS[i]))
passages = get_lm_passages(tok, cfg)[:32]
dcol = dirs["consensus"][:, None]
if mu is None:
    from causal_subspaces.experiment import get_reference
    mu = get_reference(model, tok, cfg, rev, dev)["mu"].to(dev)
print(f"{len(rows)} eval prompts", flush=True)

@torch.no_grad()
def evaluate(ctx_factory, gains=None, base=None):
    """ctx_factory(record) -> context manager (or None for intact). Returns per-example arrays."""
    N = len(PROMPTS)
    out = dict(margin=np.zeros(N, np.float32), c=np.zeros(N, np.float32), direct=np.full(N, np.nan, np.float32), dh=np.zeros(N, np.float32), lpm=np.zeros(N, np.float32))
    h_store = out.setdefault("_h", None)
    hs = np.zeros((N, 4096), np.float32) if base is None else None
    for s in range(0, N, 32):
        idx = ORDER[s:s + 32]
        ids, mask, pos = encode(tok, [PROMPTS[i] for i in idx], dev)
        sink, rec = [], AblationRecord()
        ctx = ctx_factory(rec) if ctx_factory else None
        with capture_residual(model, idx_hook, sink):
            if ctx is None:
                lg = last_logits(model, ids, mask, pos)
            else:
                with ctx:
                    lg = last_logits(model, ids, mask, pos)
        ta, tb = TA[idx], TB[idx]
        la, lb = lg.gather(1, ta[:, None])[:, 0], lg.gather(1, tb[:, None])[:, 0]
        h = sink[0][:, -1].float()
        out["margin"][idx] = (la - lb).cpu().numpy()
        out["lpm"][idx] = (torch.logsumexp(torch.stack([la, lb], 1), 1) - torch.logsumexp(lg, -1)).cpu().numpy()
        out["c"][idx] = (h @ dirs["consensus"]).cpu().numpy()
        if base is not None:
            out["dh"][idx] = (h - torch.tensor(base["_h"][idx], device=dev)).norm(dim=-1).cpu().numpy()
        else:
            hs[idx] = h.cpu().numpy()
        td = rec.total_direct()
        if td is not None:
            out["direct"][idx] = td.cpu().numpy()
    if base is None:
        out["_h"] = hs
    else:
        out.pop("_h", None)
    # WikiText KL
    kl = 0.0
    if ctx_factory is not None:
        for s in range(0, len(passages), 4):
            ids = passages[s:s + 4].to(dev)
            clean = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 32:].float(), -1)
            with ctx_factory(AblationRecord()):
                lg2 = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 32:].float(), -1)
            kl += float((clean.exp() * (clean - lg2)).sum(-1).mean()) / (len(passages) // 4)
    out["kl"] = np.float32(kl)
    return out

STORE, STORE_PATH = {}, OUT / "conditions.npz"
if STORE_PATH.exists():
    z = np.load(STORE_PATH, allow_pickle=True)
    STORE = {k: z[k] for k in z.files}
INTACT_KEY = "intact"
def save():
    np.savez_compressed(STORE_PATH, **STORE)

def put(name, out):
    for k, v in out.items():
        STORE[f"{name}||{k}"] = np.asarray(v)

def have(name):
    return f"{name}||margin" in STORE

def run(name, factory, gains=None):
    if have(name):
        return
    base = {"_h": STORE[f"{INTACT_KEY}||_h"]} if INTACT_KEY != name else None
    put(name, evaluate(factory, gains, base))
    if time.time() - T0 > a.budget_hours * 3600:
        save(); raise SystemExit("budget cap reached; saved")
    if len(STORE) % 40 < 8:
        save()

def native_factory(layers, idx, dname, repl="mean", pos="all"):
    spec = to_spec(np.asarray(layers), np.asarray(idx), ABAR if repl == "mean" else {l: torch.zeros(m) for l in LS}, dev)
    return lambda rec: unit_ablation(model, spec, idx_hook, gains=GAIN[dname], position=pos, record=rec)

def rotated_factory(spec_rot, dname, pos="all"):
    return lambda rec: unit_ablation(model, spec_rot, idx_hook, gains=GAIN[dname], position=pos, record=rec)

feat_act = {l: torch.stack([torch.log(NS[l]["abs"] + 1e-6), torch.log(NS[l]["sq"] + 1e-9)], 1).numpy() for l in LS}
feat_act = {l: (f - f.mean(0)) / f.std(0) for l, f in feat_act.items()}
def feat_align(dname):
    out = {}
    for l in LS:
        f = torch.log(GAIN[dname][l].abs().cpu() + 1e-9).numpy()[:, None]
        out[l] = (f - f.mean()) / f.std()
    return out
def layer_counts(L_, ix):
    return {int(l): int((np.asarray(L_) == l).sum()) for l in np.unique(L_)}

def rotated_set(dname, task, k, r, mode):
    """True-Haar rotated coordinates. mode='top': top-k by pooled |Delta_rot| (rotated analogue of the native ranking); 'random': same per-layer counts at random."""
    dl = {l: torch.tensor(V1[f"{task}/mean_acc"][l] - V1[f"{task}/mean_rej"][l], device=dev) for l in LS}
    Dr = {}
    for l in LS:
        R = H.haar_orthogonal(m, 10_000 * (l + 1) + r, dev)
        Dr[l] = (R @ dl[l]) * (R @ GAIN[dname][l]); del R
    flat = torch.cat([Dr[l].abs() for l in LS])
    pos = flat.topk(k).indices.cpu().numpy()
    counts = {l: int(((pos // m) == i).sum()) for i, l in enumerate(LS)}
    sel = {l: (pos[(pos // m) == i] % m) for i, l in enumerate(LS)}
    if mode == "random":
        g = np.random.default_rng(777 * (r + 1) + k)
        sel = {l: g.choice(m, counts[l], replace=False) for l in LS}
    spec = {}
    for l in LS:
        if counts[l] == 0:
            continue
        R = H.haar_orthogonal(m, 10_000 * (l + 1) + r, dev)
        spec[l] = dict(RS=R[torch.as_tensor(sel[l], device=dev)].clone(), abar=ABAR[l]); del R
    return spec

# ================================================ phases =========================================================================
Dc = delta_pool("logic", "consensus")
phaseA_done = False
if "A" in a.phase:
    run(INTACT_KEY, None)
    run("dc_ablation", lambda rec: residual_intervention(model, idx_hook, "ablate", dcol, mu))
    for k in K_GRID:
        L_, I_ = top_units(Dc, k)
        run(f"native|consensus|k={k}|mean|all", native_factory(L_, I_, "consensus"))
    L_, I_ = top_units(Dc, 512)
    run("native|consensus|k=512|mean|last", native_factory(L_, I_, "consensus", pos="last"))
    for k in (64, 512):
        L_, I_ = top_units(Dc, k)
        run(f"native|consensus|k={k}|zero|all", native_factory(L_, I_, "consensus", repl="zero"))
        Ea2 = torch.stack([NS[l]["sq"] for l in LS]); flat = Ea2.reshape(-1); pos = flat.topk(k).indices.numpy()
        run(f"actonly|consensus|k={k}|mean|all", native_factory(np.array(LS)[pos // m], pos % m, "consensus"))
    for k in (4, 16, 256, 2458):
        L_, I_ = top_units(Dc, k)
        Ea2 = torch.stack([NS[l]["sq"] for l in LS]); pos = Ea2.reshape(-1).topk(k).indices.numpy()
        run(f"actonly|consensus|k={k}|mean|all", native_factory(np.array(LS)[pos // m], pos % m, "consensus"))
    save(); print(f"phase A done ({time.time() - T0:.0f}s)", flush=True)

def controls_for(k, n_rand, n_act, n_ali, n_rt, n_rr, dname="consensus", task="logic", Dpool=None):
    Dp = Dc if Dpool is None else Dpool
    L_, I_ = top_units(Dp, k)
    cnt = layer_counts(L_, I_)
    excl = {int(l): I_[L_ == l] for l in np.unique(L_)}
    tag = f"{dname}|k={k}"
    for d in range(n_rand):
        g = np.random.default_rng(10_000 * k + d)
        rl, ri = random_units(cnt, m, g, exclude=excl)
        run(f"random|{tag}|d={d}", native_factory(rl, ri, dname))
    for d in range(n_act):
        g = np.random.default_rng(20_000 * k + d)
        ml, mi = matched_units(L_, I_, feat_act, g, n_cand=5)
        run(f"actmatched|{tag}|d={d}", native_factory(ml, mi, dname))
    fa = feat_align(dname)
    for d in range(n_ali):
        g = np.random.default_rng(30_000 * k + d)
        ml, mi = matched_units(L_, I_, fa, g, n_cand=5)
        run(f"alignmatched|{tag}|d={d}", native_factory(ml, mi, dname))
    for r in range(n_rt):
        run(f"rotated_top|{tag}|r={r}", rotated_factory(rotated_set(dname, task, k, r, "top"), dname))
    for r in range(n_rr):
        run(f"rotated_random|{tag}|r={r}", rotated_factory(rotated_set(dname, task, k, r, "random"), dname))

def gap_of(name, group):
    mg, acc = STORE[f"{name}||margin"][GROUPS == group], ACCEPT[GROUPS == group]
    return mg[acc].mean() - mg[~acc].mean()
def E_gap(name, group):
    d0, ddc = gap_of(INTACT_KEY, group), gap_of("dc_ablation", group)
    return (d0 - gap_of(name, group)) / (d0 - ddc)

if "B" in a.phase:
    for k in (512, 64):
        controls_for(k, 20, 20, 20, 10, 10)
        save(); print(f"phase B k={k} done ({time.time() - T0:.0f}s)", flush=True)
    e5 = E_gap("native|consensus|k=2458|mean|all", "logic_num_orig")
    ctrl_max = max(E_gap(n, "logic_num_orig") for n in [x.split("||")[0] for x in STORE if x.endswith("||margin")] if n.startswith(("random|consensus|k=512", "actmatched|consensus|k=512", "alignmatched|consensus|k=512", "rotated_top|consensus|k=512")))
    gate = dict(E_gap_top5pct_logic=float(e5), E_gap_top1pct_logic=float(E_gap("native|consensus|k=512|mean|all", "logic_num_orig")), control_max_E_gap_k512=float(ctrl_max))
    early_stop = bool(e5 < 0.10 and e5 <= ctrl_max)
    gate["early_stop"] = early_stop
    json.dump(gate, open(OUT / "early_stop_gate.json", "w"), indent=1)
    print("EARLY-STOP GATE", gate, flush=True)
else:
    early_stop = False

if "C" in a.phase:
    for k in (4, 16, 256, 2458):
        controls_for(k, 6, 6, 0, 4, 4)
        save(); print(f"phase C k={k} done ({time.time() - T0:.0f}s)", flush=True)

if "D" in a.phase and not early_stop:
    # replication: per-seed shared directions (own gains, own top-k), and cross-task rankings
    for dn in ("shared_s0", "shared_s1", "shared_s2"):
        Dd = delta_pool("logic", dn)
        L_, I_ = top_units(Dd, 512)
        run(f"native|{dn}|k=512|mean|all", native_factory(L_, I_, dn))
        controls_for(512, 10, 10, 0, 0, 0, dname=dn, Dpool=Dd)
        save(); print(f"phase D {dn} done ({time.time() - T0:.0f}s)", flush=True)
    for task in ("localizer", "prontoqa"):
        Dt = delta_pool(task, "consensus")
        L_, I_ = top_units(Dt, 512)
        run(f"native_rankedby_{task}|consensus|k=512|mean|all", native_factory(L_, I_, "consensus"))
save()
json.dump(dict(groups={g: int((GROUPS == g).sum()) for g in np.unique(GROUPS)}, K_GRID=K_GRID, layers=LS, n_conditions=len({k.split("||")[0] for k in STORE}), seconds=time.time() - T0),
          open(OUT / "meta.json", "w"), indent=1)
np.savez_compressed(OUT / "eval_meta.npz", groups=GROUPS, accept=ACCEPT)
print("done", time.time() - T0)
