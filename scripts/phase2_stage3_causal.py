"""Phase 2 Stage 3: causal head ablations on the CONFIRMATORY set (resumable; budget-tracked; gate S3). docs/phase2_plan.md section 5.
Phases: A (regression, references, native conditions), B (k=8 controls, then gate S3), C (k grid controls + k=1 head-effect map), D (generation checks).
Usage: python scripts/phase2_stage3_causal.py --phase A B C D"""
import argparse
import json
import time
import zlib

import numpy as np
import torch
import yaml

from causal_subspaces.heads.ablation import head_ablation
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.heads.evalset import EvalSet, evaluate, generation_check
from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import load_neutral_corpus_ids
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--phase", nargs="+", default=["A", "B", "C", "D"])
ap.add_argument("--budget-hours", type=float, default=2.6)
a = ap.parse_args()
T0 = time.time()
OUT = ROOT / "results/phase2/stage3"; OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
mu = torch.load(next((ROOT / "cache").glob("ref_Qwen*23*.pt")))["mu"].to(dev)
dcol = d_c[:, None]
LAY = list(range(idx_hook))
GAIN = {l: head_gain(model, l, d_c) for l in LAY}
S1 = json.load(open(ROOT / "results/phase2/stage1.json"))
A1 = np.load(ROOT / "results/phase2/stage1_arrays.npz")

# ---------------- neutral-corpus statistics: z-bar and output magnitude per head --------------------------------------------------------
sp = ROOT / "results/phase2/neutral_head_stats.pt"
if sp.exists():
    NS = torch.load(sp)
else:
    r = cfg["reference"]
    win = load_neutral_corpus_ids(tok, r["dataset"], r["config"], r["split"], 500_000, r["window"])
    s1 = {l: torch.zeros(H * HD, dtype=torch.float64, device=dev) for l in LAY}; mag = {l: torch.zeros(H, dtype=torch.float64, device=dev) for l in LAY}
    gram = {l: torch.stack([(lambda W: W.T @ W)(model.model.layers[l].self_attn.o_proj.weight[:, h * HD:(h + 1) * HD].float()) for h in range(H)]) for l in LAY}
    cnt = [0]
    def mk(l):
        def hook(mod, inp):
            z = inp[0].float().reshape(-1, H, HD)
            s1[l] += z.sum((0,)).reshape(-1).double()
            mag[l] += torch.einsum("nhd,hde,nhe->nh", z, gram[l], z).clamp_min(0).sqrt().sum(0).double()
            if l == 0:
                cnt[0] += z.shape[0]
        return hook
    hs = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(mk(l)) for l in LAY]
    saved = model.model.layers; model.model.layers = saved[:idx_hook]
    with torch.no_grad():
        for s in range(0, len(win), 16):
            model.model(input_ids=win[s:s + 16].to(dev), use_cache=False)
    model.model.layers = saved
    for h in hs: h.remove()
    NS = dict(zbar={l: (s1[l] / cnt[0]).float().cpu() for l in LAY}, mag={l: (mag[l] / cnt[0]).float().cpu() for l in LAY}, n_tokens=cnt[0])
    torch.save(NS, sp)
ZBAR = {l: NS["zbar"][l].to(dev) for l in LAY}
MAG = np.stack([NS["mag"][l].numpy() for l in LAY])              # (L, H)
GN = np.stack([GAIN[l].view(H, HD).norm(dim=-1).cpu().numpy() for l in LAY])   # alignment strength ||W_O,h^T d||
print(f"neutral stats ready ({time.time() - T0:.0f}s)", flush=True)

# ---------------- frozen sets and rankings (training data only) ------------------------------------------------------------------------
sel = json.load(open(ROOT / "results/phase2/selected_heads.json"))
flat_abs = np.abs(A1["delta_logic"]).reshape(-1)
def heads_from(order, k): return [(int(i // H), int(i % H)) for i in order[:k]]
RANK = {"abs": np.argsort(-flat_abs), "signed": np.argsort(-A1["delta_logic"].reshape(-1)), "magnitude": np.argsort(-MAG.reshape(-1)),
        "localizer": np.argsort(-np.abs(A1["delta_localizer"]).reshape(-1)), "prontoqa": np.argsort(-np.abs(A1["delta_prontoqa"]).reshape(-1))}
assert heads_from(RANK["abs"], 8) == [(s["layer"], s["head"]) for s in sel["S8"]], "S8 must equal the frozen selection"
def to_spec(hs):
    spec = {}
    for l, h in hs: spec.setdefault(l, []).append(h)
    return spec

# ---------------- evaluation machinery -----------------------------------------------------------------------------------------------------
EXP = EvalSet(model, tok, cfg, "exploratory", dev)
CONF = EvalSet(model, tok, cfg, "confirmatory", dev)
print("eval sets:", EXP.n, CONF.n, flush=True)
STORE, STORE_PATH = {}, OUT / "conditions.npz"
if STORE_PATH.exists():
    z = np.load(STORE_PATH, allow_pickle=True); STORE = {k: z[k] for k in z.files}
def save(): np.savez_compressed(STORE_PATH, **STORE)
def have(name): return f"{name}||margin" in STORE
def run(name, factory):
    if have(name): return
    base = STORE.get("intact||_h")
    o = evaluate(model, tok, CONF, idx_hook, d_c, factory, base_h=None if name == "intact" else base, device=dev)
    for k, v in o.items(): STORE[f"{name}||{k}"] = np.asarray(v)
    if time.time() - T0 > a.budget_hours * 3600:
        save(); raise SystemExit("budget cap reached; saved")
    if len(STORE) % 30 < 8: save()
def hfac(hs, mode="mean", position="all"):
    spec = to_spec(hs)
    return lambda rec: head_ablation(model, spec, idx_hook, mode=mode, zbar=ZBAR, position=position, gains=GAIN, record=rec)
DC = lambda rec: residual_intervention(model, idx_hook, "ablate", dcol, mu)

# controls ------------------------------------------------------------------------------------------------------------------------------------
def nearest_in_layer(l, val_arr, native_h, rng, used, n_cand=3):
    v = np.log(val_arr[l] + 1e-9)
    cands = np.argsort(np.abs(v - v[native_h]))
    cands = [c for c in cands if (l, int(c)) not in used][:n_cand]
    return int(rng.choice(cands))
def ctrl_sets(family, hs, d):
    rng = np.random.default_rng(100_003 * len(hs) + 7919 * d + zlib.crc32(family.encode()) % 1000)   # deterministic (str hash() is randomised per process)
    nat = set(hs); out, used = [], set(hs)
    if family == "random":                                   # same number of heads per layer, any other head of that layer
        for l in sorted({l for l, _ in hs}):
            k = sum(1 for x, _ in hs if x == l); pool = [h for h in range(H) if (l, h) not in nat]
            take = min(k, len(pool))
            out += [(l, int(h)) for h in rng.choice(pool, take, replace=False)]
            if take < k:   # layer has too few non-native heads (only happens for k = 64): fill the shortfall from other non-native heads of layers 19-22
                extra = [(x, y) for x in range(19, 23) for y in range(H) if (x, y) not in nat and (x, y) not in out]
                out += [extra[i] for i in rng.choice(len(extra), k - take, replace=False)]
    elif family == "layer_restricted":                       # uniform over heads of layers 19-22 not in the native set
        pool = [(l, h) for l in range(19, 23) for h in range(H) if (l, h) not in nat]
        out = [pool[i] for i in rng.choice(len(pool), len(hs), replace=False)]
    elif family in ("magnitude_matched", "alignment_matched"):
        arr = MAG if family == "magnitude_matched" else GN
        for l, h in hs:
            c = nearest_in_layer(l, arr, h, rng, used); used.add((l, c)); out.append((l, c))
    return out

def native(k, rank="abs"): return heads_from(RANK[rank], k)

REG_PATH = OUT / "control_sets.json"
REG = json.load(open(REG_PATH)) if REG_PATH.exists() else {}
if not REG_PATH.exists():   # purge the k=8 controls drawn with the non-deterministic seed; they are re-run and recorded
    for key in [k for k in STORE if k.split('|')[0] in ('random', 'layer_restricted', 'magnitude_matched', 'alignment_matched') and '|k=8|' in k.split('||')[0]]:
        del STORE[key]
def crun(family, k, d):
    hs = ctrl_sets(family, native(k), d); nm = f"{family}|k={k}|d={d}"
    REG[nm] = hs; json.dump(REG, open(REG_PATH, 'w'))
    run(nm, hfac(hs))

# ================================ phases ================================================================================================
if "A" in a.phase:
    # regression on the Phase-1 exploratory set: intact and d_c ablation must reproduce the stored Phase-1 margins
    z1 = np.load(ROOT / "results/phase1/stage4/conditions.npz", allow_pickle=True)
    reg = {}
    e0 = evaluate(model, tok, EXP, idx_hook, d_c, None, device=dev, kl=False)
    e1 = evaluate(model, tok, EXP, idx_hook, d_c, DC, base_h=e0["_h"], device=dev, kl=False)
    reg["intact_max_abs_diff"] = float(np.abs(e0["margin"] - z1["intact||margin"]).max()); reg["dc_max_abs_diff"] = float(np.abs(e1["margin"] - z1["dc_ablation||margin"]).max())
    reg["intact_mean_abs_diff"] = float(np.abs(e0["margin"] - z1["intact||margin"]).mean()); reg["dc_mean_abs_diff"] = float(np.abs(e1["margin"] - z1["dc_ablation||margin"]).mean())
    reg["ok"] = bool(reg["intact_max_abs_diff"] < 1.0 and reg["dc_max_abs_diff"] < 1.0)
    json.dump(reg, open(OUT / "regression.json", "w"), indent=1); print("REGRESSION", reg, flush=True)
    run("intact", None)
    run("dc_ablation", DC)
    S8, S16 = native(8), native(16)
    run("native|S8|mean|all", hfac(S8)); run("native|S16|mean|all", hfac(S16))
    for mode, pos in (("mean", "last"), ("zero", "all"), ("resample", "all")):
        run(f"native|S8|{mode}|{pos}", hfac(S8, mode, pos)); run(f"native|S16|{mode}|{pos}", hfac(S16, mode, pos))
    for k in (1, 2, 4, 32, 64):
        run(f"native|k={k}|rank=abs|mean|all", hfac(native(k)))
    for rank in ("signed", "magnitude", "localizer", "prontoqa"):
        for k in (8, 16):
            run(f"native|k={k}|rank={rank}|mean|all", hfac(native(k, rank)))
    save(); print(f"phase A done ({time.time() - T0:.0f}s)", flush=True)

def gap(n, g):
    m_, ac = STORE[f"{n}||margin"][CONF.groups == g], CONF.accept[CONF.groups == g]; return m_[ac].mean() - m_[~ac].mean()
if "B" in a.phase:
    for fam in ("random", "layer_restricted", "magnitude_matched", "alignment_matched"):
        for d in range(30):
            crun(fam, 8, d)
        save(); print(f"phase B {fam} done ({time.time() - T0:.0f}s)", flush=True)
    R_dc = gap("intact", "logic_num_orig") - gap("dc_ablation", "logic_num_orig")
    E = lambda n: (gap("intact", "logic_num_orig") - gap(n, "logic_num_orig")) / R_dc
    ctrl_max = max(E(n) for n in [k.split("||")[0] for k in STORE if k.endswith("||margin")] if n.split("|")[0] in ("random", "layer_restricted", "magnitude_matched", "alignment_matched") and "k=8" in n)
    g = dict(E_gap_S8_logic_num_orig=float(E("native|S8|mean|all")), E_gap_S16=float(E("native|S16|mean|all")), E_gap_k64=float(E("native|k=64|rank=abs|mean|all")),
             control_max_E_gap_k8=float(ctrl_max), R_dc_logic_num_orig=float(R_dc))
    g["decision"] = ("stop: heads not causal" if (g["E_gap_S8_logic_num_orig"] < 0.05 and g["E_gap_S8_logic_num_orig"] <= ctrl_max) else
                     ("limited stage 4" if g["E_gap_S8_logic_num_orig"] < 0.15 else "full plan"))
    json.dump(g, open(OUT / "gate_S3.json", "w"), indent=1); print("GATE S3", g, flush=True)
    if g["decision"].startswith("stop"):
        a.phase = [p for p in a.phase if p not in ("C", "D")]

if "C" in a.phase:
    for k in (2, 4, 16, 32):
        for fam in ("random", "layer_restricted", "magnitude_matched"):
            for d in range(10): crun(fam, k, d)
        save(); print(f"phase C k={k} done ({time.time() - T0:.0f}s)", flush=True)
    for fam in ("random", "layer_restricted", "magnitude_matched"):
        for d in range(5): crun(fam, 64, d)
    for l in range(19, 23):                                  # k = 1 head-effect map over layers 19-22
        for h in range(H): run(f"map|L{l}H{h}", hfac([(l, h)]))
        save(); print(f"phase C map layer {l} done ({time.time() - T0:.0f}s)", flush=True)

if "D" in a.phase:
    gen = {}
    for name, fac in (("intact", None), ("dc_ablation", DC), ("S8", hfac(native(8))), ("S16", hfac(native(16)))):
        gen[name] = generation_check(model, tok, CONF, fac, dev)
        print("generation", name, gen[name]["multiplication_acc"], flush=True)
    json.dump(gen, open(OUT / "generation.json", "w"), indent=1)
save()
json.dump(dict(n_conditions=len({k.split("||")[0] for k in STORE}), seconds=time.time() - T0, groups={g: int((CONF.groups == g).sum()) for g in np.unique(CONF.groups)}), open(OUT / "meta.json", "w"), indent=1)
np.savez_compressed(OUT / "eval_meta.npz", groups=CONF.groups, accept=CONF.accept)
print("done", time.time() - T0)
