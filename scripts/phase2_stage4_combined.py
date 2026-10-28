"""Phase 2 Stage 4: attention heads + MLP units, same confirmatory examples and the same d_c-ablation denominator (docs/plans/phase2_plan.md section 6).
Conditional on gate S3. Usage: python scripts/phase2_stage4_combined.py"""
import json
import time
from contextlib import ExitStack

import numpy as np
import torch
import yaml

from causal_subspaces.heads.ablation import head_ablation
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.heads.evalset import EvalSet, evaluate, generation_check
from causal_subspaces.interventions.neuron_ablation import random_units, to_spec, unit_ablation
from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time()
OUT = ROOT / "results/phase2/stage4"; OUT.mkdir(parents=True, exist_ok=True)
gate = json.load(open(ROOT / "results/phase2/stage3/gate_S3.json"))
if gate["decision"].startswith("stop"):
    raise SystemExit("gate S3 says stop; Stage 4 not run")
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model); m = 12288
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
mu = torch.load(next((ROOT / "cache").glob("ref_Qwen*23*.pt")))["mu"].to(dev); dcol = d_c[:, None]
GAIN = {l: head_gain(model, l, d_c) for l in range(idx_hook)}
NSH = torch.load(ROOT / "results/phase2/neutral_head_stats.pt"); ZBAR = {l: NSH["zbar"][l].to(dev) for l in range(idx_hook)}
MAG = np.stack([NSH["mag"][l].numpy() for l in range(idx_hook)]); GN = np.stack([GAIN[l].view(H, HD).norm(dim=-1).cpu().numpy() for l in range(idx_hook)])
NSU = torch.load(ROOT / "results/phase1/neutral_unit_stats.pt"); ABAR = {l: NSU[l]["mean"].to(dev) for l in (19, 20, 21, 22)}
V1 = np.load(ROOT / "results/phase1/stage1_vectors.npz"); LS = json.load(open(ROOT / "results/phase1/stage1.json"))["selection"]["layers"]
A1 = np.load(ROOT / "results/phase2/stage1_arrays.npz")
order_abs = np.argsort(-np.abs(A1["delta_logic"]).reshape(-1))
heads = lambda k: [(int(i // H), int(i % H)) for i in order_abs[:k]]

def top_units(k):
    D = torch.stack([torch.tensor(V1["logic/mean_acc"][l] - V1["logic/mean_rej"][l]) * torch.tensor(V1["consensus/g"][l]) for l in LS])
    pos = D.abs().reshape(-1).topk(k).indices.numpy(); return np.array(LS)[pos // m], pos % m
CONF = EvalSet(model, tok, cfg, "confirmatory", dev)
S3 = np.load(ROOT / "results/phase2/stage3/conditions.npz", allow_pickle=True)
base_h = S3["intact||_h"]
STORE, SP = {}, OUT / "conditions.npz"
if SP.exists():
    z = np.load(SP, allow_pickle=True); STORE = {k: z[k] for k in z.files}
def run(name, factory):
    if f"{name}||margin" in STORE: return
    o = evaluate(model, tok, CONF, idx_hook, d_c, factory, base_h=base_h, device=dev)
    for k, v in o.items(): STORE[f"{name}||{k}"] = np.asarray(v)
    if len(STORE) % 30 < 8: np.savez_compressed(SP, **STORE)
def spec_h(hs):
    s = {}
    for l, h in hs: s.setdefault(l, []).append(h)
    return s
def combo(hs=None, units=None, heads_all=False, units_all=False):
    def f(rec):
        st = ExitStack()
        if heads_all: st.enter_context(head_ablation(model, {l: list(range(H)) for l in range(19, 23)}, idx_hook, mode="mean", zbar=ZBAR, gains=GAIN, record=rec))
        elif hs: st.enter_context(head_ablation(model, spec_h(hs), idx_hook, mode="mean", zbar=ZBAR, gains=GAIN, record=rec))
        if units_all: st.enter_context(unit_ablation(model, {l: dict(idx=torch.arange(m, device=dev), repl=ABAR[l]) for l in ABAR}, idx_hook))
        elif units is not None: st.enter_context(unit_ablation(model, to_spec(units[0], units[1], ABAR, dev), idx_hook))
        return st
    return f

# ---- references and component-only conditions (attention-only conditions are re-run here so every condition shares the code path) --------------
for k in (8, 16): run(f"attn|k={k}", combo(hs=heads(k)))
run("attn|k=1", combo(hs=heads(1)))
U64, U512 = top_units(64), top_units(512)
run("mlp|k=64", combo(units=U64)); run("mlp|k=512", combo(units=U512))
run("combined|S8+mlp512", combo(heads(8), U512)); run("combined|S8+mlp64", combo(heads(8), U64))
run("combined|S16+mlp512", combo(heads(16), U512)); run("combined|L22H5+mlp512", combo(heads(1), U512)); run("combined|L22H5+mlp64", combo(heads(1), U64))
run("ref|all_heads_19_22", combo(heads_all=True)); run("ref|all_units_19_22", combo(units_all=True)); run("ref|all_heads+all_units_19_22", combo(heads_all=True, units_all=True))
np.savez_compressed(SP, **STORE); print(f"core conditions done ({time.time() - T0:.0f}s)", flush=True)

# ---- generic / mixed controls ----------------------------------------------------------------------------------------------------------------
Ls, Is = U512; cnt = {int(l): int((Ls == l).sum()) for l in np.unique(Ls)}; excl = {int(l): Is[Ls == l] for l in np.unique(Ls)}
def rand_heads(hs, d):
    rng = np.random.default_rng(100_003 * len(hs) + 7919 * d + 17); nat = set(hs); out = []
    for l in sorted({l for l, _ in hs}):
        k = sum(1 for x, _ in hs if x == l); pool = [h for h in range(H) if (l, h) not in nat]; out += [(l, int(h)) for h in rng.choice(pool, k, replace=False)]
    return out
def rand_units(d):
    return random_units(cnt, m, np.random.default_rng(555_000 + d), exclude=excl)
for d in range(20): run(f"ctrl_both|random_heads+random_units512|d={d}", combo(rand_heads(heads(8), d), rand_units(d)))
for d in range(6):   # reduced from 10 (budget): mixed controls
    run(f"ctrl_mixed|S8+random_units512|d={d}", combo(heads(8), rand_units(d)))
    run(f"ctrl_mixed|random_heads+mlp512|d={d}", combo(rand_heads(heads(8), d), U512))
np.savez_compressed(SP, **STORE); print(f"controls done ({time.time() - T0:.0f}s)", flush=True)
gen = {}
for nm, fac in (("S8+mlp512", combo(heads(8), U512)), ("mlp512", combo(units=U512))):
    gen[nm] = generation_check(model, tok, CONF, fac, dev); print("generation", nm, gen[nm]["multiplication_acc"], flush=True)
json.dump(gen, open(OUT / "generation.json", "w"), indent=1)
np.savez_compressed(OUT / "eval_meta.npz", groups=CONF.groups, accept=CONF.accept)
print("done", time.time() - T0)
