"""Phase 8 S2 compensation test (protocol section 6 + addendum S2-b). --mode dev: gate C1' on development data only (writes comp_dev_h{h}.json).
--mode conf: block test C2 on confirm items for the dev-selected sets (run only after comp_dev is committed)."""
import argparse, json, pickle
from contextlib import ExitStack
import numpy as np, torch
from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import encode, last_logits
from causal_subspaces.phase7.stage1 import class_gap, cluster_boot, texts_of
from causal_subspaces.phase8.s2lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--hook", type=int, default=25); ap.add_argument("--mode", required=True, choices=["dev", "conf"]); a = ap.parse_args()
h = a.hook; w = World(h); ctrl = torch.load(R2 / f"controls_h{h}.pt"); M = json.load(open(R2 / f"match_h{h}.json"))
COMPS = [(l, wh) for l in range(h, 36) for wh in ("attn", "mlp")]
CI = {c: i for i, c in enumerate(COMPS)}
gw = w.norm.weight.float()


def sub(l, wh):
    return w.model.model.layers[l].self_attn if wh == "attn" else w.model.model.layers[l].mlp


@torch.no_grad()
def run_comp(items, U=None, zero=(), pin=None, capture=False, bs=32):
    """Answer-position block-35 output hL (N,d) float32 [+ the 22 component outputs at the answer position] with an optional carrier ablation,
    zeroed components and components pinned to values pin[(l,wh)] (N,d)."""
    texts = texts_of(w.tok, items); N = len(items); order = sorted(range(N), key=lambda i: len(texts[i]))
    hL = np.zeros((N, 4096), np.float32); comps = np.zeros((N, len(COMPS), 4096), np.float32) if capture else None
    for s in range(0, N, bs):
        ii = order[s:s + bs]; ids, mask, pos = encode(w.tok, [texts[i] for i in ii], "cuda"); got = {}; sink36 = []
        with ExitStack() as st:
            if U is not None:
                st.enter_context(residual_intervention(w.model, h, "ablate", U, w.mu))
            for c in COMPS:
                def hook(_m, _i, o, c=c):
                    x = o[0] if isinstance(o, tuple) else o
                    if capture:
                        got[c] = x[:, -1].float().cpu().numpy()
                    if c in zero or (pin is not None and c in pin):
                        new = x.clone()
                        new[:, -1] = 0 if c in zero else torch.tensor(pin[c][ii], device=x.device, dtype=x.dtype)
                        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                    return o
                st.enter_context(_h(sub(*c), hook))
            def h36(_m, _i, o):
                sink36.append((o[0] if isinstance(o, tuple) else o)[:, -1].float())
            st.enter_context(_h(w.model.model.layers[35], h36))
            last_logits(w.model, ids, mask, pos)
        hL[ii] = sink36[0].cpu().numpy()
        if capture:
            for c in COMPS:
                comps[ii, CI[c]] = got[c]
    return hL, comps


from contextlib import contextmanager


@contextmanager
def _h(mod, fn):
    hd = mod.register_forward_hook(fn)
    try:
        yield
    finally:
        hd.remove()


def logits_of(hL, rms=None):
    x = torch.tensor(hL, device="cuda")
    r = torch.sqrt(x.pow(2).mean(-1) + w.norm.variance_epsilon) if rms is None else torch.tensor(rms, device="cuda")
    return ((x / r[:, None]) * gw) @ w.Wc.T, r


def margin(hL, gold, rms=None):
    lg, r = logits_of(hL, rms); lg = lg.cpu().numpy()
    g = lg[np.arange(len(gold)), gold]; oth = (lg.sum(1) - g) / 3
    return g - oth, lg, r.cpu().numpy()


def own_rms(hL):
    return np.sqrt((hL.astype(np.float64) ** 2).mean(-1) + w.norm.variance_epsilon).astype(np.float32)


if a.mode == "dev":
    items = w.dev; gold = np.array([i.gold for i in items]); out = dict(hook=h, chains={})
    hI, cI = run_comp(items, capture=True); mI, _, _ = margin(hI, gold); Gm = float(mI.mean())
    ci_out = {}
    for s, Us in w.chains.items():
        U14 = w.union(Us, range(1, 5)); te14 = M["dev_carrier_te"][str(s)]["cum4"][0]
        hA, cA = run_comp(items, U14, capture=True)
        mA = margin(hA, gold)[0]
        rI, rA = own_rms(hI), own_rms(hA)
        E = {}
        for c in COMPS:
            hz, _ = run_comp(items, None, zero={c}); hzA, _ = run_comp(items, U14, zero={c})
            E[c] = dict(int=float(mI.mean() - margin(hz, gold)[0].mean()), abl=float(mA.mean() - margin(hzA, gold)[0].mean()),
                        int_f=float(mI.mean() - margin(hz, gold, rI)[0].mean()), abl_f=float(mA.mean() - margin(hzA, gold, rA)[0].mean()))
        cig = {c: (abs(E[c]["abl"]) - abs(E[c]["int"])) / Gm for c in COMPS}
        top = sorted(COMPS, key=lambda c: -cig[c])[:6]
        def S(zero, U, key="", ref_h=None):
            """(|E_X(set)| - |E_int(set)|)/G with set zeroed jointly, X = U-ablated run; unfrozen and LN-frozen."""
            hz, _ = run_comp(items, None, zero=zero); hx, _ = run_comp(items, U); hxz, _ = run_comp(items, U, zero=zero)
            mi = margin(hI, gold)[0].mean(); mx = margin(hx, gold)[0].mean(); rx = own_rms(hx)
            e_int = mi - margin(hz, gold)[0].mean(); e_x = mx - margin(hxz, gold)[0].mean()
            e_int_f = mi - margin(hz, gold, rI)[0].mean(); e_x_f = mx - margin(hxz, gold, rx)[0].mean()
            return (abs(e_x) - abs(e_int)) / Gm, (abs(e_x_f) - abs(e_int_f)) / Gm
        S_C = S(set(top), U14)
        rng = np.random.default_rng(31 + s); nat = sum(c[1] == "attn" for c in top)
        rand_sets = []
        for _ in range(20):
            at = rng.choice([c for c in COMPS if c[1] == "attn"], nat, replace=False) if False else [COMPS[i] for i in rng.choice([i for i, c in enumerate(COMPS) if c[1] == "attn"], nat, replace=False)]
            ml = [COMPS[i] for i in rng.choice([i for i, c in enumerate(COMPS) if c[1] == "mlp"], 6 - nat, replace=False)]
            rand_sets.append(set(at) | set(ml))
        S_rand = [S(rs, U14) for rs in rand_sets]
        S_ctrl = {}
        for fam in ("energy", "disp"):
            for j in range(4):
                S_ctrl[f"{fam}|{j}"] = S(set(top), ctrl[(fam, 4, s, j)].cuda().float())
        for j in range(4):
            S_ctrl[f"haar|{j}"] = S(set(top), ctrl[("haar", 4, "all", j)].cuda().float())
        gate = dict(S_C=S_C[0], S_C_frozen=S_C[1], rand_max=max(r[0] for r in S_rand), rand_max_frozen=max(r[1] for r in S_rand),
                    ctrl_max=max(v[0] for v in S_ctrl.values()), ctrl_max_frozen=max(v[1] for v in S_ctrl.values()))
        gate["pass"] = bool(S_C[0] >= 0.10 and S_C[0] > gate["rand_max"] and S_C[0] > gate["ctrl_max"] and S_C[1] >= 0.10 and S_C[1] > gate["rand_max_frozen"] and S_C[1] > gate["ctrl_max_frozen"])
        out["chains"][s] = dict(TE14_dev=te14, applicable=bool(te14 <= 0.50), top=[list(c) for c in top], cig={f"{c[0]}{c[1]}": cig[c] for c in COMPS}, gate=gate, S_ctrl=S_ctrl, S_rand=S_rand,
                                preserved_gap_dev_G_m=Gm)
        print(s, "TE14", round(te14, 3), "top", top, gate, flush=True)
    json.dump(out, open(R2 / f"comp_dev_h{h}.json", "w"), indent=0, default=float)
    print("S2 COMP DEV DONE")
else:
    dev = json.load(open(R2 / f"comp_dev_h{h}.json")); items = w.conf; gold = np.array([i.gold for i in items]); grp = np.array([i.group for i in items]); out = dict(hook=h, chains={})
    cl_int_arr = None
    hI, cI = run_comp(items, capture=True)
    pinI = {c: cI[:, CI[c]] for c in COMPS}
    lgI = logits_of(hI)[0].cpu().numpy()
    for s, Us in w.chains.items():
        d = dev["chains"][str(s)]
        if not (d["applicable"] and d["gate"]["pass"]):
            out["chains"][s] = dict(status="not run: applicable=%s gate_pass=%s" % (d["applicable"], d["gate"]["pass"])); print(s, out["chains"][s], flush=True); continue
        C = [tuple(c) for c in d["top"]]; U14 = w.union(Us, range(1, 5))
        def be_for(U, comps):
            hx, _ = run_comp(items, U); lgx = logits_of(hx)[0].cpu().numpy()
            hp, _ = run_comp(items, U, pin={c: pinI[c] for c in comps}); lgp = logits_of(hp)[0].cpu().numpy()
            return lgI, lgx, lgp
        def rel_abs(lgI_, lgx, lgp, ix):
            G = class_gap(lgI_[ix], gold[ix]); te_x = (G - class_gap(lgx[ix], gold[ix])) / G; te_p = (G - class_gap(lgp[ix], gold[ix])) / G
            return te_p - te_x, (te_p - te_x) * G
        lgI_, lgx, lgp = be_for(U14, C)
        be = rel_abs(lgI_, lgx, lgp, np.arange(len(gold)))
        lo, hi = cluster_boot(lambda ix: rel_abs(lgI_, lgx, lgp, ix)[0], grp, 400, s)
        # intact-run pin
        hp0, _ = run_comp(items, None, pin={c: pinI[c] for c in C}); lg0 = logits_of(hp0)[0].cpu().numpy()
        intact_pin = float(np.abs(lg0 - lgI_).max())
        rng = np.random.default_rng(77 + s); nat = sum(c[1] == "attn" for c in C); rs_be = []
        for _ in range(20):
            at = [COMPS[i] for i in rng.choice([i for i, c in enumerate(COMPS) if c[1] == "attn"], nat, replace=False)]
            ml = [COMPS[i] for i in rng.choice([i for i, c in enumerate(COMPS) if c[1] == "mlp"], 6 - nat, replace=False)]
            _, _, lgr = be_for(U14, at + ml); rs_be.append(rel_abs(lgI_, lgx, lgr, np.arange(len(gold))))
        cb = {}
        for fam in ("energy", "disp"):
            for j in range(4):
                Uc = ctrl[(fam, 4, s, j)].cuda().float(); l_, lx, lp = be_for(Uc, C); cb[f"{fam}|{j}"] = rel_abs(l_, lx, lp, np.arange(len(gold)))
        out["chains"][s] = dict(C=[list(c) for c in C], BE_rel=be[0], BE_abs=be[1], ci=[lo, hi], intact_pin_maxdiff=intact_pin, rand_set_BE_rel_mean=float(np.mean([r[0] for r in rs_be])),
                                rand_set_BE_rel_max_abs=float(np.max(np.abs([r[0] for r in rs_be]))), matched_ctrl_BE_rel=cb,
                                TE14=(lambda G: (G - class_gap(lgx, gold)) / G)(class_gap(lgI_, gold)))
        c_ = out["chains"][s]
        c_["pass"] = bool(be[0] >= 0.10 and be[1] >= 5 and lo > 0.05 and abs(c_["rand_set_BE_rel_mean"]) <= 0.03 and all(be[0] > v[0] for v in cb.values()) and intact_pin < 1e-3)
        print(s, {k: v for k, v in c_.items() if k != "matched_ctrl_BE_rel"}, flush=True)
    json.dump(out, open(R2 / f"comp_conf_h{h}.json", "w"), indent=0, default=float)
    print("S2 COMP CONF DONE")
