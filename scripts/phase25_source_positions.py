"""Phase 2.5 step 2: descriptive source-position analysis of the frozen S8 heads using the MODEL'S OWN attention implementation (eager attention weights,
the model's own v_proj, gains from o_proj). The custom recompute path is not used. Exact accounting: head contribution at the last token = sum_s A_s * (gain_h . v_s).
Routing/content terms are an accounting identity only (descriptive, not causal)."""
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.loading import encode, format_prompt, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
LAY = [19, 20, 21, 22]
S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
GAIN = {l: head_gain(model, l, d_c).view(H, HD) for l in LAY}
splits = get_splits(cfg); contents = build_contents("ordinary")
test_ids = sorted({e.content_id for e in splits["items"]})

# ---------------------------------------------------------------- prompts and token categories -------------------------------------------------------------
def spans_logic(user, syl_lines, family):
    instr_end = user.index("\n\n")
    roles = ["fact", "rule", "conclusion"] if family == "fact_first" else ["rule", "fact", "conclusion"]
    spans, pos = [("instruction", 0, instr_end + 2)], instr_end + 2
    starts = []
    for ln in syl_lines:
        st = user.index(ln, pos); starts.append(st); pos = st + len(ln)
    ends = starts[1:] + [len(user)]
    for r, st, en in zip(roles, starts, ends): spans.append((r, st, en))
    return spans
def spans_localizer(user):
    q = user.index('\n\n"'); return [("instruction", 0, q + 2), ("body", q + 2, len(user))]
def spans_pronto(user):
    ls = user.split("\n"); a = len(ls[0]) + 1; b = a + len(ls[1]) + 1
    return [("instruction", 0, a), ("body", a, b), ("question", b, len(user))]
def categorize(prompt, user, spans):
    """token -> category; tokens: first token = 'first_token'; chat template prefix; spans (shifted to the prompt); tail from <|im_end|>; last token = 'self'."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    off, ids = enc["offset_mapping"], enc["input_ids"]
    u0 = prompt.index(user); u1 = u0 + len(user); cats = []
    for i, (a, b) in enumerate(off):
        if i == 0: c = "first_token"
        elif a < u0: c = "chat_prefix"
        elif a >= u1: c = "template_tail"
        else:
            c = "other"
            for nm, s0, s1 in spans:
                if u0 + s0 <= a < u0 + s1: c = nm; break
        cats.append(c)
    cats[-1] = "self"
    # last token of the final statement (the sentence-final '.') separately
    last_stmt = max((i for i, c in enumerate(cats) if c in ("conclusion", "question", "body") and i < len(cats) - 1), default=None)
    if last_stmt is not None:
        nm = cats[last_stmt]; cats[last_stmt] = nm + "_final_token"
    return cats

pairs = P5.build_pairs(contents, test_ids, TRAIN_FAMILIES + HELDOUT_FAMILIES, tok)
fmt = "num_orig"
logic_items, kept = [], 0
for p in pairs:
    pv, pi = format_prompt(tok, P5.build_prompt(fmt, p["valid_syl"]), False), format_prompt(tok, P5.build_prompt(fmt, p["invalid_syl"]), False)
    ok, why = A.pair_alignment(tok, pv, p["valid_syl"].split("\n"), pi, p["invalid_syl"].split("\n"))
    if not ok: continue
    kept += 1
    for syl, y in ((p["valid_syl"], True), (p["invalid_syl"], False)):
        user = P5.build_prompt(fmt, syl); prompt = format_prompt(tok, user, False)
        logic_items.append(dict(prompt=prompt, accept=y, cats=categorize(prompt, user, spans_logic(user, syl.split("\n"), p["family"]))))
print(f"logic: {kept} aligned pairs -> {len(logic_items)} prompts", flush=True)
c5 = build_controls(5, 128, tuple(cfg["controls"]["mmlu_subjects"]), cfg["controls"]["mmlu_per_subject"])
def ctrl_items(key, span_fn):
    out = []
    for t, g in c5[key]["items"][:128]:
        prompt = format_prompt(tok, t, False); out.append(dict(prompt=prompt, accept=(g == "1"), cats=categorize(prompt, t, span_fn(t))))
    return out
loc_items, pro_items = ctrl_items("language_localizer", spans_localizer), ctrl_items("prontoqa", spans_pronto)

# ---------------------------------------------------------------- model passes (sdpa + eager) -------------------------------------------------------------------
def run(items, cat_names, paired=False, bs=16):
    N = len(items); cidx = {c: i for i, c in enumerate(cat_names)}
    mass = np.zeros((N, 4, H, len(cat_names)), np.float32); contrib = np.zeros_like(mass)
    s_eager = np.zeros((N, 4, H), np.float32); s_sdpa = np.zeros((N, 4, H), np.float32); s_ident = np.zeros((N, 4, H), np.float32)
    ent = np.zeros((N, 4, H), np.float32)
    rc = {"routing": np.zeros((N // 2, 8, len(cat_names)), np.float32), "content": np.zeros((N // 2, 8, len(cat_names)), np.float32), "total": np.zeros((N // 2, 8), np.float32)} if paired else None
    for s0 in range(0, N, bs):
        b = items[s0:s0 + bs]; ids, mask, pos = encode(tok, [x["prompt"] for x in b], dev); B, T = ids.shape
        onehot = torch.zeros(B, T, len(cat_names), device=dev)
        for i, x in enumerate(b):
            npad = T - len(x["cats"])
            for j, c in enumerate(x["cats"]):
                if c in cidx: onehot[i, npad + j, cidx[c]] = 1
        cap = {}
        hs = []
        for l in LAY:
            hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: cap.__setitem__(("z", l), i[0].detach().float().view(B, T, H, HD)[:, -1])))
            hs.append(model.model.layers[l].self_attn.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(("w", l), o[1].detach().float() if o[1] is not None else None)))
            hs.append(model.model.layers[l].self_attn.v_proj.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(("v", l), o.detach().float().view(B, T, KV, HD))))
        def fwd():
            saved = model.model.layers; model.model.layers = saved[:max(LAY) + 1]
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            model.model.layers = saved
        model.config._attn_implementation = "sdpa"; fwd()
        for li, l in enumerate(LAY): s_sdpa[s0:s0 + B, li] = (cap[("z", l)] * GAIN[l]).sum(-1).cpu().numpy()
        model.config._attn_implementation = "eager"; fwd(); model.config._attn_implementation = "sdpa"
        for h_ in hs: h_.remove()
        for li, l in enumerate(LAY):
            w = cap[("w", l)][:, :, -1]                                                  # (B,H,T): the model's own attention weights of the last query
            v = cap[("v", l)]                                                            # (B,T,KV,HD)
            vh = v[:, :, torch.arange(H, device=dev) // (H // KV)]                       # (B,T,H,HD): value of each query head's KV group
            u = (vh * GAIN[l]).sum(-1).permute(0, 2, 1)                                   # (B,H,T): gain_h . v_s
            c = w * u
            mass[s0:s0 + B, li] = torch.einsum("bht,btc->bhc", w, onehot).cpu().numpy()
            contrib[s0:s0 + B, li] = torch.einsum("bht,btc->bhc", c, onehot).cpu().numpy()
            s_eager[s0:s0 + B, li] = (cap[("z", l)] * GAIN[l]).sum(-1).cpu().numpy() if False else 0
            s_ident[s0:s0 + B, li] = c.sum(-1).cpu().numpy()
            ent[s0:s0 + B, li] = (-(w.clamp_min(1e-12).log() * w).sum(-1)).cpu().numpy()
            if paired:   # in-batch pairs: rows 2k (valid), 2k+1 (invalid); aligned pairs share spans, so categories coincide
                for hi, (ll, hh) in enumerate(S8):
                    if ll != l: continue
                    for k in range(B // 2):
                        a_, b_ = 2 * k, 2 * k + 1
                        Av, Ai, uv, ui = w[a_, hh], w[b_, hh], u[a_, hh], u[b_, hh]
                        r, ct, tot = A.routing_content(Av, Ai, uv, ui)
                        rr = ((Av - Ai) * (uv + ui) / 2); cc = ((Av + Ai) / 2 * (uv - ui))
                        rc["routing"][(s0 // 2) + k, hi] = (rr.unsqueeze(-1) * onehot[a_]).sum(0).cpu().numpy(); rc["content"][(s0 // 2) + k, hi] = (cc.unsqueeze(-1) * onehot[a_]).sum(0).cpu().numpy(); rc["total"][(s0 // 2) + k, hi] = float(tot)
        # eager head contribution from the model's own eager z (captured on the eager run)
        for li, l in enumerate(LAY): pass
    return dict(mass=mass, contrib=contrib, s_sdpa=s_sdpa, s_ident=s_ident, ent=ent, rc=rc)

LOGIC_CATS = ["first_token", "chat_prefix", "instruction", "rule", "fact", "conclusion", "conclusion_final_token", "template_tail", "self"]
LOC_CATS = ["first_token", "chat_prefix", "instruction", "body", "body_final_token", "template_tail", "self"]
PRO_CATS = ["first_token", "chat_prefix", "instruction", "body", "question", "question_final_token", "template_tail", "self"]
R = {}
for name, items, cats, paired in (("logic", logic_items, LOGIC_CATS, True), ("localizer", loc_items, LOC_CATS, False), ("prontoqa", pro_items, PRO_CATS, False)):
    R[name] = dict(items=items, cats=cats, **run(items, cats, paired)); print(name, "done", flush=True)

# ---------------------------------------------------------------- tables ---------------------------------------------------------------------------------------
def head_idx(l, h): return LAY.index(l), h
out = {"S8": [dict(layer=l, head=h) for l, h in S8], "tasks": {}}
md = ["# Phase 2.5 source-position tables (model's own eager attention; auto-generated by scripts/phase25_source_positions.py)\n"]
for name, d in R.items():
    acc = np.array([x["accept"] for x in d["items"]]); cats = d["cats"]
    # identity: sum over all sources (all categories incl. 'other') of A*u vs head contribution from the model's own z (sdpa run); eager vs sdpa
    tot_cat = d["contrib"].sum(-1)                                                    # categories exclude 'other'/pad: compare with ident
    ident_err = np.abs(d["s_ident"] - d["s_sdpa"])
    ident = {f"L{l}H{h}": dict(max_abs_err=float(ident_err[:, LAY.index(l), h].max()), rms_err=float(np.sqrt((ident_err[:, LAY.index(l), h] ** 2).mean())), head_std=float(d["s_sdpa"][:, LAY.index(l), h].std()),
                               corr=float(np.corrcoef(d["s_ident"][:, LAY.index(l), h], d["s_sdpa"][:, LAY.index(l), h])[0, 1]), eager_minus_sdpa_class_diff=float((d["s_ident"][acc, LAY.index(l), h].mean() - d["s_ident"][~acc, LAY.index(l), h].mean()) - (d["s_sdpa"][acc, LAY.index(l), h].mean() - d["s_sdpa"][~acc, LAY.index(l), h].mean())),
                               class_diff_sdpa=float(d["s_sdpa"][acc, LAY.index(l), h].mean() - d["s_sdpa"][~acc, LAY.index(l), h].mean()), class_diff_eager=float(d["s_ident"][acc, LAY.index(l), h].mean() - d["s_ident"][~acc, LAY.index(l), h].mean())) for l, h in S8}
    T = dict(n=len(d["items"]), categories=cats, identity=ident, heads={})
    md.append(f"\n## {name} (n = {len(d['items'])}; accept {int(acc.sum())} / reject {int((~acc).sum())})\n")
    md.append("Identity check (sum of per-source contributions from the model's own eager weights/values vs head contribution from the model's own SDPA output): " + "; ".join(f"L{l}H{h}: corr {v['corr']:.4f}, rms err {v['rms_err']:.3f} (head std {v['head_std']:.2f}), class diff eager {v['class_diff_eager']:+.3f} vs sdpa {v['class_diff_sdpa']:+.3f}" for (l, h), v in zip(S8, ident.values())) + "\n")
    for (l, h) in S8 + [(l, "avg") for l in LAY]:
        li = LAY.index(l)
        if h == "avg":
            mass, con = d["mass"][:, li].mean(1), d["contrib"][:, li].mean(1)
        else:
            mass, con = d["mass"][:, li, h], d["contrib"][:, li, h]
        key = f"L{l}H{h}" if h != "avg" else f"L{l}_layer_average_all_32_heads"
        row = {}
        for ci, c in enumerate(cats):
            m_a, m_r = mass[acc, ci], mass[~acc, ci]; c_a, c_r = con[acc, ci], con[~acc, ci]
            row[c] = dict(mass_accept=float(m_a.mean()), mass_reject=float(m_r.mean()), contrib_accept=float(c_a.mean()), contrib_reject=float(c_r.mean()),
                          contrib_class_diff=float(c_a.mean() - c_r.mean()), contrib_class_diff_se=float(np.sqrt(c_a.var(ddof=1) / len(c_a) + c_r.var(ddof=1) / len(c_r))),
                          mass_class_diff=float(m_a.mean() - m_r.mean()), mass_class_diff_se=float(np.sqrt(m_a.var(ddof=1) / len(m_a) + m_r.var(ddof=1) / len(m_r))))
        tot_diff = sum(r["contrib_class_diff"] for r in row.values())
        for c in cats: row[c]["share_of_class_diff"] = row[c]["contrib_class_diff"] / tot_diff if abs(tot_diff) > 1e-9 else float("nan")
        T["heads"][key] = dict(categories=row, total_class_diff=float(tot_diff), last_query_attention_entropy_accept=float(d["ent"][acc, li, h].mean()) if h != "avg" else float(d["ent"][acc, li].mean()), last_query_attention_entropy_reject=float(d["ent"][~acc, li, h].mean()) if h != "avg" else float(d["ent"][~acc, li].mean()))
        if h != "avg" or l == 22:
            md.append(f"\n### {name}: {key}  (total class-difference of the head's contribution {tot_diff:+.3f})\n\n| source category | attention mass accept | reject | Δmass (±SE) | contribution accept | reject | Δcontribution (±SE) | share of class diff |\n|---|---|---|---|---|---|---|---|")
            for c in cats:
                r = row[c]; md.append(f"| {c} | {r['mass_accept']:.3f} | {r['mass_reject']:.3f} | {r['mass_class_diff']:+.3f} ± {r['mass_class_diff_se']:.3f} | {r['contrib_accept']:+.3f} | {r['contrib_reject']:+.3f} | {r['contrib_class_diff']:+.3f} ± {r['contrib_class_diff_se']:.3f} | {r['share_of_class_diff']:.2f} |")
    out["tasks"][name] = T
# routing / content accounting on aligned logic pairs
rc = R["logic"]["rc"]; cats = R["logic"]["cats"]
out["routing_content_accounting"] = {}
md.append("\n## Logic aligned pairs: symmetric routing/content ACCOUNTING identity (descriptive, not causal), per S8 head and category: mean pair difference (valid − invalid) of the contribution\n")
md.append("| head | total pair diff | routing total | content total | " + " | ".join(f"{c} (routing / content)" for c in cats if c not in ("first_token", "chat_prefix", "self")) + " |\n|---|---|---|---|" + "---|" * (len(cats) - 3))
for hi, (l, h) in enumerate(S8):
    tot = rc["total"][:, hi]; ro = rc["routing"][:, hi].sum(-1); co = rc["content"][:, hi].sum(-1)
    chk = float(np.abs((rc["routing"][:, hi].sum(-1) + rc["content"][:, hi].sum(-1)) - tot).max())
    out["routing_content_accounting"][f"L{l}H{h}"] = dict(pair_diff=float(tot.mean()), routing=float(ro.mean()), content=float(co.mean()), identity_max_abs_residual=chk,
                                                         per_category={c: dict(routing=float(rc["routing"][:, hi, ci].mean()), content=float(rc["content"][:, hi, ci].mean())) for ci, c in enumerate(cats)})
    md.append(f"| L{l}H{h} | {tot.mean():+.3f} | {ro.mean():+.3f} | {co.mean():+.3f} | " + " | ".join(f"{rc['routing'][:, hi, ci].mean():+.2f} / {rc['content'][:, hi, ci].mean():+.2f}" for ci, c in enumerate(cats) if c not in ("first_token", "chat_prefix", "self")) + f" |")
    md.append("")
json.dump(out, open(ROOT / "results/phase25/source_positions.json", "w"), indent=1)
(ROOT / "results/phase25/source_positions_tables.md").write_text("\n".join(md))
print("\n".join(md[:60]))
