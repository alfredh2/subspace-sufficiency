"""Phase 8 S1 Gate 0: numerical validation (no scientific intervention beyond retention of U1 as a consistency check)."""
import json, pickle
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.hooks import residual_intervention, capture_residual
from causal_subspaces.models.loading import encode, last_logits, load_model
from causal_subspaces.mvs.projection import apply_mode, orthonormality_error
from causal_subspaces.phase7.stage1 import run_stats, class_gap, gold_array, primary_accuracy_vec, texts_of
from causal_subspaces.phase7.stage1b import build_1b
from causal_subspaces.utils.config import ROOT, load_config

R = ROOT / "results/phase8/stage1"
S1B = ROOT / "results/phase7/stage1b"
out = {}


def u1_path(h, s):
    f = S1B / f"final/M_letters_h{h}_r4_s{s}.pt"
    return f if f.exists() else S1B / f"armA/M_h{h}_r4_s{s}.pt"


# 1. orthonormality of the existing U1 files
out["u1_orth_err"] = {f"h{h}_s{s}": orthonormality_error(torch.load(u1_path(h, s))) for h, ss in ((25, range(5)), (21, range(2))) for s in ss}
# 2. projector algebra (pure tensor math, fp32/fp64)
g = torch.Generator().manual_seed(0)
d = 4096
Q, _ = torch.linalg.qr(torch.randn(d, 12, generator=g, dtype=torch.float64))
h = torch.randn(7, d, generator=g, dtype=torch.float64); mu = torch.randn(d, generator=g, dtype=torch.float64)
U = Q[:, :4]
rec = apply_mode(h, "retain", U, mu).double() + apply_mode(h, "ablate", U, mu).double() - h - mu
out["retain_plus_ablate_minus_h_minus_mu_max"] = float(rec.abs().max())     # retain+ablate = h + mu in exact arithmetic
P = lambda B: B @ B.T
out["idempotence_err"] = float((P(U) @ P(U) - P(U)).abs().max())
Ua, Ub = Q[:, :4], Q[:, 4:8]
out["union_projector_equals_sum_err"] = float((P(torch.cat([Ua, Ub], 1)) - P(Ua) - P(Ub)).abs().max())
out["cross_gram_err"] = float((Ua.T @ Ub).abs().max())
# 3. model-level checks
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
cfg["mvs"]["residual_index"] = 25
ref = get_reference(model, tok, cfg, rev, "cuda"); mu25 = ref["mu"].cuda()
dev = build_1b("M", tok)["letters"]["val"]
fresh = pickle.load(open(R / "fresh_items.pkl", "rb"))
dev_items = dev + fresh["devf"]["items"]
gold = gold_array(dev_items)
base = run_stats(model, tok, dev_items, bs=32)
b5 = run_stats(model, tok, dev_items, bs=5)
b1 = run_stats(model, tok, dev_items[:64], bs=1)
out["batch_noise"] = dict(max_abs_logit_diff_bs32_vs_bs5=float(np.abs(base["cand_logits"] - b5["cand_logits"]).max()),
                          gap_bs32=class_gap(base["cand_logits"], gold), gap_bs5=class_gap(b5["cand_logits"], gold),
                          gap_abs_diff=abs(class_gap(base["cand_logits"], gold) - class_gap(b5["cand_logits"], gold)),
                          max_abs_logit_diff_bs32_vs_bs1_first64=float(np.abs(base["cand_logits"][:64] - b1["cand_logits"]).max()),
                          greedy_agree_bs32_bs5=float((base["greedy"] == b5["greedy"]).mean()))
ids, mask, pos = encode(tok, texts_of(tok, dev_items[:16]), "cuda")
with torch.no_grad():
    l0 = last_logits(model, ids, mask, pos)
    with residual_intervention(model, 25, "noop"):
        l1 = last_logits(model, ids, mask, pos)
out["noop_hook_max_abs_diff"] = float((l0 - l1).abs().max())
# fp32-vs-bf16 projection error on real residuals
sink = []
with torch.no_grad(), capture_residual(model, 25, sink):
    last_logits(model, ids, mask, pos)
hh = sink[0]; U1 = torch.load(u1_path(25, 0)).cuda()
ret32 = (mu25.float() + ((hh.float() - mu25.float()) @ U1.float()) @ U1.float().T)
ret16 = apply_mode(hh, "retain", U1, mu25).float()
out["bf16_cast_rel_err_retain"] = float((ret16 - ret32).norm() / ret32.norm())
# consistency: U1 seed 0 retention on dev
st = run_stats(model, tok, dev_items, 25, "retain", U1, mu25, with_kl=True)
out["u1_s0_dev_retain"] = dict(kl=float(st["kl"].mean()), acc=float(primary_accuracy_vec(st, dev_items).mean()),
                               acc_intact=float(primary_accuracy_vec(base, dev_items).mean()))
out["effect_threshold_note"] = "all effect thresholds must exceed 2 x batch_noise.gap_abs_diff (logits); intact gap = %.1f" % class_gap(base["cand_logits"], gold)
out["noise_floor_gap_logits"] = out["batch_noise"]["gap_abs_diff"]
json.dump(out, open(R / "gate0.json", "w"), indent=1)
print(json.dumps(out, indent=1))
