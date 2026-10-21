"""Phase 0.5, Experiment 2: direction-level analysis of the three rank-2 MVSs (no retraining).
Aligns subspaces with principal vectors, defines per-seed 'shared' (most contained in the other seeds' spans) and
'specific' (orthogonal complement inside the same MVS) rank-1 directions, plus a cross-seed consensus direction, and
runs rank-1 ablation and rank-1 retention. Retention tests sufficiency; ablation tests necessity - they are reported separately.
Usage: python scripts/phase05_directions.py"""
import itertools
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P
from causal_subspaces.experiment import evaluate_condition, get_controls, get_lm_passages, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

SEEDS = [0, 1, 2]
OUT = ROOT / "results" / "phase05"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx, bs = cfg["hardware"]["device"], cfg["mvs"]["residual_index"], cfg["hardware"]["eval_batch_size"]
cks = {s: load_checkpoint(ROOT / f"results/mvs_rank2_seed{s}/checkpoints/step_1600.pt") for s in SEEDS}
Us = {s: cks[s]["U"] for s in SEEDS}
mu = cks[0]["mu"].to(dev)

# ---- geometry (CPU) ---------------------------------------------------------------------------------------------
geo = {"pairs": {}}
for a_, b_ in itertools.combinations(SEEDS, 2):
    S, Pa, Pb = P.pairwise_principal_angles(Us[a_], Us[b_])
    geo["pairs"][f"{a_}-{b_}"] = dict(cosines=S.tolist(), angles_deg=torch.rad2deg(torch.acos(S.clamp(max=1))).tolist())
cons = P.consensus_directions(Us)
geo["consensus"] = {str(s): dict(eig_shared=c["eig_shared"], eig_specific=c["eig_specific"]) for s, c in cons.items()}
cos = lambda u, v: float((u.double() @ v.double()) / (u.double().norm() * v.double().norm()))
geo["shared_vs_shared_cos"] = {f"{a_}-{b_}": cos(cons[a_]["shared"], cons[b_]["shared"]) for a_, b_ in itertools.combinations(SEEDS, 2)}
geo["specific_vs_specific_cos"] = {f"{a_}-{b_}": cos(cons[a_]["specific"], cons[b_]["specific"]) for a_, b_ in itertools.combinations(SEEDS, 2)}
geo["shared_vs_specific_cos"] = {f"{a_}-{b_}": cos(cons[a_]["shared"], cons[b_]["specific"]) for a_ in SEEDS for b_ in SEEDS if a_ != b_}
consensus = torch.nn.functional.normalize(sum(cons[s]["shared"] for s in SEEDS), dim=0)
geo["consensus_cos_to_shared"] = {str(s): cos(consensus, cons[s]["shared"]) for s in SEEDS}
geo["consensus_cos_to_specific"] = {str(s): cos(consensus, cons[s]["specific"]) for s in SEEDS}
print(json.dumps(geo, indent=1))

# ---- interventions -----------------------------------------------------------------------------------------------
splits, controls = get_splits(cfg), get_controls(cfg)
model, tok, rev = load_model(cfg["model"], dev)
passages = get_lm_passages(tok, cfg)
ex = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
gold_valid = np.array([e.valid for e in ex])
spl = np.array([e.split for e in ex])
texts = [P.build_prompt("num_orig", e.syllogism) for e in ex]
ivs = {}
for s in SEEDS:
    for kind in ("shared", "specific"):
        ivs[f"s{s}_{kind}"] = cons[s][kind][:, None]
ivs["consensus"] = consensus[:, None]
results = {"geometry": geo, "interventions": {}}
def one(name, mode, U):
    U = U.to(dev)
    m, _ = evaluate_condition(model, tok, cfg, splits, controls, mode, U, mu, passages, dev)
    rec = P.collect(model, tok, texts, "1", "2", device=dev, batch_size=bs, residual_index=idx, mode=mode, U=U, mu=mu)
    st = P.answer_stats(tok, rec, gold_valid, "1", "2", "1", spl)
    out = dict(logic={k: v["overall"]["acc"] for k, v in m["logic"].items()},
               logic_pred_first_char={k: v["response_distribution"] for k, v in m["logic"].items()},
               logic_stats=st,
               controls={k: (v if k == "lm_kl" else dict(acc=v["overall"]["acc"], pred=v["response_distribution"], mass=v["overall"]["mass_on_labels"]))
                         for k, v in m["controls"].items()})
    for cn in ("language_localizer", "prontoqa"):
        items = controls[cn]["items"]
        r = P.collect(model, tok, [t for t, _ in items], "1", "2", device=dev, batch_size=bs, residual_index=idx, mode=mode, U=U, mu=mu)
        out["controls"][cn]["stats"] = P.answer_stats(tok, r, np.array([g == "1" for _, g in items]), "1", "2", "1")
    results["interventions"][f"{mode}/{name}"] = out
    print(f"{mode:7s} {name:16s} logic acc "+" ".join(f"{k}={v:.3f}" for k, v in out["logic"].items())+
          f" | pred_aff={st['pred_aff']:.2f} auc={st['auc']:.3f} margin={st['margin_mean']:+.1f}"
          f" | localizer={out['controls']['language_localizer']['acc']:.3f} prontoqa={out['controls']['prontoqa']['acc']:.3f}", flush=True)
for name, v in ivs.items():
    for mode in ("ablate", "retain"):
        one(name, mode, v)
json.dump(results, open(OUT / "exp2_directions.json", "w"), indent=1)
print("saved")
