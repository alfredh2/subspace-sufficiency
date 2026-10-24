"""Phase 8 S1: freeze the discovery outcome from DEV metrics only (no model, no confirm items): K* per chain, carrier file hashes and the
operationalisation of the confirm-stage rules. Commit the output before running phase8_s1_confirm.py."""
import hashlib, json, glob
import torch
from causal_subspaces.utils.config import ROOT

C = ROOT / "results/phase8/stage1/carriers"
out = dict(rules=dict(
    sufficient="dev (val + fresh-dev) retained KL <= 0.05 and primary accuracy >= 0.9 x intact; one retry with init seed +1000; chain stops at first double failure; cap K=8 (hook 25), K=4 (hook 21)",
    confirm_carrier_sufficient="fresh-confirm retained KL <= 0.05 and accuracy >= 0.9 x intact, point estimate; additionally the entity-cluster bootstrap (1000 resamples) 95% upper bound of KL <= 0.05 for the MSR verdict",
    kstar_confirm="number of consecutive carriers from k=1 that are confirm-sufficient, within the frozen dev chain (cannot exceed K*_dev)",
    MSR="PASS: median K*_confirm >= 4 and K*_confirm >= 4 in >= 4 of 5 chains (hook 25) with the bootstrap upper-bound condition on carriers 1-4 of those chains; FAIL: median K*_confirm <= 2; else INCONCLUSIVE. K* is a search-dependent lower bound (B3).",
    readout_like="logit-lens (final norm + unembedding of the four letter tokens) of the RETAINED residual at the hook, final position: accuracy >= 0.9",
    semantic_duplicate="two carriers of a chain are flagged when both decode the answer letter on confirm with accuracy >= 0.9 and their retained-logit-deviation cosine >= 0.5 (descriptive flag only; S1 makes no SCR/JCD claim)",
    anomaly_gates="numerical-validation failure; orthogonality violation; median K*_dev < 2; determinism refit max-angle cosine < 0.99 or KL differing by > 2x; < 36 confirm entities; hook-21 chains infeasible"), chains={})
for f in sorted(C.glob("h*_s*_k*.json")):
    if "par" in f.name:
        continue
    m = json.load(open(f)); h, s, k = m["hook"], m["seed"], m["k"]
    ch = out["chains"].setdefault(f"h{h}_s{s}", dict(carriers={}))
    pt = C / f"h{h}_s{s}_k{k}.pt"
    ch["carriers"][k] = dict(sufficient=m["sufficient"], dev_kl=m["dev_kl"], dev_acc=m["dev_acc"], retry_used=m.get("retry_used", False),
                             file=str(pt.relative_to(ROOT)) if pt.exists() else m.get("source"), sha256=hashlib.sha256(open(pt if pt.exists() else ROOT / m["source"], "rb").read()).hexdigest()[:16])
for name, ch in out["chains"].items():
    ks = 0
    for k in sorted(ch["carriers"]):
        if ch["carriers"][k]["sufficient"] and k == ks + 1:
            ks = k
    ch["kstar_dev"] = ks
json.dump(out, open(ROOT / "results/phase8/stage1/discovery_freeze.json", "w"), indent=1)
print({n: c["kstar_dev"] for n, c in out["chains"].items()})
