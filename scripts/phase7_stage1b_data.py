"""Stage 1b-0: build the 1b partitions for all tasks, leakage tests, sizes and hashes -> results/phase7/stage1b/manifest.json.
No model forward passes (tokenizer only). Run before any fit; the manifest is committed before Arm A."""
import json
from transformers import AutoTokenizer
from causal_subspaces.phase7.tasks import data_hash
from causal_subspaces.phase7.stage1b import build_1b, build_N_fixedpool, leakage_report, TASKS_1B, FORMATS_1B
from causal_subspaces.utils.config import ROOT

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
man, ok = {}, True
for t in TASKS_1B + ["N_fixedpool"]:
    data = build_N_fixedpool(tok) if t == "N_fixedpool" else build_1b(t, tok)
    rep = leakage_report(data)
    for fmt, r in rep.items():
        r["hash"] = {sp: data_hash(data[fmt][sp]) for sp in ("train", "val", "test")}
        bad = [k for k, v in r["group_overlap"].items() if v]   # content groups must be disjoint; identical content-free prompts (L semi-formal P/W family) are reported, not a leak
        if bad and t != "N_fixedpool":
            ok = False
        print(t, fmt, r["n"], r["n_groups"], "pairs", r["n_pairs"], "LEAK" if bad else "disjoint", flush=True)
    man[t] = rep
man["all_disjoint"] = ok
(ROOT / "results/phase7/stage1b").mkdir(parents=True, exist_ok=True)
json.dump(man, open(ROOT / "results/phase7/stage1b/manifest.json", "w"), indent=1)
print("ALL DISJOINT" if ok else "LEAKAGE")
