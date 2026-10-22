"""Phase 6 amendment 8: freeze the unseen nonword set C BEFORE any intervention on it. Tokenisation only (no model forward).
Writes results/phase6/set_c.json: seed, procedure, code hash, contents, eligibility rules, per-format manifests (prompt hashes), disjointness checks."""
import hashlib
import json

import yaml
from transformers import AutoTokenizer

from causal_subspaces.data.logic import build_contents, FAMILIES
from causal_subspaces.upstream.data import build_trials
from causal_subspaces.upstream.pipeline import FAMS
from causal_subspaces.utils.config import ROOT

SEED = 20261023
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], padding_side="left")
C = build_contents("nonword", n=48, seed=SEED)
words = lambda cs: {str(v) for c in cs for v in (c.values() if isinstance(c, dict) else []) if isinstance(v, str)}
earlier = {"ordinary": words(build_contents("ordinary")), "nonword_seed0": words(build_contents("nonword", seed=cfg.get("data_seed", cfg["seed"])))}
wC = words(C)
overlap = {k: sorted(wC & v) for k, v in earlier.items()}
out = dict(seed=SEED, procedure="build_contents('nonword', n=48, seed=20261023); alignment: Phase 2.5 pair_alignment + equal token length + identical category masks (upstream.data.build_trials); sampling: family-stratified, max_pairs=200, seed=1",
           code_sha256={f: hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest() for f in ("src/causal_subspaces/data/logic.py", "src/causal_subspaces/upstream/data.py", "src/causal_subspaces/phase05.py")},
           families=FAMS, min_trials_per_format=150, contents=C, n_contents=len(C), n_words=len(wC), word_overlap_with_earlier_lexicons=overlap, formats={})
for fmt in ("num_orig", "num_rev"):
    trials, census = build_trials(tok, C, list(range(48)), FAMS, fmt, max_pairs=200, seed=1, tag="phase6C")
    h = hashlib.sha256("\n".join(f"{t['tgt']}\x00{t['src']}" for t in trials).encode()).hexdigest()
    out["formats"][fmt] = dict(census=census, n_trials=len(trials), n_pairs=len(trials) // 2, prompts_sha256=h, underpowered=bool(len(trials) < 150), pair_content_family=[[t["content"], t["family"]] for t in trials[::2]])
json.dump(out, open(ROOT / "results/phase6/set_c.json", "w"), indent=1, default=str)
print({k: v for k, v in out.items() if k not in ("contents", "formats")}); print({f: (v["n_trials"], v["census"], v["underpowered"]) for f, v in out["formats"].items()})
