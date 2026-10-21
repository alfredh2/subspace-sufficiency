"""Phase 1 Stage 0: cost of true Haar generation at m = 12288 (and the cheaper nulls). Decides the draw counts (docs/phase1_plan.md section 5)."""
import json
import time

import torch

from causal_subspaces.rotations import haar as H
from causal_subspaces.utils.config import ROOT

m, dev = 12288, "cuda"
out = {"m": m, "gpu": torch.cuda.get_device_name(0)}
for name, f in (("haar_qr_fp32", lambda s: H.haar_orthogonal(m, s, dev)),):
    ts, peaks, errs = [], [], []
    for s in range(3):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        t0 = time.time(); R = f(s); torch.cuda.synchronize(); ts.append(time.time() - t0)
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        v = torch.randn(64, m, device=dev)
        errs.append(float((v @ R.T @ R - v).abs().max() / v.abs().max()))  # orthogonality probe: R^T R v = v
        del R
    out[name] = dict(seconds=ts, peak_gb=peaks, orth_rel_err=errs)
t0 = time.time(); S = H.StructuredRotation(m, 0, block=1024, rounds=3, device=dev); torch.cuda.synchronize(); out["structured_build_s"] = time.time() - t0
x = torch.randn(64, m, device=dev); t0 = time.time(); y = S.apply(x); torch.cuda.synchronize(); out["structured_apply_s"] = time.time() - t0
out["structured_norm_err"] = float((y.norm(dim=1) / x.norm(dim=1) - 1).abs().max())
d, g = torch.randn(m, device=dev), torch.randn(m, device=dev)
t0 = time.time(); H.two_frame_pair(d, g, 1000, 0); torch.cuda.synchronize(); out["two_frame_1000_draws_s"] = time.time() - t0
mx, my = max(out["haar_qr_fp32"]["seconds"]), max(out["haar_qr_fp32"]["peak_gb"])
out["rule"] = "20 true-Haar per layer (Stage 2/3) and 10 per layer (Stage 4)" if (mx <= 20 and my <= 4) else "REDUCE true-Haar counts; label small Haar reference"
(ROOT / "results/phase1").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results/phase1/rotation_benchmark.json", "w"), indent=1)
print(json.dumps(out, indent=1))
