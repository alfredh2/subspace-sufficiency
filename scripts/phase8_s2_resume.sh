#!/bin/sh
# Resume Phase 8 S2 after a pause. Done at pause: hook-25 match/conf/analyze/comp-dev/overlap/trace. Remaining: hook 21 (match -> commit -> conf -> trace -> overlap), then analysis + report.
cd /home/ahof16/ai/verbalize_project/causal-subspaces; P=~/miniforge3/envs/ml/bin/python
$P scripts/phase8_s2_match.py --hook 21 > results/phase8/stage2/match21.log 2>&1
git add -f results/phase8/stage2/match_h21.json; sha256sum results/phase8/stage2/controls_h21.pt | cut -c1-16 > results/phase8/stage2/controls_h21.sha; git add -f results/phase8/stage2/controls_h21.sha
git commit -q -m "Phase 8 S2: hook-21 control matching on dev committed before hook-21 confirm"
$P scripts/phase8_s2_conf.py --hook 21 > results/phase8/stage2/conf21.log 2>&1   # resumable per chain
$P scripts/phase8_s2_trace.py --hook 21 > results/phase8/stage2/trace21.log 2>&1
$P scripts/phase8_s2_overlap.py --hook 21 > results/phase8/stage2/overlap21.log 2>&1
$P scripts/phase8_s2_analyze.py --hook 21 > results/phase8/stage2/analyze21.log 2>&1
