## Gate (a): intact test accuracy (frozen screen)

| task/format | n | primary acc | 95% CI | threshold | pass |
|---|---|---|---|---|---|
| L/num_orig | 256 | 0.945 | 0.921-0.969 | 0.9 | True |
| L/num_rev | 256 | 0.938 | 0.912-0.961 | 0.9 | True |
| S/words | 384 | 0.979 | 0.943-0.998 | 0.9 | True |
| S/num_orig | 384 | 0.927 | 0.869-0.975 | 0.9 | True |
| S/num_rev | 384 | 0.992 | 0.981-1.000 | 0.9 | True |
| G/pres | 384 | 0.990 | 0.980-0.997 | 0.9 | True |
| G/past | 384 | 0.987 | 0.973-0.997 | 0.9 | True |
| N/cont | 384 | 1.000 | 1.000-1.000 | 0.9 | True |
| N/qa | 384 | 0.977 | 0.961-0.990 | 0.9 | True |
| M/letters | 264 | 1.000 | 1.000-1.000 | 0.8 | True |
| M/digits | 264 | 1.000 | 1.000-1.000 | 0.8 | True |

## Sweep (val, 300 steps, seed 0): frozen choice

| task | passing configs of 12 | choice (hook, rank) | passes literal 10-logit floor |
|---|---|---|---|
| L | 1 | (23, 2) | True |
| S | 1 | (25, 4) | True |
| G | 0 | None | False |
| N | 0 | None | False |
| M | 0 | None | False |

## Final MVS gates on TEST (seed-median; 3 seeds)

| task/format | hook,rank | KL ret | acc I / R / A | gap I | gap reduction (rel) | transplant all (3 seeds) | random all mean / max | transplant last | consensus cos mean / min | b | c | c literal | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L/num_orig | 23,2 | 0.0062 | 0.945 / 0.941 / 0.500 | 28.0 | 20.9 (0.75; seeds 0.75, 0.77, 0.37) | 0.60, 0.63, 0.57 | 0.000 / 0.001 | 0.48, 0.50, 0.45 | 0.84 / 0.82 | True | True | True | True |
| L/num_rev | 23,2 | 0.0067 | 0.938 / 0.930 / 0.789 | 23.1 | 8.9 (0.39; seeds 0.39, 0.67, 0.35) | 0.57, 0.54, 0.50 | 0.001 / 0.002 | 0.41, 0.38, 0.35 | 0.83 / 0.80 | True | False | False | True |
| S/words | 25,4 | 0.0045 | 0.979 / 0.982 / 0.935 | 29.0 | 14.8 (0.51; seeds 0.59, 0.51, 0.38) | 0.60, 0.60, 0.67 | 0.001 / 0.003 | 0.57, 0.58, 0.65 | 0.74 / 0.73 | True | True | True | True |
| S/num_orig | 25,4 | 0.0025 | 0.927 / 0.930 / 0.990 | 23.0 | 10.5 (0.45; seeds 0.45, 0.30, 0.50) | 0.59, 0.71, 0.78 | 0.001 / 0.003 | 0.59, 0.70, 0.78 | 0.84 / 0.81 | True | False | False | True |
| S/num_rev | 25,4 | 0.0014 | 0.992 / 0.992 / 0.984 | 22.0 | 8.2 (0.37; seeds 0.51, 0.27, 0.37) | 0.65, 0.67, 0.64 | 0.001 / 0.003 | 0.65, 0.67, 0.64 | 0.79 / 0.77 | True | False | False | True |

## Task verdicts

| task | gate a | primary-format b,c,d | >=2 formats (b&d) incl. reversed, consensus>=0.7 | verdict |
|---|---|---|---|---|
| L | True | True, True, True | True (['num_orig', 'num_rev']) | PASS |
| S | True | True, True, True | True (['words', 'num_orig', 'num_rev']) | INCONCLUSIVE |
| G | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |
| N | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |
| M | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |

## Task-direction geometry (cosines of consensus directions; only comparable at the same hook)

| | L/num_orig | L/num_rev | S/words | S/num_orig | S/num_rev |
|---|---|---|---|---|---|
| L/num_orig | 1.00 | -0.38 | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) |
| L/num_rev | -0.38 | 1.00 | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) |
| S/words | n/a (hook 25 vs 23) | n/a (hook 25 vs 23) | 1.00 | 0.38 | 0.10 |
| S/num_orig | n/a (hook 25 vs 23) | n/a (hook 25 vs 23) | 0.38 | 1.00 | -0.56 |
| S/num_rev | n/a (hook 25 vs 23) | n/a (hook 25 vs 23) | 0.10 | -0.56 | 1.00 |