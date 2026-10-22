
## Set A, num_orig: value-origin curve for P = Concl (434 trials, 217 pairs, 8 contents)

Denominator T*_V(19, Concl) = 0.169 (lower bound 0.153; defined=True). Reference T_ref = T*_V(19, KV) = 0.337.

Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.

| index | T*_V (signed) [pair] / [content] | F | Δmargin (logits, signed toward source) | Δ H* readout (signed) | vproj (value change projection) | U(i): unrestricted T |
|---|---|---|---|---|---|---|
| 0 | 0.097 [0.09, 0.11] / [0.09, 0.11] | 0.58 | 2.703 | 0.246 | 0.269 | 0.530 |
| 1 | 0.098 [0.09, 0.11] / [0.09, 0.11] | 0.58 | 2.708 | 0.246 | 0.269 | 0.530 |
| 2 | 0.099 [0.09, 0.11] / [0.09, 0.11] | 0.59 | 2.743 | 0.247 | 0.270 | 0.529 |
| 3 | 0.100 [0.09, 0.11] / [0.09, 0.11] | 0.59 | 2.764 | 0.247 | 0.271 | 0.527 |
| 4 | 0.106 [0.10, 0.12] / [0.10, 0.11] | 0.63 | 2.934 | 0.252 | 0.276 | 0.523 |
| 5 | 0.106 [0.10, 0.12] / [0.10, 0.11] | 0.63 | 2.935 | 0.252 | 0.277 | 0.523 |
| 6 | 0.118 [0.11, 0.13] / [0.11, 0.13] | 0.70 | 3.270 | 0.267 | 0.289 | 0.531 |
| 7 | 0.119 [0.11, 0.13] / [0.11, 0.13] | 0.71 | 3.308 | 0.269 | 0.291 | 0.527 |
| 8 | 0.124 [0.11, 0.14] / [0.12, 0.13] | 0.73 | 3.430 | 0.277 | 0.298 | 0.531 |
| 9 | 0.126 [0.11, 0.14] / [0.12, 0.13] | 0.75 | 3.487 | 0.279 | 0.305 | 0.534 |
| 10 | 0.128 [0.12, 0.14] / [0.12, 0.14] | 0.76 | 3.557 | 0.294 | 0.318 | 0.543 |
| 11 | 0.129 [0.12, 0.14] / [0.12, 0.14] | 0.77 | 3.588 | 0.294 | 0.322 | 0.544 |
| 12 | 0.132 [0.12, 0.14] / [0.12, 0.14] | 0.78 | 3.650 | 0.314 | 0.338 | 0.551 |
| 13 | 0.129 [0.12, 0.14] / [0.12, 0.14] | 0.77 | 3.581 | 0.305 | 0.346 | 0.549 |
| 14 | 0.136 [0.12, 0.15] / [0.12, 0.15] | 0.80 | 3.759 | 0.315 | 0.370 | 0.568 |
| 15 | 0.122 [0.11, 0.14] / [0.11, 0.14] | 0.72 | 3.374 | 0.278 | 0.361 | 0.544 |
| 16 | 0.121 [0.11, 0.13] / [0.11, 0.13] | 0.72 | 3.360 | 0.269 | 0.381 | 0.533 |
| 17 | 0.131 [0.12, 0.14] / [0.12, 0.14] | 0.78 | 3.646 | 0.247 | 0.421 | 0.516 |
| 18 | 0.133 [0.12, 0.14] / [0.12, 0.14] | 0.79 | 3.700 | 0.238 | 0.453 | 0.511 |
| 19 | 0.169 [0.15, 0.18] / [0.16, 0.18] | 1.00 | 4.675 | 0.267 | 0.513 | 0.500 |

Onsets (F): θ=0.1: first 0, sustained 0; θ=0.5: first 0, sustained 0; θ=0.9: first 19, sustained 19; trajectory class: **flat**; F(4) = 0.63 pair [0.57, 0.70] / content [0.58, 0.69] (LOCO 0.61…0.64).
Head-readout transfer onset (θ=0.5): first 0, sustained 0 (within 2 of the margin onset: True).

### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)

| group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV T | 0.337 | 0.337 | 0.331 | 0.326 | 0.326 | 0.325 | 0.322 | 0.322 | 0.325 | 0.330 | 0.330 | 0.337 |
| KV R | 1.00 | 1.00 | 0.98 | 0.97 | 0.97 | 0.96 | 0.96 | 0.96 | 0.96 | 0.98 | 0.98 | 1.00 |
| KV F | 1.00 | 1.00 | 0.98 | 0.97 | 0.97 | 0.96 | 0.96 | 0.96 | 0.96 | 0.98 | 0.98 | 1.00 |   (F defined; closing index None)
| Fact T | 0.130 | 0.131 | 0.137 | 0.140 | 0.135 | 0.112 | 0.087 | 0.060 | 0.056 | 0.041 | 0.034 | 0.020 |
| Fact R | 0.38 | 0.39 | 0.41 | 0.42 | 0.40 | 0.33 | 0.26 | 0.18 | 0.17 | 0.12 | 0.10 | 0.06 |
| Fact F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.015; closing index 19)
| G1 T | 0.000 | -0.000 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.003 | 0.010 | 0.021 | 0.026 | 0.049 |
| G1 R | 0.00 | -0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | 0.06 | 0.08 | 0.14 |
| G1 F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.042; closing index None)
| Concl+G1 T | 0.097 | 0.099 | 0.106 | 0.117 | 0.124 | 0.130 | 0.134 | 0.142 | 0.138 | 0.170 | 0.185 | 0.255 |
| Concl+G1 R | 0.29 | 0.29 | 0.31 | 0.35 | 0.37 | 0.38 | 0.40 | 0.42 | 0.41 | 0.50 | 0.55 | 0.76 |
| Concl+G1 F | 0.38 | 0.39 | 0.42 | 0.46 | 0.49 | 0.51 | 0.53 | 0.56 | 0.54 | 0.67 | 0.73 | 1.00 |   (F defined; closing index None)
| Rule T | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.003 | 0.004 | 0.004 | 0.004 | 0.003 | 0.002 | 0.000 |
| Rule R | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| Rule F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.000; closing index 0)
| Concl_same T | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | -0.000 | 0.001 | -0.004 | -0.000 | 0.008 | 0.014 | 0.035 |
| Concl_same R | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | 0.00 | -0.01 | -0.00 | 0.02 | 0.04 | 0.10 |
| Concl_same F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.023; closing index None)
| Concl_diff T | 0.097 | 0.099 | 0.106 | 0.118 | 0.125 | 0.129 | 0.131 | 0.139 | 0.120 | 0.113 | 0.105 | 0.118 |
| Concl_diff R | 0.29 | 0.29 | 0.31 | 0.35 | 0.37 | 0.38 | 0.39 | 0.41 | 0.35 | 0.33 | 0.31 | 0.35 |
| Concl_diff F | 0.82 | 0.84 | 0.90 | 1.00 | 1.05 | 1.09 | 1.11 | 1.17 | 1.01 | 0.95 | 0.89 | 1.00 |   (F defined; closing index None)

### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))
- clean random keys: i=0: mean 0.000, max 0.000 (n=3); i=8: mean 0.001, max 0.002 (n=3); i=14: mean 0.002, max 0.003 (n=3); i=17: mean 0.002, max 0.002 (n=3); i=19: mean 0.000, max 0.000 (n=3)
- norm-matched random direction (Concl): i=0: mean 0.010, max 0.010 (n=5); i=8: mean 0.006, max 0.006 (n=5); i=14: mean 0.026, max 0.026 (n=5); i=17: mean 0.016, max 0.016 (n=5); i=19: mean 0.008, max 0.008 (n=5)
- random head sets (Concl): i=0: mean 0.014, max 0.018 (n=3); i=14: mean 0.007, max 0.012 (n=3); i=19: mean 0.005, max 0.010 (n=3)

### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions
Concl@0: max 0.78, equal 0.81; Concl@10: max 18.12, equal 0.00; Concl@19: max 17.75, equal 0.00; Concl_same@0: max 0.00, equal 1.00; Concl_same@10: max 5.00, equal 0.00; Concl_same@19: max 6.00, equal 0.00; Fact@0: max 0.78, equal 0.77; Fact@10: max 20.69, equal 0.51; Fact@19: max 19.47, equal 0.51; Other@0: max 0.00, equal 1.00; Other@10: max 0.00, equal 1.00; Other@19: max 0.00, equal 1.00; Rule@0: max 0.00, equal 1.00; Rule@10: max 10.75, equal 0.93; Rule@19: max 10.50, equal 0.93

### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP

| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.072 | 0.098 | 0.097 | -0.073 | 0.001 | 0.001 | 0.001 | -0.001 | +0.096 |
| 1 | 0.001 | 0.008 | 0.011 | +0.003 | 0.000 | 0.000 | 0.001 | +0.000 | +0.011 |
| 2 | 0.001 | -0.000 | -0.001 | -0.001 | 0.000 | 0.001 | 0.001 | -0.000 | -0.001 |
| 3 | 0.001 | 0.002 | 0.002 | +0.000 | 0.000 | 0.001 | 0.000 | -0.001 | +0.002 |
| 4 | 0.000 | -0.001 | -0.001 | +0.000 | 0.000 | 0.001 | 0.001 | +0.000 | -0.001 |
| 5 | 0.000 | -0.001 | -0.002 | -0.001 | 0.001 | 0.001 | 0.000 | -0.001 | -0.002 |
| 6 | -0.000 | -0.001 | 0.000 | +0.001 | -0.000 | 0.000 | -0.000 | +0.000 | +0.000 |
| 7 | -0.001 | -0.004 | -0.004 | +0.001 | -0.000 | 0.001 | 0.001 | +0.000 | -0.004 |
| 8 | 0.001 | -0.000 | -0.000 | -0.001 | -0.000 | 0.002 | 0.001 | -0.000 | -0.002 |
| 9 | 0.002 | 0.003 | 0.004 | -0.000 | -0.000 | 0.001 | 0.000 | -0.000 | +0.004 |
| 10 | 0.001 | -0.002 | -0.002 | -0.001 | 0.000 | -0.000 | -0.000 | -0.000 | -0.002 |
| 11 | 0.002 | 0.004 | 0.005 | -0.000 | -0.000 | 0.004 | 0.003 | -0.001 | +0.003 |
| 12 | -0.002 | 0.002 | 0.001 | +0.001 | 0.001 | 0.000 | 0.000 | -0.001 | +0.001 |
| 13 | 0.019 | 0.003 | 0.018 | -0.004 | 0.001 | -0.000 | 0.000 | -0.000 | +0.018 |
| 14 | 0.003 | 0.004 | 0.004 | -0.003 | 0.001 | 0.003 | 0.004 | +0.000 | +0.000 |
| 15 | 0.045 | 0.025 | 0.059 | -0.011 | 0.002 | 0.005 | 0.007 | -0.001 | +0.052 |
| 16 | 0.025 | 0.027 | 0.046 | -0.006 | 0.004 | 0.010 | 0.014 | +0.000 | +0.032 |
| 17 | 0.011 | 0.022 | 0.029 | -0.004 | -0.000 | 0.019 | 0.019 | +0.000 | +0.010 |
| 18 | 0.037 | 0.066 | 0.089 | -0.015 | 0.011 | 0.066 | 0.088 | +0.012 | +0.000 |

Writers (window 0–19, rise 0.42, threshold 0.3×rise = 0.13 in F_comp units, lower bound > 0.05): total|attn: [0, 15, 16, 18]; total|mlp: [0, 15, 16, 17, 18]; total|attn+mlp: [0, 15, 16, 17, 18]; direct|attn: []; direct|mlp: [18]; direct|attn+mlp: [18]

## Set A, num_rev: value-origin curve for P = Concl (434 trials, 217 pairs, 8 contents)

Denominator T*_V(19, Concl) = 0.200 (lower bound 0.183; defined=True). Reference T_ref = T*_V(19, KV) = 0.378.

Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.

| index | T*_V (signed) [pair] / [content] | F | Δmargin (logits, signed toward source) | Δ H* readout (signed) | vproj (value change projection) | U(i): unrestricted T |
|---|---|---|---|---|---|---|
| 0 | 0.149 [0.14, 0.16] / [0.14, 0.16] | 0.75 | 3.409 | 0.244 | 0.267 | 0.551 |
| 1 | 0.149 [0.14, 0.16] / [0.14, 0.16] | 0.74 | 3.404 | 0.244 | 0.267 | 0.552 |
| 2 | 0.150 [0.14, 0.16] / [0.14, 0.16] | 0.75 | 3.440 | 0.244 | 0.268 | 0.550 |
| 3 | 0.151 [0.14, 0.16] / [0.14, 0.16] | 0.75 | 3.448 | 0.245 | 0.269 | 0.550 |
| 4 | 0.154 [0.14, 0.17] / [0.14, 0.17] | 0.77 | 3.523 | 0.249 | 0.274 | 0.549 |
| 5 | 0.154 [0.14, 0.17] / [0.14, 0.17] | 0.77 | 3.517 | 0.249 | 0.275 | 0.548 |
| 6 | 0.161 [0.15, 0.17] / [0.15, 0.17] | 0.81 | 3.692 | 0.264 | 0.287 | 0.559 |
| 7 | 0.162 [0.15, 0.17] / [0.15, 0.17] | 0.81 | 3.700 | 0.265 | 0.290 | 0.554 |
| 8 | 0.165 [0.15, 0.18] / [0.15, 0.18] | 0.82 | 3.772 | 0.273 | 0.296 | 0.560 |
| 9 | 0.168 [0.16, 0.18] / [0.16, 0.18] | 0.84 | 3.835 | 0.275 | 0.304 | 0.564 |
| 10 | 0.167 [0.15, 0.18] / [0.16, 0.18] | 0.84 | 3.831 | 0.291 | 0.317 | 0.574 |
| 11 | 0.169 [0.16, 0.18] / [0.16, 0.18] | 0.84 | 3.861 | 0.290 | 0.321 | 0.571 |
| 12 | 0.167 [0.15, 0.18] / [0.15, 0.18] | 0.83 | 3.815 | 0.312 | 0.337 | 0.580 |
| 13 | 0.164 [0.15, 0.18] / [0.15, 0.18] | 0.82 | 3.747 | 0.302 | 0.344 | 0.576 |
| 14 | 0.168 [0.15, 0.18] / [0.15, 0.18] | 0.84 | 3.840 | 0.314 | 0.369 | 0.603 |
| 15 | 0.153 [0.14, 0.17] / [0.14, 0.17] | 0.77 | 3.503 | 0.277 | 0.361 | 0.574 |
| 16 | 0.149 [0.13, 0.16] / [0.14, 0.16] | 0.75 | 3.408 | 0.270 | 0.381 | 0.563 |
| 17 | 0.161 [0.15, 0.18] / [0.15, 0.17] | 0.81 | 3.685 | 0.249 | 0.424 | 0.552 |
| 18 | 0.163 [0.15, 0.18] / [0.15, 0.17] | 0.82 | 3.729 | 0.241 | 0.455 | 0.554 |
| 19 | 0.200 [0.18, 0.22] / [0.19, 0.21] | 1.00 | 4.575 | 0.270 | 0.516 | 0.546 |

Onsets (F): θ=0.1: first 0, sustained 0; θ=0.5: first 0, sustained 0; θ=0.9: first 19, sustained 19; trajectory class: **flat**; F(4) = 0.77 pair [0.71, 0.84] / content [0.72, 0.83] (LOCO 0.75…0.79).
Head-readout transfer onset (θ=0.5): first 0, sustained 0 (within 2 of the margin onset: True).

### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)

| group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV T | 0.378 | 0.377 | 0.372 | 0.368 | 0.367 | 0.366 | 0.364 | 0.363 | 0.366 | 0.370 | 0.371 | 0.378 |
| KV R | 1.00 | 1.00 | 0.99 | 0.97 | 0.97 | 0.97 | 0.96 | 0.96 | 0.97 | 0.98 | 0.98 | 1.00 |
| KV F | 1.00 | 1.00 | 0.99 | 0.97 | 0.97 | 0.97 | 0.96 | 0.96 | 0.97 | 0.98 | 0.98 | 1.00 |   (F defined; closing index None)
| Fact T | 0.158 | 0.158 | 0.162 | 0.160 | 0.153 | 0.126 | 0.096 | 0.066 | 0.061 | 0.040 | 0.030 | 0.011 |
| Fact R | 0.42 | 0.42 | 0.43 | 0.42 | 0.41 | 0.33 | 0.25 | 0.17 | 0.16 | 0.10 | 0.08 | 0.03 |
| Fact F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.005; closing index 19)
| Concl_same T | 0.000 | 0.000 | 0.000 | -0.001 | -0.001 | -0.001 | -0.000 | -0.005 | -0.001 | 0.009 | 0.018 | 0.041 |
| Concl_same R | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.01 | -0.00 | 0.02 | 0.05 | 0.11 |
| Concl_same F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.026; closing index None)
| Concl_diff T | 0.149 | 0.150 | 0.154 | 0.162 | 0.167 | 0.170 | 0.168 | 0.171 | 0.146 | 0.136 | 0.126 | 0.140 |
| Concl_diff R | 0.39 | 0.40 | 0.41 | 0.43 | 0.44 | 0.45 | 0.44 | 0.45 | 0.39 | 0.36 | 0.33 | 0.37 |
| Concl_diff F | 1.07 | 1.08 | 1.10 | 1.16 | 1.20 | 1.22 | 1.20 | 1.22 | 1.04 | 0.97 | 0.90 | 1.00 |   (F defined; closing index None)

### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))
- clean random keys: i=0: mean 0.000, max 0.000 (n=3); i=8: mean 0.001, max 0.001 (n=3); i=14: mean 0.003, max 0.003 (n=3); i=17: mean 0.001, max 0.002 (n=3); i=19: mean 0.000, max 0.000 (n=3)
- norm-matched random direction (Concl): i=0: mean 0.011, max 0.011 (n=5); i=8: mean 0.006, max 0.006 (n=5); i=14: mean 0.030, max 0.030 (n=5); i=17: mean 0.021, max 0.021 (n=5); i=19: mean 0.011, max 0.011 (n=5)

### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions
Concl@0: max 0.78, equal 0.81; Concl@10: max 17.75, equal 0.00; Concl@19: max 17.66, equal 0.00; Concl_same@0: max 0.00, equal 1.00; Concl_same@10: max 4.88, equal 0.00; Concl_same@19: max 6.00, equal 0.00; Fact@0: max 0.78, equal 0.77; Fact@10: max 20.31, equal 0.51; Fact@19: max 19.48, equal 0.51; Other@0: max 0.00, equal 1.00; Other@10: max 0.00, equal 1.00; Other@19: max 0.00, equal 1.00

### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP

| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.093 | 0.150 | nan | +nan | -0.000 | -0.000 | nan | +nan | +nan |
| 1 | 0.001 | 0.009 | nan | +nan | 0.000 | -0.001 | nan | +nan | +nan |
| 2 | -0.000 | -0.000 | nan | +nan | -0.001 | 0.001 | nan | +nan | +nan |
| 3 | -0.001 | -0.000 | nan | +nan | -0.000 | -0.000 | nan | +nan | +nan |
| 4 | 0.001 | -0.002 | nan | +nan | -0.000 | 0.000 | nan | +nan | +nan |
| 5 | -0.000 | -0.001 | nan | +nan | -0.000 | 0.000 | nan | +nan | +nan |
| 6 | 0.000 | -0.001 | nan | +nan | -0.001 | 0.000 | nan | +nan | +nan |
| 7 | -0.002 | -0.006 | nan | +nan | -0.000 | 0.001 | nan | +nan | +nan |
| 8 | 0.001 | -0.001 | nan | +nan | -0.001 | 0.001 | nan | +nan | +nan |
| 9 | 0.003 | 0.003 | nan | +nan | -0.000 | -0.000 | nan | +nan | +nan |
| 10 | 0.001 | -0.002 | nan | +nan | 0.000 | -0.001 | nan | +nan | +nan |
| 11 | 0.002 | 0.004 | nan | +nan | -0.001 | 0.004 | nan | +nan | +nan |
| 12 | -0.002 | 0.001 | nan | +nan | 0.001 | 0.000 | nan | +nan | +nan |
| 13 | 0.024 | 0.001 | nan | +nan | 0.000 | -0.001 | nan | +nan | +nan |
| 14 | 0.002 | 0.002 | nan | +nan | 0.001 | 0.002 | nan | +nan | +nan |
| 15 | 0.051 | 0.026 | nan | +nan | 0.002 | 0.005 | nan | +nan | +nan |
| 16 | 0.030 | 0.030 | nan | +nan | 0.006 | 0.010 | nan | +nan | +nan |
| 17 | 0.014 | 0.027 | nan | +nan | 0.001 | 0.023 | nan | +nan | +nan |
| 18 | 0.044 | 0.078 | nan | +nan | 0.012 | 0.077 | nan | +nan | +nan |

Writers (window 0–19, rise 0.25, threshold 0.3×rise = 0.08 in F_comp units, lower bound > 0.05): total|attn: [0, 13, 15, 16, 18]; total|mlp: [0, 15, 16, 17, 18]; direct|attn: []; direct|mlp: [17, 18]

## Controls on set A (num_orig): RMS of the H* read-out change relative to the matched pair M (ρ_ctrl); point [pair] / [content]

| source | group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X | Concl | 1.05 | 1.05 | 0.96 | 0.91 | 0.88 | 0.85 | 0.81 | 0.84 | 0.94 | 0.97 | 0.95 | 0.98 |
| X | KV | 0.86 | 0.87 | 0.83 | 0.82 | 0.80 | 0.82 | 0.83 | 0.93 | 1.00 | 1.01 | 1.01 | 1.01 |
| L | Concl | 0.96 | 0.96 | 0.83 | 0.74 | 0.67 | 0.52 | 0.48 | 0.33 | 0.35 | 0.35 | 0.35 | 0.35 |
| L | KV | 0.75 | 0.75 | 0.70 | 0.67 | 0.63 | 0.51 | 0.47 | 0.32 | 0.29 | 0.30 | 0.30 | 0.31 |
| Fm | Concl | 1.44 | 1.44 | 1.43 | 1.32 | 1.25 | 1.17 | 0.98 | 0.92 | 0.92 | 0.80 | 0.68 | 0.57 |
| Fm | KV | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.69 | 0.69 | 0.69 | 0.68 | 0.68 | 0.68 | 0.68 |

Signed head-readout transfer, group Concl: M 0:0.52, 2:0.52, 4:0.53, 6:0.56, 8:0.58, 10:0.62, 12:0.66, 14:0.66, 16:0.56, 17:0.52, 18:0.50, 19:0.56; X 0:0.42, 2:0.42, 4:0.42, 6:0.43, 8:0.44, 10:0.47, 12:0.48, 14:0.52, 16:0.52, 17:0.49, 18:0.47, 19:0.54

Signed head-readout transfer, group KV: M 0:0.91, 2:0.91, 4:0.90, 6:0.90, 8:0.89, 10:0.89, 12:0.89, 14:0.89, 16:0.89, 17:0.90, 18:0.90, 19:0.91; X 0:0.49, 2:0.49, 4:0.50, 6:0.52, 8:0.53, 10:0.62, 12:0.64, 14:0.77, 16:0.87, 17:0.89, 18:0.88, 19:0.90

## Controls on set A (num_rev): RMS of the H* read-out change relative to the matched pair M (ρ_ctrl); point [pair] / [content]

| source | group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fm | Concl | 1.48 | 1.48 | 1.47 | 1.36 | 1.29 | 1.21 | 1.02 | 0.95 | 0.93 | 0.81 | 0.69 | 0.61 |
| Fm | KV | 0.71 | 0.71 | 0.71 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.71 | 0.71 | 0.71 | 0.71 |

## Independent set B, num_orig: value-origin curve for P = Concl (400 trials, 200 pairs, 43 contents)

Denominator T*_V(19, Concl) = 0.179 (lower bound 0.164; defined=True). Reference T_ref = T*_V(19, KV) = 0.294.

Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.

| index | T*_V (signed) [pair] / [content] | F | Δmargin (logits, signed toward source) | Δ H* readout (signed) | vproj (value change projection) | U(i): unrestricted T |
|---|---|---|---|---|---|---|
| 0 | 0.118 [0.11, 0.13] / [0.10, 0.13] | 0.66 | 3.235 | 0.304 | 0.312 | 0.538 |
| 2 | 0.119 [0.11, 0.13] / [0.11, 0.13] | 0.66 | 3.256 | 0.305 | 0.314 | 0.538 |
| 4 | 0.132 [0.12, 0.15] / [0.12, 0.15] | 0.74 | 3.628 | 0.319 | 0.331 | 0.539 |
| 6 | 0.141 [0.13, 0.16] / [0.13, 0.16] | 0.79 | 3.874 | 0.332 | 0.344 | 0.545 |
| 8 | 0.145 [0.13, 0.16] / [0.13, 0.16] | 0.81 | 3.978 | 0.339 | 0.350 | 0.547 |
| 10 | 0.151 [0.14, 0.17] / [0.13, 0.17] | 0.84 | 4.133 | 0.359 | 0.374 | 0.541 |
| 12 | 0.156 [0.14, 0.17] / [0.14, 0.18] | 0.87 | 4.279 | 0.371 | 0.394 | 0.526 |
| 14 | 0.152 [0.14, 0.17] / [0.14, 0.17] | 0.85 | 4.184 | 0.353 | 0.413 | 0.517 |
| 16 | 0.145 [0.13, 0.16] / [0.13, 0.16] | 0.81 | 3.991 | 0.325 | 0.441 | 0.517 |
| 17 | 0.144 [0.13, 0.16] / [0.13, 0.16] | 0.80 | 3.944 | 0.296 | 0.466 | 0.531 |
| 18 | 0.150 [0.14, 0.16] / [0.14, 0.17] | 0.84 | 4.122 | 0.290 | 0.492 | 0.535 |
| 19 | 0.179 [0.17, 0.19] / [0.16, 0.19] | 1.00 | 4.919 | 0.313 | 0.540 | 0.551 |

Onsets (F): θ=0.1: first 0, sustained 0; θ=0.5: first 0, sustained 0; θ=0.9: first 19, sustained 19; trajectory class: **flat**; F(4) = 0.74 pair [0.67, 0.81] / content [0.67, 0.81] (LOCO 0.72…0.75).
Head-readout transfer onset (θ=0.5): first 0, sustained 0 (within 2 of the margin onset: True).

### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)

| group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV T | 0.294 | 0.294 | 0.292 | 0.289 | 0.288 | 0.287 | 0.285 | 0.283 | 0.287 | 0.288 | 0.289 | 0.294 |
| KV R | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.96 | 0.98 | 0.98 | 0.98 | 1.00 |
| KV F | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.96 | 0.98 | 0.98 | 0.98 | 1.00 |   (F defined; closing index None)
| Fact T | 0.113 | 0.113 | 0.113 | 0.108 | 0.102 | 0.092 | 0.071 | 0.064 | 0.060 | 0.051 | 0.038 | 0.015 |
| Fact R | 0.38 | 0.38 | 0.38 | 0.37 | 0.35 | 0.31 | 0.24 | 0.22 | 0.20 | 0.17 | 0.13 | 0.05 |
| Fact F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.011; closing index 19)
| Concl_same T | 0.000 | 0.000 | 0.000 | -0.000 | -0.001 | -0.000 | 0.000 | 0.001 | 0.001 | 0.005 | 0.009 | 0.014 |
| Concl_same R | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.03 | 0.05 |
| Concl_same F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.011; closing index 0)
| Concl_diff T | 0.118 | 0.119 | 0.133 | 0.142 | 0.146 | 0.151 | 0.156 | 0.150 | 0.142 | 0.135 | 0.137 | 0.153 |
| Concl_diff R | 0.40 | 0.41 | 0.45 | 0.48 | 0.50 | 0.52 | 0.53 | 0.51 | 0.48 | 0.46 | 0.47 | 0.52 |
| Concl_diff F | 0.77 | 0.78 | 0.87 | 0.93 | 0.96 | 0.99 | 1.02 | 0.98 | 0.93 | 0.88 | 0.89 | 1.00 |   (F defined; closing index None)

### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))
- clean random keys: i=0: mean 0.000, max 0.000 (n=1); i=8: mean 0.002, max 0.002 (n=1); i=14: mean 0.005, max 0.005 (n=1); i=17: mean 0.001, max 0.001 (n=1); i=19: mean 0.000, max 0.000 (n=1)
- norm-matched random direction (Concl): i=0: mean 0.003, max 0.003 (n=5); i=8: mean 0.010, max 0.010 (n=5); i=14: mean 0.024, max 0.024 (n=5); i=17: mean 0.014, max 0.014 (n=5); i=19: mean 0.005, max 0.005 (n=5)

### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions
Concl@0: max 0.78, equal 0.67; Concl@10: max 28.72, equal 0.00; Concl@19: max 26.50, equal 0.00; Concl_same@0: max 0.00, equal 1.00; Concl_same@10: max 14.12, equal 0.00; Concl_same@19: max 12.12, equal 0.00; Fact@0: max 0.78, equal 0.61; Fact@10: max 29.09, equal 0.45; Fact@19: max 27.25, equal 0.45; Other@0: max 0.00, equal 1.00; Other@10: max 0.00, equal 1.00; Other@19: max 0.00, equal 1.00

### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP

| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.064 | 0.122 | nan | +nan | nan | nan | nan | +nan | +nan |
| 1 | 0.006 | 0.003 | nan | +nan | nan | nan | nan | +nan | +nan |
| 2 | 0.001 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 3 | 0.001 | 0.003 | nan | +nan | nan | nan | nan | +nan | +nan |
| 4 | 0.001 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 5 | 0.002 | 0.004 | nan | +nan | nan | nan | nan | +nan | +nan |
| 6 | 0.001 | 0.002 | nan | +nan | nan | nan | nan | +nan | +nan |
| 7 | 0.000 | -0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 8 | 0.001 | 0.002 | nan | +nan | nan | nan | nan | +nan | +nan |
| 9 | 0.005 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 10 | 0.001 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 11 | 0.006 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 12 | 0.006 | 0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 13 | 0.018 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 14 | 0.002 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 15 | 0.032 | 0.017 | nan | +nan | nan | nan | nan | +nan | +nan |
| 16 | 0.020 | 0.027 | nan | +nan | nan | nan | nan | +nan | +nan |
| 17 | 0.016 | 0.028 | nan | +nan | nan | nan | nan | +nan | +nan |
| 18 | 0.023 | 0.067 | nan | +nan | nan | nan | nan | +nan | +nan |

Writers (window 0–19, rise 0.34, threshold 0.3×rise = 0.10 in F_comp units, lower bound > 0.05): total|attn: [0, 15, 16, 18]; total|mlp: [0, 16, 17, 18]

## Independent set B, num_rev: value-origin curve for P = Concl (400 trials, 200 pairs, 43 contents)

Denominator T*_V(19, Concl) = 0.209 (lower bound 0.192; defined=True). Reference T_ref = T*_V(19, KV) = 0.330.

Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.

| index | T*_V (signed) [pair] / [content] | F | Δmargin (logits, signed toward source) | Δ H* readout (signed) | vproj (value change projection) | U(i): unrestricted T |
|---|---|---|---|---|---|---|
| 0 | 0.161 [0.15, 0.17] / [0.15, 0.18] | 0.77 | 3.661 | 0.308 | 0.311 | 0.563 |
| 2 | 0.163 [0.15, 0.18] / [0.15, 0.18] | 0.78 | 3.698 | 0.308 | 0.313 | 0.564 |
| 4 | 0.178 [0.16, 0.19] / [0.16, 0.19] | 0.85 | 4.040 | 0.323 | 0.330 | 0.567 |
| 6 | 0.185 [0.17, 0.20] / [0.17, 0.20] | 0.88 | 4.197 | 0.337 | 0.343 | 0.578 |
| 8 | 0.190 [0.18, 0.21] / [0.17, 0.21] | 0.91 | 4.316 | 0.345 | 0.349 | 0.583 |
| 10 | 0.192 [0.18, 0.21] / [0.17, 0.21] | 0.92 | 4.366 | 0.365 | 0.373 | 0.575 |
| 12 | 0.189 [0.17, 0.21] / [0.17, 0.21] | 0.91 | 4.308 | 0.377 | 0.394 | 0.552 |
| 14 | 0.182 [0.17, 0.20] / [0.16, 0.20] | 0.87 | 4.139 | 0.360 | 0.413 | 0.542 |
| 16 | 0.173 [0.16, 0.19] / [0.16, 0.19] | 0.83 | 3.936 | 0.333 | 0.441 | 0.543 |
| 17 | 0.171 [0.16, 0.18] / [0.15, 0.19] | 0.82 | 3.888 | 0.305 | 0.467 | 0.558 |
| 18 | 0.177 [0.16, 0.19] / [0.16, 0.19] | 0.85 | 4.030 | 0.300 | 0.494 | 0.566 |
| 19 | 0.209 [0.20, 0.22] / [0.19, 0.23] | 1.00 | 4.750 | 0.321 | 0.541 | 0.585 |

Onsets (F): θ=0.1: first 0, sustained 0; θ=0.5: first 0, sustained 0; θ=0.9: first 8, sustained 19; trajectory class: **flat**; F(4) = 0.85 pair [0.79, 0.92] / content [0.78, 0.93] (LOCO 0.84…0.87).
Head-readout transfer onset (θ=0.5): first 0, sustained 0 (within 2 of the margin onset: True).

### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)

| group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV T | 0.330 | 0.329 | 0.327 | 0.325 | 0.323 | 0.322 | 0.321 | 0.319 | 0.322 | 0.324 | 0.325 | 0.330 |
| KV R | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.98 | 0.98 | 0.99 | 1.00 |
| KV F | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.98 | 0.98 | 0.99 | 1.00 |   (F defined; closing index None)
| Fact T | 0.137 | 0.137 | 0.135 | 0.127 | 0.118 | 0.104 | 0.078 | 0.070 | 0.064 | 0.053 | 0.037 | 0.010 |
| Fact R | 0.42 | 0.42 | 0.41 | 0.38 | 0.36 | 0.31 | 0.24 | 0.21 | 0.19 | 0.16 | 0.11 | 0.03 |
| Fact F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.005; closing index 19)
| Concl_same T | 0.000 | 0.000 | 0.000 | -0.001 | -0.001 | -0.001 | -0.000 | -0.001 | 0.001 | 0.007 | 0.011 | 0.017 |
| Concl_same R | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | 0.02 | 0.03 | 0.05 |
| Concl_same F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.012; closing index 0)
| Concl_diff T | 0.161 | 0.162 | 0.178 | 0.185 | 0.191 | 0.193 | 0.190 | 0.180 | 0.169 | 0.159 | 0.161 | 0.178 |
| Concl_diff R | 0.49 | 0.49 | 0.54 | 0.56 | 0.58 | 0.58 | 0.57 | 0.55 | 0.51 | 0.48 | 0.49 | 0.54 |
| Concl_diff F | 0.90 | 0.91 | 1.00 | 1.04 | 1.07 | 1.08 | 1.06 | 1.01 | 0.95 | 0.89 | 0.90 | 1.00 |   (F defined; closing index None)

### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))
- clean random keys: i=0: mean 0.000, max 0.000 (n=1); i=8: mean 0.002, max 0.002 (n=1); i=14: mean 0.005, max 0.005 (n=1); i=17: mean 0.000, max 0.000 (n=1); i=19: mean 0.000, max 0.000 (n=1)
- norm-matched random direction (Concl): i=0: mean 0.003, max 0.003 (n=5); i=8: mean 0.012, max 0.012 (n=5); i=14: mean 0.028, max 0.028 (n=5); i=17: mean 0.015, max 0.015 (n=5); i=19: mean 0.006, max 0.006 (n=5)

### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions
Concl@0: max 0.78, equal 0.67; Concl@10: max 29.02, equal 0.00; Concl@19: max 26.62, equal 0.00; Concl_same@0: max 0.00, equal 1.00; Concl_same@10: max 13.50, equal 0.00; Concl_same@19: max 12.06, equal 0.00; Fact@0: max 0.78, equal 0.61; Fact@10: max 29.19, equal 0.45; Fact@19: max 27.38, equal 0.45; Other@0: max 0.00, equal 1.00; Other@10: max 0.00, equal 1.00; Other@19: max 0.00, equal 1.00

### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP

| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.072 | 0.165 | nan | +nan | nan | nan | nan | +nan | +nan |
| 1 | 0.007 | 0.004 | nan | +nan | nan | nan | nan | +nan | +nan |
| 2 | 0.002 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 3 | 0.001 | 0.003 | nan | +nan | nan | nan | nan | +nan | +nan |
| 4 | 0.000 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 5 | 0.001 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 6 | 0.002 | 0.002 | nan | +nan | nan | nan | nan | +nan | +nan |
| 7 | -0.000 | -0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 8 | 0.001 | 0.002 | nan | +nan | nan | nan | nan | +nan | +nan |
| 9 | 0.006 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 10 | 0.001 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 11 | 0.006 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 12 | 0.006 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 13 | 0.023 | 0.007 | nan | +nan | nan | nan | nan | +nan | +nan |
| 14 | 0.003 | 0.007 | nan | +nan | nan | nan | nan | +nan | +nan |
| 15 | 0.036 | 0.019 | nan | +nan | nan | nan | nan | +nan | +nan |
| 16 | 0.025 | 0.032 | nan | +nan | nan | nan | nan | +nan | +nan |
| 17 | 0.019 | 0.034 | nan | +nan | nan | nan | nan | +nan | +nan |
| 18 | 0.024 | 0.074 | nan | +nan | nan | nan | nan | +nan | +nan |

Writers (window 0–19, rise 0.23, threshold 0.3×rise = 0.07 in F_comp units, lower bound > 0.05): total|attn: [0, 13, 15, 16, 17, 18]; total|mlp: [0, 15, 16, 17, 18]

## Controls on set B (num_orig): RMS of the H* read-out change relative to the matched pair M (ρ_ctrl); point [pair] / [content]

| source | group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | Concl | 0.81 | 0.81 | 0.75 | 0.68 | 0.57 | 0.42 | 0.36 | 0.27 | 0.28 | 0.30 | 0.29 | 0.28 |
| L | KV | 0.63 | 0.62 | 0.61 | 0.59 | 0.52 | 0.38 | 0.34 | 0.23 | 0.22 | 0.22 | 0.22 | 0.22 |
| X | Concl | 1.03 | 1.03 | 1.00 | 0.96 | 0.94 | 0.90 | 0.89 | 0.94 | 1.00 | 1.05 | 1.06 | 1.12 |
| X | KV | 0.84 | 0.83 | 0.84 | 0.84 | 0.82 | 0.86 | 0.89 | 1.01 | 1.06 | 1.06 | 1.06 | 1.06 |

Signed head-readout transfer, group Concl: M 0:0.62, 2:0.62, 4:0.65, 6:0.67, 8:0.69, 10:0.73, 12:0.75, 14:0.72, 16:0.66, 17:0.60, 18:0.59, 19:0.64; X 0:0.51, 2:0.51, 4:0.51, 6:0.52, 8:0.55, 10:0.57, 12:0.59, 14:0.63, 16:0.63, 17:0.60, 18:0.60, 19:0.67

Signed head-readout transfer, group KV: M 0:0.96, 2:0.96, 4:0.95, 6:0.94, 8:0.94, 10:0.93, 12:0.93, 14:0.92, 16:0.93, 17:0.94, 18:0.95, 19:0.96; X 0:0.57, 2:0.57, 4:0.58, 6:0.59, 8:0.62, 10:0.70, 12:0.74, 14:0.85, 16:0.92, 17:0.92, 18:0.93, 19:0.94

## Independent set C, num_orig: value-origin curve for P = Concl (400 trials, 200 pairs, 43 contents)

Denominator T*_V(19, Concl) = 0.162 (lower bound 0.150; defined=True). Reference T_ref = T*_V(19, KV) = 0.283.

Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.

| index | T*_V (signed) [pair] / [content] | F | Δmargin (logits, signed toward source) | Δ H* readout (signed) | vproj (value change projection) | U(i): unrestricted T |
|---|---|---|---|---|---|---|
| 0 | 0.121 [0.11, 0.13] / [0.11, 0.14] | 0.74 | 3.263 | 0.289 | 0.306 | 0.539 |
| 2 | 0.122 [0.11, 0.14] / [0.11, 0.14] | 0.75 | 3.306 | 0.290 | 0.309 | 0.539 |
| 4 | 0.133 [0.12, 0.15] / [0.12, 0.15] | 0.82 | 3.597 | 0.305 | 0.325 | 0.528 |
| 6 | 0.142 [0.13, 0.16] / [0.13, 0.16] | 0.87 | 3.833 | 0.323 | 0.340 | 0.533 |
| 8 | 0.143 [0.13, 0.16] / [0.13, 0.16] | 0.88 | 3.874 | 0.326 | 0.345 | 0.531 |
| 10 | 0.149 [0.13, 0.16] / [0.13, 0.17] | 0.92 | 4.034 | 0.347 | 0.368 | 0.531 |
| 12 | 0.149 [0.13, 0.16] / [0.13, 0.17] | 0.92 | 4.022 | 0.353 | 0.387 | 0.517 |
| 14 | 0.138 [0.12, 0.15] / [0.12, 0.16] | 0.85 | 3.721 | 0.338 | 0.408 | 0.511 |
| 16 | 0.129 [0.12, 0.14] / [0.12, 0.14] | 0.79 | 3.486 | 0.310 | 0.434 | 0.501 |
| 17 | 0.126 [0.11, 0.14] / [0.12, 0.14] | 0.78 | 3.408 | 0.277 | 0.459 | 0.504 |
| 18 | 0.135 [0.12, 0.15] / [0.12, 0.15] | 0.83 | 3.651 | 0.276 | 0.487 | 0.505 |
| 19 | 0.162 [0.15, 0.17] / [0.15, 0.17] | 1.00 | 4.389 | 0.295 | 0.534 | 0.514 |

Onsets (F): θ=0.1: first 0, sustained 0; θ=0.5: first 0, sustained 0; θ=0.9: first 10, sustained 19; trajectory class: **flat**; F(4) = 0.82 pair [0.74, 0.90] / content [0.73, 0.93] (LOCO 0.79…0.83).
Head-readout transfer onset (θ=0.5): first 0, sustained 0 (within 2 of the margin onset: True).

### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)

| group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV T | 0.283 | 0.283 | 0.279 | 0.277 | 0.277 | 0.276 | 0.275 | 0.273 | 0.276 | 0.278 | 0.280 | 0.283 |
| KV R | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.97 | 0.98 | 0.99 | 1.00 |
| KV F | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.97 | 0.98 | 0.99 | 1.00 |   (F defined; closing index None)
| Fact T | 0.118 | 0.119 | 0.116 | 0.106 | 0.101 | 0.090 | 0.068 | 0.060 | 0.059 | 0.052 | 0.039 | 0.016 |
| Fact R | 0.42 | 0.42 | 0.41 | 0.37 | 0.36 | 0.32 | 0.24 | 0.21 | 0.21 | 0.19 | 0.14 | 0.06 |
| Fact F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.012; closing index 19)
| Concl_same T | 0.000 | 0.000 | 0.001 | -0.000 | -0.000 | -0.000 | -0.000 | 0.000 | 0.001 | 0.004 | 0.009 | 0.012 |
| Concl_same R | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 0.02 | 0.03 | 0.04 |
| Concl_same F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.009; closing index 0)
| Concl_diff T | 0.121 | 0.122 | 0.133 | 0.142 | 0.143 | 0.149 | 0.149 | 0.138 | 0.128 | 0.118 | 0.122 | 0.138 |
| Concl_diff R | 0.43 | 0.43 | 0.47 | 0.50 | 0.51 | 0.53 | 0.53 | 0.49 | 0.45 | 0.42 | 0.43 | 0.49 |
| Concl_diff F | 0.87 | 0.88 | 0.96 | 1.02 | 1.04 | 1.07 | 1.07 | 0.99 | 0.92 | 0.86 | 0.88 | 1.00 |   (F defined; closing index None)

### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))
- clean random keys: i=0: mean 0.000, max 0.000 (n=1); i=8: mean 0.002, max 0.002 (n=1); i=14: mean 0.003, max 0.003 (n=1); i=17: mean 0.001, max 0.001 (n=1); i=19: mean 0.000, max 0.000 (n=1)
- norm-matched random direction (Concl): i=0: mean 0.006, max 0.006 (n=5); i=8: mean 0.005, max 0.005 (n=5); i=14: mean 0.024, max 0.024 (n=5); i=17: mean 0.017, max 0.017 (n=5); i=19: mean 0.004, max 0.004 (n=5)

### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions
Concl@0: max 0.80, equal 0.67; Concl@10: max 23.44, equal 0.00; Concl@19: max 32.25, equal 0.00; Concl_same@0: max 0.00, equal 1.00; Concl_same@10: max 6.00, equal 0.00; Concl_same@19: max 10.25, equal 0.00; Fact@0: max 0.80, equal 0.61; Fact@10: max 27.53, equal 0.45; Fact@19: max 37.38, equal 0.45; Other@0: max 0.00, equal 1.00; Other@10: max 0.00, equal 1.00; Other@19: max 0.00, equal 1.00

### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP

| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.066 | 0.122 | nan | +nan | nan | nan | nan | +nan | +nan |
| 1 | 0.006 | 0.003 | nan | +nan | nan | nan | nan | +nan | +nan |
| 2 | 0.002 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 3 | 0.003 | 0.002 | nan | +nan | nan | nan | nan | +nan | +nan |
| 4 | 0.001 | 0.007 | nan | +nan | nan | nan | nan | +nan | +nan |
| 5 | 0.002 | 0.004 | nan | +nan | nan | nan | nan | +nan | +nan |
| 6 | 0.002 | 0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 7 | 0.001 | 0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 8 | 0.001 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 9 | 0.006 | -0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 10 | 0.000 | 0.004 | nan | +nan | nan | nan | nan | +nan | +nan |
| 11 | 0.003 | 0.007 | nan | +nan | nan | nan | nan | +nan | +nan |
| 12 | 0.009 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 13 | 0.013 | 0.006 | nan | +nan | nan | nan | nan | +nan | +nan |
| 14 | 0.004 | 0.012 | nan | +nan | nan | nan | nan | +nan | +nan |
| 15 | 0.027 | 0.013 | nan | +nan | nan | nan | nan | +nan | +nan |
| 16 | 0.019 | 0.025 | nan | +nan | nan | nan | nan | +nan | +nan |
| 17 | 0.018 | 0.023 | nan | +nan | nan | nan | nan | +nan | +nan |
| 18 | 0.017 | 0.057 | nan | +nan | nan | nan | nan | +nan | +nan |

Writers (window 0–19, rise 0.26, threshold 0.3×rise = 0.08 in F_comp units, lower bound > 0.05): total|attn: [0, 13, 15, 16, 17, 18]; total|mlp: [0, 16, 17, 18]

## Independent set C, num_rev: value-origin curve for P = Concl (400 trials, 200 pairs, 43 contents)

Denominator T*_V(19, Concl) = 0.191 (lower bound 0.177; defined=True). Reference T_ref = T*_V(19, KV) = 0.313.

Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.

| index | T*_V (signed) [pair] / [content] | F | Δmargin (logits, signed toward source) | Δ H* readout (signed) | vproj (value change projection) | U(i): unrestricted T |
|---|---|---|---|---|---|---|
| 0 | 0.161 [0.15, 0.18] / [0.15, 0.18] | 0.85 | 3.625 | 0.293 | 0.306 | 0.568 |
| 2 | 0.164 [0.15, 0.18] / [0.15, 0.18] | 0.86 | 3.677 | 0.293 | 0.308 | 0.567 |
| 4 | 0.177 [0.16, 0.19] / [0.16, 0.20] | 0.92 | 3.963 | 0.310 | 0.325 | 0.557 |
| 6 | 0.184 [0.17, 0.20] / [0.17, 0.20] | 0.96 | 4.121 | 0.328 | 0.339 | 0.568 |
| 8 | 0.184 [0.17, 0.20] / [0.17, 0.21] | 0.97 | 4.138 | 0.331 | 0.345 | 0.567 |
| 10 | 0.187 [0.17, 0.20] / [0.17, 0.21] | 0.98 | 4.198 | 0.351 | 0.367 | 0.567 |
| 12 | 0.181 [0.16, 0.20] / [0.16, 0.21] | 0.95 | 4.058 | 0.360 | 0.386 | 0.544 |
| 14 | 0.165 [0.15, 0.18] / [0.15, 0.19] | 0.86 | 3.698 | 0.345 | 0.407 | 0.538 |
| 16 | 0.155 [0.14, 0.17] / [0.14, 0.17] | 0.81 | 3.485 | 0.318 | 0.435 | 0.527 |
| 17 | 0.152 [0.14, 0.17] / [0.14, 0.16] | 0.79 | 3.406 | 0.285 | 0.461 | 0.533 |
| 18 | 0.160 [0.15, 0.17] / [0.15, 0.17] | 0.84 | 3.598 | 0.284 | 0.489 | 0.536 |
| 19 | 0.191 [0.18, 0.20] / [0.18, 0.20] | 1.00 | 4.285 | 0.302 | 0.537 | 0.547 |

Onsets (F): θ=0.1: first 0, sustained 0; θ=0.5: first 0, sustained 0; θ=0.9: first 4, sustained 19; trajectory class: **non-monotone**; F(4) = 0.92 pair [0.85, 1.00] / content [0.83, 1.04] (LOCO 0.90…0.94).
Head-readout transfer onset (θ=0.5): first 0, sustained 0 (within 2 of the margin onset: True).

### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)

| group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KV T | 0.313 | 0.312 | 0.308 | 0.307 | 0.306 | 0.305 | 0.304 | 0.303 | 0.305 | 0.308 | 0.309 | 0.313 |
| KV R | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.98 | 0.98 | 0.99 | 1.00 |
| KV F | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.98 | 0.98 | 0.99 | 1.00 |   (F defined; closing index None)
| Fact T | 0.140 | 0.141 | 0.135 | 0.121 | 0.114 | 0.100 | 0.072 | 0.065 | 0.062 | 0.054 | 0.039 | 0.010 |
| Fact R | 0.45 | 0.45 | 0.43 | 0.39 | 0.37 | 0.32 | 0.23 | 0.21 | 0.20 | 0.17 | 0.12 | 0.03 |
| Fact F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.005; closing index 19)
| Concl_same T | 0.000 | 0.001 | 0.001 | -0.000 | 0.000 | -0.001 | -0.000 | 0.000 | 0.001 | 0.007 | 0.010 | 0.015 |
| Concl_same R | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 0.02 | 0.03 | 0.05 |
| Concl_same F | — | — | — | — | — | — | — | — | — | — | — | — |   (F UNDEFINED: denominator lower bound 0.012; closing index 0)
| Concl_diff T | 0.161 | 0.164 | 0.176 | 0.183 | 0.185 | 0.188 | 0.182 | 0.164 | 0.152 | 0.142 | 0.144 | 0.162 |
| Concl_diff R | 0.52 | 0.52 | 0.56 | 0.59 | 0.59 | 0.60 | 0.58 | 0.53 | 0.49 | 0.45 | 0.46 | 0.52 |
| Concl_diff F | 1.00 | 1.01 | 1.09 | 1.13 | 1.14 | 1.16 | 1.12 | 1.02 | 0.94 | 0.88 | 0.89 | 1.00 |   (F defined; closing index None)

### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))
- clean random keys: i=0: mean 0.000, max 0.000 (n=1); i=8: mean 0.002, max 0.002 (n=1); i=14: mean 0.003, max 0.003 (n=1); i=17: mean 0.001, max 0.001 (n=1); i=19: mean 0.000, max 0.000 (n=1)
- norm-matched random direction (Concl): i=0: mean 0.006, max 0.006 (n=5); i=8: mean 0.009, max 0.009 (n=5); i=14: mean 0.029, max 0.029 (n=5); i=17: mean 0.020, max 0.020 (n=5); i=19: mean 0.004, max 0.004 (n=5)

### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions
Concl@0: max 0.80, equal 0.67; Concl@10: max 23.34, equal 0.00; Concl@19: max 31.75, equal 0.00; Concl_same@0: max 0.00, equal 1.00; Concl_same@10: max 6.12, equal 0.00; Concl_same@19: max 10.75, equal 0.00; Fact@0: max 0.80, equal 0.61; Fact@10: max 27.34, equal 0.45; Fact@19: max 37.00, equal 0.45; Other@0: max 0.00, equal 1.00; Other@10: max 0.00, equal 1.00; Other@19: max 0.00, equal 1.00

### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP

| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.076 | 0.162 | nan | +nan | nan | nan | nan | +nan | +nan |
| 1 | 0.007 | 0.004 | nan | +nan | nan | nan | nan | +nan | +nan |
| 2 | 0.003 | 0.005 | nan | +nan | nan | nan | nan | +nan | +nan |
| 3 | 0.004 | 0.003 | nan | +nan | nan | nan | nan | +nan | +nan |
| 4 | 0.001 | 0.008 | nan | +nan | nan | nan | nan | +nan | +nan |
| 5 | 0.002 | 0.006 | nan | +nan | nan | nan | nan | +nan | +nan |
| 6 | 0.003 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 7 | 0.001 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 8 | 0.002 | 0.002 | nan | +nan | nan | nan | nan | +nan | +nan |
| 9 | 0.008 | 0.000 | nan | +nan | nan | nan | nan | +nan | +nan |
| 10 | 0.001 | 0.004 | nan | +nan | nan | nan | nan | +nan | +nan |
| 11 | 0.005 | 0.009 | nan | +nan | nan | nan | nan | +nan | +nan |
| 12 | 0.011 | 0.001 | nan | +nan | nan | nan | nan | +nan | +nan |
| 13 | 0.017 | 0.007 | nan | +nan | nan | nan | nan | +nan | +nan |
| 14 | 0.005 | 0.013 | nan | +nan | nan | nan | nan | +nan | +nan |
| 15 | 0.031 | 0.017 | nan | +nan | nan | nan | nan | +nan | +nan |
| 16 | 0.023 | 0.031 | nan | +nan | nan | nan | nan | +nan | +nan |
| 17 | 0.022 | 0.029 | nan | +nan | nan | nan | nan | +nan | +nan |
| 18 | 0.019 | 0.064 | nan | +nan | nan | nan | nan | +nan | +nan |

Writers (window 0–19, rise 0.15, threshold 0.3×rise = 0.05 in F_comp units, lower bound > 0.05): total|attn: [0, 13, 15, 16, 17, 18]; total|mlp: [0, 15, 16, 17, 18]

## Controls on set C (num_orig): RMS of the H* read-out change relative to the matched pair M (ρ_ctrl); point [pair] / [content]

| source | group | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | Concl | 0.84 | 0.83 | 0.79 | 0.71 | 0.59 | 0.41 | 0.37 | 0.26 | 0.26 | 0.28 | 0.27 | 0.28 |
| L | KV | 0.63 | 0.62 | 0.60 | 0.59 | 0.52 | 0.38 | 0.34 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 |
| X | Concl | 1.10 | 1.10 | 1.08 | 1.05 | 1.05 | 1.00 | 1.01 | 1.06 | 1.12 | 1.17 | 1.16 | 1.21 |
| X | KV | 0.87 | 0.87 | 0.88 | 0.87 | 0.86 | 0.88 | 0.91 | 1.03 | 1.08 | 1.09 | 1.09 | 1.09 |

Signed head-readout transfer, group Concl: M 0:0.61, 2:0.62, 4:0.65, 6:0.69, 8:0.69, 10:0.74, 12:0.75, 14:0.72, 16:0.66, 17:0.59, 18:0.59, 19:0.63; X 0:0.53, 2:0.53, 4:0.54, 6:0.55, 8:0.58, 10:0.60, 12:0.61, 14:0.65, 16:0.66, 17:0.62, 18:0.62, 19:0.67

Signed head-readout transfer, group KV: M 0:0.96, 2:0.96, 4:0.95, 6:0.95, 8:0.95, 10:0.94, 12:0.94, 14:0.93, 16:0.94, 17:0.95, 18:0.95, 19:0.96; X 0:0.58, 2:0.58, 4:0.60, 6:0.61, 8:0.63, 10:0.70, 12:0.73, 14:0.84, 16:0.91, 17:0.92, 18:0.92, 19:0.93

## Hypothesis scoring (mechanical; each set scored separately)
```json
{
 "A_num_orig": {
  "trajectory": "flat",
  "i_sus_half": 0,
  "window": "early",
  "F4": 0.6277264325323476,
  "O1": {
   "F4_ok": true,
   "controls_available": true,
   "verdict_related": false,
   "lexical": true,
   "supported": false,
   "lexical_early_content": true
  },
  "O2": {
   "window": "early",
   "F4_low": false,
   "ctrl19": false,
   "supported": false
  },
  "O3": {
   "span": 19,
   "supported": true
  },
  "O4": {
   "supported": false
  }
 },
 "A_num_rev": {
  "trajectory": "flat",
  "i_sus_half": 0,
  "window": "early",
  "F4": 0.7700831024930747,
  "O1": {
   "F4_ok": true,
   "controls_available": false,
   "verdict_related": false,
   "lexical": true,
   "supported": false,
   "lexical_early_content": true
  },
  "O2": {
   "window": "early",
   "F4_low": false,
   "ctrl19": false,
   "supported": false
  },
  "O3": {
   "span": 19,
   "supported": true
  },
  "O4": {
   "supported": false
  }
 },
 "B_num_orig": {
  "trajectory": "flat",
  "i_sus_half": 0,
  "window": "early",
  "F4": 0.7376111817026684,
  "O1": {
   "F4_ok": true,
   "controls_available": false,
   "verdict_related": false,
   "lexical": true,
   "supported": false,
   "lexical_early_content": true
  },
  "O2": {
   "window": "early",
   "F4_low": false,
   "ctrl19": false,
   "supported": false
  },
  "O3": {
   "span": 19,
   "supported": true
  },
  "O4": {
   "supported": false
  }
 },
 "C_num_orig": {
  "trajectory": "flat",
  "i_sus_half": 0,
  "window": "early",
  "F4": 0.8195670749074337,
  "O1": {
   "F4_ok": true,
   "controls_available": false,
   "verdict_related": false,
   "lexical": true,
   "supported": false,
   "lexical_early_content": true
  },
  "O2": {
   "window": "early",
   "F4_low": false,
   "ctrl19": false,
   "supported": false
  },
  "O3": {
   "span": 19,
   "supported": true
  },
  "O4": {
   "supported": false
  }
 },
 "B_num_rev": {
  "trajectory": "flat",
  "i_sus_half": 0,
  "window": "early",
  "F4": 0.8505263157894737,
  "O1": {
   "F4_ok": true,
   "controls_available": false,
   "verdict_related": false,
   "lexical": false,
   "supported": false,
   "lexical_early_content": false
  },
  "O2": {
   "window": "early",
   "F4_low": false,
   "ctrl19": false,
   "supported": false
  },
  "O3": {
   "span": 19,
   "supported": true
  },
  "O4": {
   "supported": false
  }
 },
 "C_num_rev": {
  "trajectory": "non-monotone",
  "i_sus_half": 0,
  "window": "early",
  "F4": 0.9248833138856475,
  "O1": {
   "F4_ok": true,
   "controls_available": false,
   "verdict_related": false,
   "lexical": false,
   "supported": false,
   "lexical_early_content": false
  },
  "O2": {
   "window": "early",
   "F4_low": false,
   "ctrl19": false,
   "supported": false
  },
  "O3": {
   "span": 19,
   "supported": false
  },
  "O4": {
   "supported": false
  }
 }
}
```

## Secondary (pinning), set A, num_orig: T_free = 0.35 [0.33, 0.36] / [0.33, 0.37]; all six components pinned: 0.54 [0.51, 0.58] / [0.51, 0.57]; S_pin = 1.57 [1.49, 1.65] / [1.48, 1.66]

| component | pin-one T | add-one share q | leave-one-out T | LOO share q' | |q − q'| |
|---|---|---|---|---|---|
| L20 attn | 0.493 | 0.74 [0.67, 0.82] / [0.67, 0.83] | 0.425 | 0.60 [0.53, 0.70] / [0.53, 0.72] | 0.13 |
| L20 mlp | 0.417 | 0.35 [0.27, 0.42] / [0.27, 0.41] | 0.548 | -0.02 [-0.13, 0.06] / [-0.14, 0.06] | 0.37 |
| L21 attn | 0.343 | -0.02 [-0.04, -0.01] / [-0.04, -0.01] | 0.574 | -0.15 [-0.18, -0.13] / [-0.18, -0.13] | 0.13 |
| L21 mlp | 0.411 | 0.32 [0.29, 0.36] / [0.29, 0.37] | 0.516 | 0.15 [0.11, 0.18] / [0.10, 0.20] | 0.18 |
| L22 attn | 0.363 | 0.08 [0.07, 0.09] / [0.07, 0.09] | 0.528 | 0.08 [0.07, 0.10] / [0.07, 0.10] | 0.00 |
| L22 mlp | 0.349 | 0.01 [-0.01, 0.02] / [-0.01, 0.02] | 0.537 | 0.03 [0.02, 0.04] / [0.03, 0.04] | 0.03 |

Layer interactions (2x2 attention × MLP pins; e = T_pin − T_free): L20: e_attn +0.145, e_mlp +0.069, e_both +0.194, interaction -0.020; L21: e_attn -0.005, e_mlp +0.064, e_both +0.055, interaction -0.004; L22: e_attn +0.016, e_mlp +0.001, e_both +0.015, interaction -0.001; cumulative: layer 20 0.542, 20–21 0.527, 20–22 0.544.
Row localisation (share of the all-row increase, rows pinned at layers 20–22 with all six components): G1 0.40 [0.36, 0.44] / [0.37, 0.43]; G2 (all tail rows) 0.64 [0.58, 0.69] / [0.60, 0.68]; single tail tokens: t0: 0.00, t1: 0.51, t2: 0.16, t3: -0.00, t4: -0.02, t5: -0.00, t6: -0.00, t7: -0.01, t8: 0.00

Decomposition of the layer-19 write at index 20 (rows G1∪G2): full: T_free 0.348, T_pin 0.544, S_pin 1.57; dc: T_free 0.024, T_pin 0.032, S_pin — (free effect below the 0.05 floor: undefined); orth: T_free 0.267, T_pin 0.439, S_pin 1.64
Pinned/free ratio by dose α: 0.25: 1.34, 0.5: 1.52, 0.75: 1.73, 1.0: 1.57, 1.25: 1.34, 1.5: 1.24 → within 0.15 across α=0.5, 1, 1.5: False
Accounting (free run, rows G1∪G2): Δd_c at indices 20–23 = [0.797, 0.658, 0.753, 0.776] (signed toward source); component Δ(d_c writes) layers 20–22 [attn, mlp each]: [0.049, -0.188, 0.074, 0.021, 0.041, -0.018]; identity residual max 0.0060 (0.007 of the local shift) → ok=True.

## Secondary (pinning), set A, num_rev: T_free = 0.39 [0.37, 0.41] / [0.38, 0.40]; all six components pinned: 0.58 [0.54, 0.61] / [0.54, 0.60]; S_pin = 1.48 [1.42, 1.55] / [1.42, 1.56]

| component | pin-one T | add-one share q | leave-one-out T | LOO share q' | |q − q'| |
|---|---|---|---|---|---|
| L20 attn | 0.540 | 0.81 [0.72, 0.92] / [0.73, 0.91] | 0.469 | 0.57 [0.50, 0.67] / [0.49, 0.69] | 0.23 |
| L20 mlp | 0.455 | 0.35 [0.26, 0.42] / [0.24, 0.43] | 0.599 | -0.12 [-0.25, -0.02] / [-0.25, -0.04] | 0.47 |
| L21 attn | 0.375 | -0.07 [-0.10, -0.06] / [-0.10, -0.06] | 0.613 | -0.19 [-0.22, -0.17] / [-0.23, -0.16] | 0.12 |
| L21 mlp | 0.465 | 0.41 [0.37, 0.45] / [0.36, 0.46] | 0.542 | 0.18 [0.15, 0.22] / [0.14, 0.23] | 0.22 |
| L22 attn | 0.404 | 0.08 [0.07, 0.10] / [0.07, 0.09] | 0.560 | 0.09 [0.08, 0.11] / [0.07, 0.12] | 0.01 |
| L22 mlp | 0.399 | 0.05 [0.03, 0.07] / [0.03, 0.06] | 0.567 | 0.05 [0.04, 0.06] / [0.04, 0.06] | 0.00 |

Layer interactions (2x2 attention × MLP pins; e = T_pin − T_free): L20: e_attn +0.151, e_mlp +0.066, e_both +0.179, interaction -0.038; L21: e_attn -0.014, e_mlp +0.076, e_both +0.062, interaction -0.001; L22: e_attn +0.015, e_mlp +0.010, e_both +0.023, interaction -0.001; cumulative: layer 20 0.568, 20–21 0.557, 20–22 0.577.
Row localisation (share of the all-row increase, rows pinned at layers 20–22 with all six components): G1 0.48 [0.44, 0.52] / [0.45, 0.51]; G2 (all tail rows) 0.61 [0.55, 0.66] / [0.55, 0.66]; single tail tokens: t0: -0.00, t1: 0.43, t2: 0.18, t3: -0.00, t4: -0.01, t5: 0.00, t6: 0.00, t7: -0.00, t8: 0.00

Decomposition of the layer-19 write at index 20 (rows G1∪G2): full: T_free 0.389, T_pin 0.577, S_pin 1.48; dc: T_free 0.024, T_pin 0.029, S_pin — (free effect below the 0.05 floor: undefined); orth: T_free 0.302, T_pin 0.483, S_pin 1.60
Pinned/free ratio by dose α: 0.25: 1.39, 0.5: 1.53, 0.75: 1.67, 1.0: 1.48, 1.25: 1.30, 1.5: 1.22 → within 0.15 across α=0.5, 1, 1.5: False
Accounting (free run, rows G1∪G2): Δd_c at indices 20–23 = [0.774, 0.647, 0.732, 0.754] (signed toward source); component Δ(d_c writes) layers 20–22 [attn, mlp each]: [0.04, -0.167, 0.075, 0.01, 0.049, -0.027]; identity residual max 0.0070 (0.009 of the local shift) → ok=True.

## Secondary (pinning), set B, num_orig: T_free = 0.31 [0.29, 0.32] / [0.29, 0.32]; all six components pinned: 0.51 [0.48, 0.54] / [0.47, 0.54]; S_pin = 1.67 [1.59, 1.75] / [1.58, 1.76]

| component | pin-one T | add-one share q | leave-one-out T | LOO share q' | |q − q'| |
|---|---|---|---|---|---|
| L20 attn | 0.442 | 0.67 [0.61, 0.74] / [0.61, 0.75] | 0.394 | 0.56 [0.51, 0.62] / [0.51, 0.63] | 0.11 |
| L20 mlp | 0.379 | 0.36 [0.30, 0.41] / [0.30, 0.41] | 0.503 | 0.03 [-0.06, 0.10] / [-0.07, 0.11] | 0.33 |
| L21 attn | 0.297 | -0.04 [-0.06, -0.03] / [-0.06, -0.03] | 0.545 | -0.18 [-0.20, -0.16] / [-0.20, -0.16] | 0.13 |
| L21 mlp | 0.371 | 0.32 [0.29, 0.36] / [0.29, 0.36] | 0.478 | 0.15 [0.13, 0.18] / [0.13, 0.18] | 0.17 |
| L22 attn | 0.319 | 0.06 [0.06, 0.07] / [0.06, 0.07] | 0.494 | 0.07 [0.07, 0.08] / [0.07, 0.08] | 0.01 |
| L22 mlp | 0.312 | 0.03 [0.02, 0.04] / [0.02, 0.04] | 0.505 | 0.02 [0.01, 0.03] / [0.01, 0.03] | 0.01 |

Layer interactions (2x2 attention × MLP pins; e = T_pin − T_free): L20: e_attn +0.136, e_mlp +0.073, e_both +0.197, interaction -0.013; L21: e_attn -0.009, e_mlp +0.066, e_both +0.055, interaction -0.002; L22: e_attn +0.013, e_mlp +0.007, e_both +0.018, interaction -0.002; cumulative: layer 20 0.502, 20–21 0.495, 20–22 0.509.
Row localisation (share of the all-row increase, rows pinned at layers 20–22 with all six components): G1 0.37 [0.34, 0.41] / [0.35, 0.40]; G2 (all tail rows) 0.64 [0.59, 0.68] / [0.60, 0.68]; single tail tokens: t0: 0.01, t1: 0.36, t2: 0.28, t3: 0.00, t4: -0.02, t5: 0.00, t6: 0.00, t7: -0.00, t8: 0.00

Decomposition of the layer-19 write at index 20 (rows G1∪G2): full: T_free 0.305, T_pin 0.509, S_pin 1.67; dc: T_free 0.024, T_pin 0.036, S_pin — (free effect below the 0.05 floor: undefined); orth: T_free 0.243, T_pin 0.406, S_pin 1.67
Pinned/free ratio by dose α: 0.25: 1.59, 0.5: 1.67, 0.75: 1.74, 1.0: 1.67, 1.25: 1.49, 1.5: 1.37 → within 0.15 across α=0.5, 1, 1.5: False
Accounting (free run, rows G1∪G2): Δd_c at indices 20–23 = [0.809, 0.685, 0.759, 0.679] (signed toward source); component Δ(d_c writes) layers 20–22 [attn, mlp each]: [0.058, -0.181, 0.08, -0.007, 0.02, -0.1]; identity residual max 0.0065 (0.008 of the local shift) → ok=True.

## Secondary (pinning), set B, num_rev: T_free = 0.34 [0.33, 0.36] / [0.32, 0.36]; all six components pinned: 0.56 [0.53, 0.60] / [0.53, 0.60]; S_pin = 1.64 [1.57, 1.72] / [1.56, 1.73]

| component | pin-one T | add-one share q | leave-one-out T | LOO share q' | |q − q'| |
|---|---|---|---|---|---|
| L20 attn | 0.489 | 0.66 [0.60, 0.73] / [0.60, 0.74] | 0.453 | 0.50 [0.45, 0.55] / [0.45, 0.56] | 0.16 |
| L20 mlp | 0.422 | 0.36 [0.30, 0.40] / [0.30, 0.41] | 0.563 | 0.00 [-0.09, 0.08] / [-0.09, 0.08] | 0.36 |
| L21 attn | 0.328 | -0.07 [-0.08, -0.05] / [-0.09, -0.05] | 0.605 | -0.19 [-0.21, -0.17] / [-0.22, -0.17] | 0.12 |
| L21 mlp | 0.430 | 0.40 [0.36, 0.44] / [0.36, 0.44] | 0.520 | 0.19 [0.17, 0.22] / [0.17, 0.22] | 0.20 |
| L22 attn | 0.358 | 0.07 [0.06, 0.08] / [0.06, 0.08] | 0.547 | 0.07 [0.06, 0.08] / [0.06, 0.08] | 0.00 |
| L22 mlp | 0.357 | 0.06 [0.05, 0.08] / [0.05, 0.08] | 0.556 | 0.03 [0.02, 0.04] / [0.02, 0.04] | 0.03 |

Layer interactions (2x2 attention × MLP pins; e = T_pin − T_free): L20: e_attn +0.146, e_mlp +0.079, e_both +0.202, interaction -0.023; L21: e_attn -0.015, e_mlp +0.087, e_both +0.071, interaction -0.001; L22: e_attn +0.015, e_mlp +0.014, e_both +0.026, interaction -0.003; cumulative: layer 20 0.545, 20–21 0.547, 20–22 0.563.
Row localisation (share of the all-row increase, rows pinned at layers 20–22 with all six components): G1 0.43 [0.40, 0.47] / [0.40, 0.46]; G2 (all tail rows) 0.60 [0.56, 0.64] / [0.56, 0.64]; single tail tokens: t0: 0.00, t1: 0.31, t2: 0.28, t3: 0.00, t4: -0.01, t5: 0.00, t6: 0.01, t7: -0.00, t8: 0.00

Decomposition of the layer-19 write at index 20 (rows G1∪G2): full: T_free 0.343, T_pin 0.563, S_pin 1.64; dc: T_free 0.024, T_pin 0.033, S_pin — (free effect below the 0.05 floor: undefined); orth: T_free 0.278, T_pin 0.472, S_pin 1.70
Pinned/free ratio by dose α: 0.25: 1.62, 0.5: 1.68, 0.75: 1.75, 1.0: 1.64, 1.25: 1.44, 1.5: 1.32 → within 0.15 across α=0.5, 1, 1.5: False
Accounting (free run, rows G1∪G2): Δd_c at indices 20–23 = [0.816, 0.69, 0.754, 0.65] (signed toward source); component Δ(d_c writes) layers 20–22 [attn, mlp each]: [0.049, -0.175, 0.085, -0.022, 0.018, -0.122]; identity residual max 0.0059 (0.007 of the local shift) → ok=True.