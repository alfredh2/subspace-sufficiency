
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
 }
}
```