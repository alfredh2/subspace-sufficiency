
## Stage M, set A num_orig (434 trials)

Norm statistics (mean over trials of the row-mean norm): norm_full: 5.931, norm_dc: 0.829, norm_orth: 5.864, norm_geff: 0.094, cos_full_dc: -0.001, scale_dc_eqorth: 9.358, scale_dc_eqfull: 9.428; max rescale factor dc→eqorth 41.6, dc→eqfull 41.6

| condition | T_margin [pair] / [content] | T_coordinate | pinned (layers 20–22 attn+MLP) T | S_pin |
|---|---|---|---|---|
| full | 0.348 [0.33, 0.37] / [0.33, 0.36] | 0.375 | 0.544 | 1.57 |
| dc | 0.024 [0.02, 0.03] / [0.02, 0.03] | 0.034 | — | — |
| orth | 0.267 [0.25, 0.28] / [0.25, 0.28] | 0.303 | 0.439 | 1.64 |
| geff | 0.004 [0.00, 0.01] / [0.00, 0.01] | 0.004 | — | — |
| rest | 0.265 [0.25, 0.28] / [0.25, 0.28] | 0.302 | 0.439 | 1.66 |
| dc_plus_geff | 0.029 [0.03, 0.03] / [0.03, 0.03] | 0.038 | — | — |
| pc1 | 0.274 [0.25, 0.30] / [0.25, 0.29] | 0.315 | — | — |
| pc3 | 0.303 [0.28, 0.33] / [0.28, 0.33] | 0.328 | — | — |
| dc_eqorth | 0.278 [0.26, 0.30] / [0.27, 0.29] | 0.313 | — | — |
| dc_eqfull | 0.284 [0.26, 0.31] / [0.27, 0.29] | 0.318 | — | — |
| geff_eqfull | -0.012 [-0.04, 0.02] / [-0.05, 0.03] | -0.064 | — | — |
| dc_x2 | 0.054 [0.05, 0.06] / [0.05, 0.06] | 0.077 | — | — |
| dc_x4 | 0.143 [0.13, 0.15] / [0.13, 0.15] | 0.185 | — | — |
| dc_x8 | 0.369 [0.34, 0.39] / [0.35, 0.39] | 0.388 | — | — |
| random ⟂ d_c, norm of the orthogonal part (mean of 20) | 0.007 [0.01, 0.01] / [0.01, 0.01] | | | |
| random, norm of Δ (mean of 5) | 0.007 [0.01, 0.01] / [0.01, 0.01] | | | |
| random, norm of the d_c part (mean of 5) | 0.001 [0.00, 0.00] / [0.00, 0.00] | | | |

Rules (A_num_orig): M-A (magnitude artefact): ratio T(dc equal-norm)/T(orth) = 1.040 1.040 [0.98, 1.11] / [0.97, 1.10] → True; M-B (orthogonal part special): T(orth)/T(full) = 0.769, random-orth/T(orth) = 0.027 (upper 0.029), dc-eqnorm/orth = 1.040 → False; M-C (position-specific direction): T(g_eff)/T(full) = 0.012 (lower 0.010), |cos(g_eff, d_c)| = 0.020 → False.

Head ranking (32 layer-19 heads): {"|write on d_c|": {"spearman_with_causal": 0.28833287629116894, "top3": [10, 31, 17], "overlap_with_causal_top3": 3}, "|write on g_eff|": {"spearman_with_causal": 0.312711943390168, "top3": [10, 31, 30], "overlap_with_causal_top3": 2}, "write norm": {"spearman_with_causal": 0.5268078108460392, "top3": [10, 31, 17], "overlap_with_causal_top3": 3}, "causal_top3": [10, 31, 17], "validation_R1_vs_test_causal_spearman": 0.9624060797693476}

## Stage M, set A num_rev (434 trials)

Norm statistics (mean over trials of the row-mean norm): norm_full: 5.783, norm_dc: 0.808, norm_orth: 5.718, norm_geff: 0.089, cos_full_dc: -0.001, scale_dc_eqorth: 9.508, scale_dc_eqfull: 9.579; max rescale factor dc→eqorth 46.0, dc→eqfull 46.1

| condition | T_margin [pair] / [content] | T_coordinate | pinned (layers 20–22 attn+MLP) T | S_pin |
|---|---|---|---|---|
| full | 0.389 [0.37, 0.41] / [0.38, 0.40] | 0.382 | 0.577 | 1.48 |
| dc | 0.024 [0.02, 0.03] / [0.02, 0.03] | 0.036 | — | — |
| orth | 0.302 [0.29, 0.32] / [0.29, 0.32] | 0.308 | 0.483 | 1.60 |
| geff | 0.003 [0.00, 0.00] / [0.00, 0.00] | 0.003 | — | — |
| rest | 0.301 [0.29, 0.32] / [0.29, 0.31] | 0.307 | 0.482 | 1.60 |
| dc_plus_geff | 0.027 [0.02, 0.03] / [0.03, 0.03] | 0.039 | — | — |
| pc1 | 0.325 [0.30, 0.35] / [0.30, 0.34] | 0.325 | — | — |
| pc3 | 0.356 [0.33, 0.38] / [0.33, 0.38] | 0.339 | — | — |
| dc_eqorth | 0.272 [0.25, 0.29] / [0.26, 0.28] | 0.326 | — | — |
| dc_eqfull | 0.278 [0.26, 0.30] / [0.27, 0.29] | 0.331 | — | — |
| geff_eqfull | -0.054 [-0.09, -0.02] / [-0.08, -0.02] | -0.113 | — | — |
| dc_x2 | 0.054 [0.05, 0.06] / [0.05, 0.06] | 0.081 | — | — |
| dc_x4 | 0.143 [0.13, 0.16] / [0.14, 0.15] | 0.194 | — | — |
| dc_x8 | 0.363 [0.34, 0.39] / [0.35, 0.38] | 0.407 | — | — |
| random ⟂ d_c, norm of the orthogonal part (mean of 20) | 0.006 [0.00, 0.01] / [0.00, 0.01] | | | |
| random, norm of Δ (mean of 5) | 0.007 [0.01, 0.01] / [0.01, 0.01] | | | |
| random, norm of the d_c part (mean of 5) | 0.000 [-0.00, 0.00] / [-0.00, 0.00] | | | |

Rules (A_num_rev): M-A (magnitude artefact): ratio T(dc equal-norm)/T(orth) = 0.900 0.900 [0.85, 0.96] / [0.85, 0.95] → True; M-B (orthogonal part special): T(orth)/T(full) = 0.778, random-orth/T(orth) = 0.019 (upper 0.022), dc-eqnorm/orth = 0.900 → False; M-C (position-specific direction): T(g_eff)/T(full) = 0.008 (lower 0.006), |cos(g_eff, d_c)| = 0.020 → False.

Head ranking (32 layer-19 heads): {"|write on d_c|": {"spearman_with_causal": 0.30461876832844575, "top3": [10, 31, 17], "overlap_with_causal_top3": 3}, "|write on g_eff|": {"spearman_with_causal": 0.281524926686217, "top3": [10, 30, 31], "overlap_with_causal_top3": 2}, "write norm": {"spearman_with_causal": 0.5183284457478006, "top3": [10, 31, 17], "overlap_with_causal_top3": 3}, "causal_top3": [10, 31, 17], "validation_R1_vs_test_causal_spearman": 0.9533328222765189}

## Stage M, set B num_orig (400 trials)

Norm statistics (mean over trials of the row-mean norm): norm_full: 5.932, norm_dc: 0.830, norm_orth: 5.867, norm_geff: 0.107, cos_full_dc: 0.000, scale_dc_eqorth: 8.675, scale_dc_eqfull: 8.747; max rescale factor dc→eqorth 38.6, dc→eqfull 38.6

| condition | T_margin [pair] / [content] | T_coordinate | pinned (layers 20–22 attn+MLP) T | S_pin |
|---|---|---|---|---|
| full | 0.305 [0.29, 0.32] / [0.29, 0.32] | 0.353 | 0.509 | 1.67 |
| dc | 0.024 [0.02, 0.03] / [0.02, 0.03] | 0.031 | — | — |
| orth | 0.243 [0.23, 0.26] / [0.23, 0.26] | 0.290 | 0.406 | 1.67 |
| geff | 0.005 [0.00, 0.01] / [0.00, 0.01] | 0.005 | — | — |
| rest | 0.239 [0.23, 0.25] / [0.22, 0.25] | 0.287 | 0.404 | 1.69 |
| dc_plus_geff | 0.029 [0.03, 0.03] / [0.03, 0.03] | 0.037 | — | — |
| pc1 | 0.252 [0.23, 0.27] / [0.23, 0.27] | 0.295 | — | — |
| pc3 | 0.285 [0.27, 0.30] / [0.27, 0.30] | 0.328 | — | — |
| dc_eqorth | 0.247 [0.23, 0.26] / [0.23, 0.26] | 0.284 | — | — |
| dc_eqfull | 0.250 [0.23, 0.27] / [0.23, 0.27] | 0.288 | — | — |
| geff_eqfull | -0.003 [-0.04, 0.04] / [-0.04, 0.03] | -0.101 | — | — |
| dc_x2 | 0.054 [0.05, 0.06] / [0.05, 0.06] | 0.069 | — | — |
| dc_x4 | 0.132 [0.12, 0.14] / [0.12, 0.14] | 0.161 | — | — |
| dc_x8 | 0.306 [0.29, 0.32] / [0.28, 0.33] | 0.345 | — | — |
| random ⟂ d_c, norm of the orthogonal part (mean of 20) | 0.006 [0.00, 0.01] / [0.00, 0.01] | | | |
| random, norm of Δ (mean of 5) | 0.006 [0.00, 0.01] / [0.00, 0.01] | | | |
| random, norm of the d_c part (mean of 5) | 0.000 [-0.00, 0.00] / [-0.00, 0.00] | | | |

Rules (B_num_orig): M-A (magnitude artefact): ratio T(dc equal-norm)/T(orth) = 1.013 1.013 [0.96, 1.06] / [0.96, 1.07] → True; M-B (orthogonal part special): T(orth)/T(full) = 0.797, random-orth/T(orth) = 0.023 (upper 0.026), dc-eqnorm/orth = 1.013 → False; M-C (position-specific direction): T(g_eff)/T(full) = 0.016 (lower 0.013), |cos(g_eff, d_c)| = 0.020 → False.

## Stage M, set B num_rev (400 trials)

Norm statistics (mean over trials of the row-mean norm): norm_full: 5.909, norm_dc: 0.838, norm_orth: 5.842, norm_geff: 0.099, cos_full_dc: 0.000, scale_dc_eqorth: 8.618, scale_dc_eqfull: 8.691; max rescale factor dc→eqorth 39.2, dc→eqfull 39.2

| condition | T_margin [pair] / [content] | T_coordinate | pinned (layers 20–22 attn+MLP) T | S_pin |
|---|---|---|---|---|
| full | 0.343 [0.33, 0.36] / [0.32, 0.36] | 0.353 | 0.563 | 1.64 |
| dc | 0.024 [0.02, 0.03] / [0.02, 0.03] | 0.032 | — | — |
| orth | 0.278 [0.27, 0.29] / [0.26, 0.29] | 0.286 | 0.472 | 1.70 |
| geff | 0.004 [0.00, 0.00] / [0.00, 0.01] | 0.005 | — | — |
| rest | 0.274 [0.26, 0.29] / [0.26, 0.29] | 0.284 | 0.469 | 1.71 |
| dc_plus_geff | 0.028 [0.03, 0.03] / [0.03, 0.03] | 0.037 | — | — |
| pc1 | 0.284 [0.27, 0.30] / [0.26, 0.30] | 0.300 | — | — |
| pc3 | 0.324 [0.30, 0.34] / [0.30, 0.34] | 0.335 | — | — |
| dc_eqorth | 0.233 [0.22, 0.25] / [0.21, 0.25] | 0.292 | — | — |
| dc_eqfull | 0.237 [0.22, 0.25] / [0.22, 0.25] | 0.297 | — | — |
| geff_eqfull | -0.068 [-0.11, -0.03] / [-0.11, -0.02] | -0.166 | — | — |
| dc_x2 | 0.053 [0.05, 0.06] / [0.05, 0.06] | 0.072 | — | — |
| dc_x4 | 0.130 [0.12, 0.14] / [0.12, 0.14] | 0.169 | — | — |
| dc_x8 | 0.290 [0.27, 0.31] / [0.27, 0.31] | 0.362 | — | — |
| random ⟂ d_c, norm of the orthogonal part (mean of 20) | 0.005 [0.00, 0.01] / [0.00, 0.01] | | | |
| random, norm of Δ (mean of 5) | 0.005 [0.00, 0.01] / [0.00, 0.01] | | | |
| random, norm of the d_c part (mean of 5) | 0.000 [-0.00, 0.00] / [-0.00, 0.00] | | | |

Rules (B_num_rev): M-A (magnitude artefact): ratio T(dc equal-norm)/T(orth) = 0.838 0.838 [0.79, 0.88] / [0.79, 0.88] → True; M-B (orthogonal part special): T(orth)/T(full) = 0.811, random-orth/T(orth) = 0.020 (upper 0.023), dc-eqnorm/orth = 0.838 → False; M-C (position-specific direction): T(g_eff)/T(full) = 0.012 (lower 0.008), |cos(g_eff, d_c)| = 0.020 → False.