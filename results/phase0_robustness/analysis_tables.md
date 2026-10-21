## 1. Logic accuracy per seed (first-character accuracy; invalid rate in brackets for ablate)

| seed | final KL (median last 5) | dataset hash | intact/items | intact/families | intact/nonwords | retain/items | retain/families | retain/nonwords | ablate/items | ablate/families | ablate/nonwords |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0088 | 96b2e84595f89e08 | 0.945 | 0.977 | 0.953 | 0.949 | 0.977 | 0.949 | 0.500 | 0.500 | 0.500 |
| 1 | 0.0003 | 96b2e84595f89e08 | 0.945 | 0.977 | 0.953 | 0.941 | 0.980 | 0.949 | 0.852 | 0.883 | 0.770 |
| 2 | 0.0027 | 96b2e84595f89e08 | 0.945 | 0.977 | 0.953 | 0.941 | 0.977 | 0.949 | 0.500 | 0.500 | 0.500 |
| **mean ± std** | 0.004 ± 0.004 | | 0.945 ± 0.000 | 0.977 ± 0.000 | 0.953 ± 0.000 | 0.944 ± 0.005 | 0.978 ± 0.002 | 0.949 ± 0.000 | 0.617 ± 0.203 | 0.628 ± 0.221 | 0.590 ± 0.156 |

Per-form accuracy (mean over seeds) for retain / ablate on Items:

| cond | AC | AC_negQ | DA | DA_negQ | MP | MP_negQ | MT | MT_negQ |
|---|---|---|---|---|---|---|---|---|
| intact | 0.969 ± 0.000 | 0.938 ± 0.000 | 1.000 ± 0.000 | 0.938 ± 0.000 | 0.969 ± 0.000 | 1.000 ± 0.000 | 0.781 ± 0.000 | 0.969 ± 0.000 |
| retain | 0.969 ± 0.000 | 0.938 ± 0.000 | 0.979 ± 0.018 | 0.958 ± 0.036 | 0.969 ± 0.000 | 1.000 ± 0.000 | 0.771 ± 0.018 | 0.969 ± 0.000 |
| ablate | 0.292 ± 0.505 | 0.260 ± 0.451 | 0.250 ± 0.433 | 0.146 ± 0.253 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.990 ± 0.018 | 1.000 ± 0.000 |

## 3. Prediction distributions
Gold is balanced 128/128 in every split. Confusion rows = gold (1 valid / 2 invalid), cols = first generated char.

### Ablation: predicted-class frequencies and entropies

| seed | split | acc | predicted-char counts | constant? | empirical class entropy (bits) | mean per-example entropy over {1,2} (bits) | mean P(1) | std P(1) | confusion (gold→pred) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | items | 0.500 | {'1': 256} | True | -0.000 | 0.000 | 1.000 | 0.000 | {'1': {'1': 128}, '2': {'1': 128}} |
| 0 | families | 0.500 | {'1': 256} | True | -0.000 | 0.002 | 1.000 | 0.002 | {'1': {'1': 128}, '2': {'1': 128}} |
| 0 | nonwords | 0.500 | {'1': 256} | True | -0.000 | 0.000 | 1.000 | 0.000 | {'1': {'1': 128}, '2': {'1': 128}} |
| 1 | items | 0.852 | {'2': 92, '1': 164} | False | 0.942 | 0.158 | 0.630 | 0.447 | {'1': {'1': 127, '2': 1}, '2': {'1': 37, '2': 91}} |
| 1 | families | 0.883 | {'2': 98, '1': 158} | False | 0.960 | 0.174 | 0.616 | 0.446 | {'1': {'1': 128, '2': 0}, '2': {'1': 30, '2': 98}} |
| 1 | nonwords | 0.770 | {'2': 69, '1': 187} | False | 0.841 | 0.198 | 0.725 | 0.397 | {'1': {'1': 128, '2': 0}, '2': {'1': 59, '2': 69}} |
| 2 | items | 0.500 | {'1': 256} | True | -0.000 | 0.000 | 1.000 | 0.000 | {'1': {'1': 128}, '2': {'1': 128}} |
| 2 | families | 0.500 | {'1': 256} | True | -0.000 | 0.000 | 1.000 | 0.000 | {'1': {'1': 128}, '2': {'1': 128}} |
| 2 | nonwords | 0.500 | {'1': 256} | True | -0.000 | 0.000 | 1.000 | 0.000 | {'1': {'1': 128}, '2': {'1': 128}} |

For contrast, intact / retain class entropy and per-example entropy (seed 0):

- intact/items: acc 0.945, preds {'2': 132, '1': 124}, class entropy 0.999, per-example entropy 0.028
- intact/families: acc 0.977, preds {'2': 132, '1': 124}, class entropy 0.999, per-example entropy 0.008
- intact/nonwords: acc 0.953, preds {'2': 122, '1': 134}, class entropy 0.998, per-example entropy 0.045
- retain/items: acc 0.949, preds {'2': 135, '1': 121}, class entropy 0.998, per-example entropy 0.053
- retain/families: acc 0.977, preds {'2': 132, '1': 124}, class entropy 0.999, per-example entropy 0.025
- retain/nonwords: acc 0.949, preds {'2': 119, '1': 137}, class entropy 0.996, per-example entropy 0.064

Ablate: accuracy by logical form (Items, per seed) — a constant predictor gives 1.0 on one label's forms and 0.0 on the other's:

| seed | AC | AC_negQ | DA | DA_negQ | MP | MP_negQ | MT | MT_negQ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | 0.88 | 0.78 | 0.75 | 0.44 | 1.00 | 1.00 | 0.97 | 1.00 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## 5/6. Controls (accuracy; lm_kl in nats). Logic-adjacent PrOntoQA is kept separate from clearly non-logical controls.

### Clearly non-logical controls

| task | intact (per seed) | retain (per seed) | ablate (per seed) | ablate mean ± std | Δ ablate−intact mean |
|---|---|---|---|---|---|
| word_recall_8 | 1.000, 1.000, 1.000 | 0.516, 0.516, 0.516 | 1.000, 1.000, 1.000 | 1.000 ± 0.000 | +0.000 |
| word_recall_48 | 1.000, 1.000, 1.000 | 0.539, 0.539, 0.539 | 1.000, 1.000, 1.000 | 1.000 ± 0.000 | +0.000 |
| language_localizer | 1.000, 1.000, 1.000 | 1.000, 1.000, 0.969 | 0.562, 0.562, 0.562 | 0.562 ± 0.000 | -0.438 |
| mmlu_binary | 0.812, 0.812, 0.812 | 0.469, 0.469, 0.469 | 0.812, 0.797, 0.789 | 0.799 ± 0.012 | -0.013 |
| stroop | 0.898, 0.898, 0.898 | 0.508, 0.508, 0.508 | 1.000, 0.891, 0.969 | 0.953 ± 0.056 | +0.055 |
| msit | 1.000, 1.000, 1.000 | 0.000, 0.000, 0.000 | 1.000, 1.000, 0.992 | 0.997 ± 0.005 | -0.003 |
| multiplication_1digit | 1.000, 1.000, 1.000 | 0.000, 0.000, 0.000 | 1.000, 1.000, 1.000 | 1.000 ± 0.000 | +0.000 |
| lm_kl | 0.000, 0.000, 0.000 | 7.991, 16.930, 7.395 | 0.005, 0.006, 0.001 | 0.004 ± 0.003 | +0.004 |

### Logic-adjacent control

| task | intact (per seed) | retain (per seed) | ablate (per seed) | ablate mean ± std | Δ ablate−intact mean |
|---|---|---|---|---|---|
| prontoqa | 0.758, 0.758, 0.758 | 0.492, 0.523, 0.469 | 0.461, 0.648, 0.453 | 0.521 ± 0.111 | -0.237 |

## 5. Language localizer under ablation (gold 1 = English sentence, 2 = pseudoword string)

| seed | acc | acc on gold=1 | acc on gold=2 | predicted-char counts | wrong rate on pseudoword (gold 2) | n wrong |
|---|---|---|---|---|---|---|
| 0 | 0.562 | 1.000 (n=72) | 0.000 (n=56) | {'1': 128} | 1.000 | 56 |
| 1 | 0.562 | 1.000 (n=72) | 0.000 (n=56) | {'1': 128} | 1.000 | 56 |
| 2 | 0.562 | 1.000 (n=72) | 0.000 (n=56) | {'1': 128} | 1.000 | 56 |

Intact errors: 0/128. Items wrong under ablation in ≥1 seed: 56; in all 3 seeds: 56.
- seeds 0/1: prediction agreement 1.000; correctness agreement 1.000; both-wrong 56
- seeds 0/2: prediction agreement 1.000; correctness agreement 1.000; both-wrong 56
- seeds 1/2: prediction agreement 1.000; correctness agreement 1.000; both-wrong 56

Examples wrong in all seeds (first 8):

- gold 2: "Ths decududs to beild aen woodenen fenen aroin the girds."
- gold 2: "Shes opined thed winded tas leen in soen frashs aars."
- gold 2: "She opunod the wundiws ti leted in se fross aied."
- gold 2: "As sma do chess thed bolled acrusen ths gras fuolded."
- gold 2: "Aed smaed du chesed ths bied acros ths gres fielden."
- gold 2: "Aed small doged chesiden thaen balen acreed thed greined fies."
- gold 2: "Sh oponi thoen windus tos luen ins soed frishen aaren."
- gold 2: "Th committen wallen annuincs it dacasied noxts woied."

## Answer-channel diagnostic: ablated accuracy by gold label and predicted-label counts (all label-based tasks)

| task | seed | acc | acc gold=first label | acc gold=second label | predicted counts | mean P(first label) |
|---|---|---|---|---|---|---|
| language_localizer | 0 | 0.562 | 1.000 | 0.000 | {'1': 128} | 1.000 |
| language_localizer | 1 | 0.562 | 1.000 | 0.000 | {'1': 128} | 0.998 |
| language_localizer | 2 | 0.562 | 1.000 | 0.000 | {'1': 128} | 1.000 |
| prontoqa | 0 | 0.461 | 1.000 | 0.000 | {'1': 128} | 0.982 |
| prontoqa | 1 | 0.648 | 0.644 | 0.652 | {'2': 66, '1': 62} | 0.483 |
| prontoqa | 2 | 0.453 | 0.983 | 0.000 | {'1': 127, '2': 1} | 0.971 |
| mmlu_binary | 0 | 0.812 | 0.885 | 0.746 | {'1': 71, '2': 57} | 0.537 |
| mmlu_binary | 1 | 0.797 | 0.754 | 0.836 | {'1': 57, '2': 71} | 0.450 |
| mmlu_binary | 2 | 0.789 | 0.852 | 0.731 | {'1': 70, '2': 58} | 0.531 |
| stroop | 0 | 1.000 | 1.000 | 1.000 | {'1': 65, '2': 63} | 0.461 |
| stroop | 1 | 0.891 | 0.785 | 1.000 | {'1': 51, '2': 77} | 0.400 |
| stroop | 2 | 0.969 | 0.938 | 1.000 | {'1': 61, '2': 67} | 0.450 |
| word_recall_8 | 0 | 1.000 | 1.000 | 1.000 | {'1': 66, '2': 62} | 0.516 |
| word_recall_8 | 1 | 1.000 | 1.000 | 1.000 | {'1': 66, '2': 62} | 0.515 |
| word_recall_8 | 2 | 1.000 | 1.000 | 1.000 | {'1': 66, '2': 62} | 0.516 |
| word_recall_48 | 0 | 1.000 | 1.000 | 1.000 | {'1': 69, '2': 59} | 0.539 |
| word_recall_48 | 1 | 1.000 | 1.000 | 1.000 | {'1': 69, '2': 59} | 0.539 |
| word_recall_48 | 2 | 1.000 | 1.000 | 1.000 | {'1': 69, '2': 59} | 0.539 |
| msit | 0 | 1.000 | 1.000 | 1.000 | {'A': 67, 'B': 61} | nan |
| msit | 1 | 1.000 | 1.000 | 1.000 | {'A': 67, 'B': 61} | nan |
| msit | 2 | 0.992 | 1.000 | 0.984 | {'A': 68, 'B': 60} | nan |

## 4. Random rank-2 ablations at the same residual location (unoptimised; seed-0 mu)

gaussian = uniform random directions; topvar = random directions inside the top-2048 variance subspace. Δ = ablated − intact accuracy.

| subspace | n | items acc mean [min,max] | families acc mean [min,max] | nonwords acc mean [min,max] | max |Δ| over splits |
|---|---|---|---|---|---|
| gaussian | 10 | 0.944 [0.941,0.945] | 0.977 [0.977,0.980] | 0.952 [0.945,0.957] | 0.008 |
| topvar | 10 | 0.944 [0.941,0.945] | 0.979 [0.977,0.980] | 0.954 [0.949,0.957] | 0.004 |
| learned MVS (seed 0) | 1 | 0.500 | 0.500 | 0.500 | 0.477 |
| learned MVS (seed 1) | 1 | 0.852 | 0.883 | 0.770 | 0.184 |
| learned MVS (seed 2) | 1 | 0.500 | 0.500 | 0.500 | 0.477 |

Controls under random ablation (mean over subspaces [min,max]) vs learned MVS (per seed) vs intact:

| task | intact | gaussian | topvar | learned s0 | learned s1 | learned s2 |
|---|---|---|---|---|---|---|
| word_recall_8 | 1.000 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | 1.000 | 1.000 |
| word_recall_48 | 1.000 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | 1.000 | 1.000 |
| language_localizer | 1.000 | 1.000 [1.000,1.000] | 0.999 [0.992,1.000] | 0.562 | 0.562 | 0.562 |
| mmlu_binary | 0.812 | 0.812 [0.805,0.828] | 0.807 [0.797,0.828] | 0.812 | 0.797 | 0.789 |
| stroop | 0.898 | 0.898 [0.898,0.898] | 0.898 [0.898,0.898] | 1.000 | 0.891 | 0.969 |
| msit | 1.000 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | 1.000 | 0.992 |
| multiplication_1digit | 1.000 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | 1.000 | 1.000 |
| prontoqa | 0.758 | 0.755 [0.742,0.781] | 0.758 [0.742,0.781] | 0.461 | 0.648 | 0.453 |
| lm_kl | 0.000 | 0.003 [0.001,0.004] | 0.003 [0.002,0.007] | 0.005 | 0.006 | 0.001 |

Largest principal-angle cosine between each random subspace and the seed-0 learned MVS: max 0.055, mean 0.026.

## Cross-seed subspace overlap (cosines of principal angles between the learned rank-2 bases; 1 = identical, ~sqrt(2/4096)=0.022 typical random)

- seeds 0/1: [0.7727, 0.1939]
- seeds 0/2: [0.8303, 0.0962]
- seeds 1/2: [0.7666, 0.0615]
- reference mu identical across seeds: True