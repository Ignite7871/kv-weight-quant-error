# Phase 5 report — locked analysis and independent recomputation (v3, lock 7325f563ce2978d8)

Date: 2026-10-31. Branch `v3-novelty`. Not a manuscript text; no claims are proposed here.

**Sources.**
- Every number below comes from `v3_runs/v3_report.json`. That file was written by `analyze_v3.py analyze`, run exactly
  as locked (sha256 f9ce364a8392bd42…).
- Exceptions are labelled "recomputation only". They come from `v3/phase5/recompute_v3.json`.
- Notation:
  - θ = mean over problems of log m^a − log m^b (positive means a is more disruptive).
  - * marks p at the permutation floor 1/10,001.
  - Holm p is within the family stated.

## 0. Independent recomputation (reported first)

- **Script.** `v3/phase5/recompute_v3_independent.py`, committed before it was run (3a1789b).
  - No project imports.
  - Multinomial bootstrap with seed 4242031.
  - 100,000 sign flips.
- **Tolerances.** Declared before the run (`RECOMPUTATION_TOLERANCES.md`, 79bd1e0; V3-D08).
- **Compared.** 1,464 quantities:
  - 791 deterministic;
  - 267 CI;
  - 154 p;
  - 252 decisions.
- **Integrity re-check of the raw records.** No problems: all flags true, step-1 JS = 0, finite, exactly once, same
  problems and S in every arm.
- **Disagreements beyond tolerance: 1.**
  - **What:** the llama-base H-C fraction f, lower 95 % CI endpoint.
    - Recomputation −106.96 against locked −98.89; the tolerance is 4.95 (5 % of the width).
    - The upper endpoint agrees (+0.084 vs +0.026).
  - **Diagnosis** (`diag_hc_fraction_mc.py`, 30 seeds per method):
    - the endpoint's Monte-Carlo SD is 5.0 with the locked method and 6.8 with the recomputation;
    - the medians are −97.4 and −97.9.
    - So this is Monte-Carlo noise of a ratio whose denominator β₀ = +0.029 is close to 0. The tolerance is about 1 SD
      for this quantity.
  - **Effect on decisions: none.** The β₀ CI includes 0, so H-C is "not applicable" for llama-base under every seed of
    both methods, and f enters no decision.
- **Agreement on the rest.**
  - Every other quantity, and every decision (all verdict words, bands, preconditions, Holm significance, readings, the
    guard), agrees.
  - The per-problem θ(NAT, C1) and D vectors match to ≤ 1e-9.
- **Borderline items** (within tolerance, but close to a threshold):
  - H-F interaction: Holm p 0.0428 (recomputation 0.0445).
  - qwen-instruct NAT − CV, descriptive: upper bound −0.0002 (recomputation −0.0001).
  - llama-base H-C: β₀ CI [−0.0072, +0.0656] (recomputation [−0.0086, +0.0667]); shrinkage p 0.052.

## a. H-C precondition: in-session Qwen-base vs Llama-instruct gap

**Pre-declared gap** g₀ = β₀^QB − β₀^LI. β₀ is the step-level mean of log JS^NAT − log JS^C1 over steps 2…S.

| β₀ QB | β₀ LI | g₀ | 95 % CI |
|---|---|---|---|
| −0.1425 | +0.0531 | **−0.196** | **[−0.259, −0.130]** |

**The precondition is met:** the CI excludes 0, in the same direction as E1/E1L (Qwen-base lower). **The E1/E1L gap
replicated in-session.**

**Problem-mean θ(NAT, C1), reported with it:**

| Model | Estimate | 95 % CI |
|---|---|---|
| Qwen-base | −0.241 | [−0.307, −0.176] |
| Llama-instruct | +0.079 | [+0.022, +0.139] |

**On the problem-mean scale** (recomputation only, descriptive): θ_LI − θ_QB = +0.320 [+0.236, +0.401], n = 200.
E1/E1L gave +0.311 [+0.214, +0.407] across sessions.

## b. H-A (transfer): primary family P, Holm over four

Estimand: D_m, NAT only. Excluded problems (S < 8): 0 in every model; n = 200.

| Model | D_m | 95 % CI | p | Holm p | Pre-declared verdict |
|---|---|---|---|---|---|
| qwen-base | −0.0630 | [−0.0908, −0.0345] | 0.0002 | 0.0004 | **Falsified** (CI entirely < 0) |
| llama-instruct | +0.0420 | [+0.0100, +0.0756] | 0.0125 | 0.0125 | **Supported** |
| qwen-instruct | −0.1603 | [−0.1860, −0.1350] | 0.0001* | 0.0004 | **Falsified** (CI entirely < 0) |
| llama-base | +0.1809 | [+0.1529, +0.2083] | 0.0001* | 0.0004 | **Supported** |
| Pooled, mean over models (secondary) | −0.0001 | [−0.0175, +0.0172] | 0.989 | — | Inconclusive |

## c. Hypotheses in F_m and family X

### H-B: θ(QA, QO), U₈

| Model | Estimate | 95 % CI | p | Holm p | Verdict |
|---|---|---|---|---|---|
| qwen-base | +3.467 | [+3.383, +3.551] | 0.0001* | 0.0007 | difference (predicted direction; not falsified) |
| llama-instruct | +2.557 | [+2.498, +2.620] | 0.0001* | 0.0007 | difference (predicted direction; not falsified) |
| qwen-instruct | +3.162 | [+3.063, +3.264] | 0.0001* | 0.0007 | difference (predicted direction; not falsified) |
| llama-base | +2.713 | [+2.654, +2.772] | 0.0001* | 0.0006 | difference (predicted direction; not falsified) |

Diagnostics (no test):

| Model | E₈ median [range] | r₈₀ median [range] | share of slots with r₈₀ ≤ 8 |
|---|---|---|---|
| qwen-base | 0.892 [0.640, 0.973] | 2.5 [1, 17] | 0.83 |
| llama-instruct | 0.869 [0.737, 0.934] | 4 [1, 14] | 0.91 |
| qwen-instruct | 0.886 [0.646, 0.974] | 3 [1, 17] | 0.83 |
| llama-base | 0.882 [0.744, 0.945] | 3 [1, 14] | 0.94 |

### H-C_m: per model; the bands apply to the shrinkage fraction f_m

The shrinkage S_m = |β₀| − |β₁| is the quantity tested in F_m, with a bootstrap p.

| Model | β₀ [95 % CI] | β₁ | S_m [95 % CI] | f_m [95 % CI] | p | Holm p | Band |
|---|---|---|---|---|---|---|---|
| qwen-base | −0.1425 [−0.1976, −0.0865] | +0.5429 | −0.400 [−0.688, −0.123] | −2.81 [−6.91, −0.70] | 0.0040 | 0.0040 | **No** (falsified) |
| llama-instruct | +0.0531 [+0.0224, +0.0839] | +0.6087 | −0.556 [−0.707, −0.398] | −10.47 [−25.52, −5.91] | 0.0001 (floor 1/B) | 0.0007 | **No** (falsified) |
| qwen-instruct | −0.2951 [−0.3380, −0.2513] | +0.3439 | −0.049 [−0.280, +0.180] | −0.17 [−1.03, +0.58] | 0.670 | 0.670 | Inconclusive |
| llama-base | +0.0285 [−0.0072, +0.0656] | +0.1980 | (−0.170 [−0.352, +0.001]) | (−5.94 [−98.9, +0.03]) | (0.052) | — | **Not applicable** (precondition failed; removed from F_m) |

In all four models |β₁| > |β₀|: the adjusted contrast is larger in magnitude than the unadjusted one, and in qwen-base and qwen-instruct it has the opposite sign.

### H-C cross-model decision: Qwen-base vs Llama-instruct (not in a Holm family; no p declared)

| g₀ [95 % CI] | g₁ [95 % CI] | F [95 % CI] | Precondition | Band |
|---|---|---|---|---|
| −0.196 [−0.259, −0.130] | −0.066 [−0.378, +0.245] | +0.663 [−1.03, +0.97] | met | **Inconclusive** (interval only) |

The other five pairs are descriptive. Gaps are oriented as the first-listed model minus the second.

| Pair | g₀ [95 % CI] | F [95 % CI] | Band |
|---|---|---|---|
| qwen-base vs qwen-instruct | +0.153 [+0.087, +0.221] | −0.30 [−2.48, +0.91] | inconclusive |
| qwen-base vs llama-base | −0.171 [−0.237, −0.104] | −1.02 [−4.33, +0.76] | inconclusive |
| llama-instruct vs qwen-instruct | +0.348 [+0.293, +0.403] | +0.24 [−0.50, +0.91] | inconclusive |
| llama-instruct vs llama-base | +0.025 [−0.025, +0.075] | — | not applicable (g₀ CI includes 0) |
| qwen-instruct vs llama-base | −0.324 [−0.381, −0.267] | +0.55 [−0.36, +0.98] | inconclusive |

### H-D: θ(NAT-K\*, NAT-V\*), tested in all four models

The H-D extrapolation rule was not triggered.

| Model | s_K / s_V | Estimate | 95 % CI | p | Holm p | Verdict / reading |
|---|---|---|---|---|---|---|
| qwen-base | 1.70 / 0.59 | +4.783 | [+4.645, +4.924] | 0.0001* | 0.0007 | difference; K more disruptive |
| llama-instruct | 1.81 / 0.55 | +1.207 | [+1.135, +1.279] | 0.0001* | 0.0007 | difference; K more disruptive |
| qwen-instruct | 1.69 / 0.59 | +4.518 | [+4.379, +4.655] | 0.0001* | 0.0007 | difference; K more disruptive |
| llama-base | 1.88 / 0.53 | +1.450 | [+1.387, +1.514] | 0.0001* | 0.0006 | difference; K more disruptive |

Pre-declared reading for each model:
- supports the storage-level (K-dominant) account;
- falsifies the weight-projection (V-dominant) account.

Pre-stated caution: the precedents concern different quantities.

Descriptive comparisons:
- **Natural NAT-K − NAT-V** at the natural sizes: −0.932, −1.008, −0.852, −0.958. Every CI excludes 0: natural V
  error is more disruptive than natural K error.
- **ρ_K / ρ_V:**

  | Model | ρ_K | ρ_V |
  |---|---|---|
  | qwen-base | 0.072 | 0.209 |
  | llama-instruct | 0.058 | 0.192 |
  | qwen-instruct | 0.076 | 0.218 |
  | llama-base | 0.058 | 0.204 |

### H-E: θ(INH, LOC)

| Model | Estimate | 95 % CI | p | Holm p | Reading |
|---|---|---|---|---|---|
| qwen-base | +0.914 | [+0.843, +0.984] | 0.0001* | 0.0007 | INH > LOC: consistent with prior work |
| llama-instruct | +1.224 | [+1.160, +1.288] | 0.0001* | 0.0007 | INH > LOC: consistent with prior work |
| qwen-instruct | +0.911 | [+0.836, +0.986] | 0.0001* | 0.0007 | INH > LOC: consistent with prior work |
| llama-base | +1.207 | [+1.156, +1.260] | 0.0001* | 0.0006 | INH > LOC: consistent with prior work |

Descriptive non-additivity, median m^NAT/(m^LOC + m^INH) [95 % CI]:

| Model | Median [95 % CI] |
|---|---|
| qwen-base | 1.035 [1.000, 1.070] |
| llama-instruct | 1.006 [0.969, 1.041] |
| qwen-instruct | 0.997 [0.970, 1.036] |
| llama-base | 1.007 [0.984, 1.024] |

### H-F: family X, per-problem θ(NAT, C1), n = 200 shared

The 25 % guard is **not triggered**, so H-F stays a test.

| Model | SD of θ_{m,i} |
|---|---|
| qwen-base | 0.472 |
| qwen-instruct | 0.507 |
| llama-base | 0.442 |
| llama-instruct | 0.428 |

The limit is 1.25 × 0.497 = 0.621. The equivalence margin is the mean of the four margins, 0.221; b̂ ≥ 0.25 in all four models.

| Contrast | Estimate | 95 % CI | 90 % CI | p | Holm p | Verdict | Pre-declared reading |
|---|---|---|---|---|---|---|---|
| family (Llama − Qwen) | +0.328 | [+0.266, +0.389] | [+0.276, +0.379] | 0.0001* | 0.0003 | difference | family account **supported** |
| tuning (instruct − base) | −0.008 | [−0.071, +0.055] | [−0.060, +0.044] | 0.801 | 0.801 | equivalent | tuning account **falsified** (by the equivalence rule) |
| interaction ½[(a−b) − (c−d)] | +0.074 | [+0.011, +0.139] | [+0.021, +0.128] | 0.0214 | 0.0428 | difference | additivity **falsified** |

Scope as pre-declared: these are contrasts between four specific models. Family is confounded with architecture, size,
data and tokenizer, and the tuning contrast is measured under the raw completion prompt (L1, L2).

## d. Per-model secondary families F_m

| Model | Family size | Members | Reason |
|---|---|---|---|
| qwen-base | 7 | NAT − C1, H-B, H-C, H-D, Prov-K, Prov-V, H-E | — |
| llama-instruct | 7 | same as qwen-base | — |
| qwen-instruct | 7 | same as qwen-base | — |
| llama-base | 6 | as above without H-C | the H-C precondition failed |

H-B, H-C, H-D and H-E are in section c. The remaining members:

**NAT − C1.** The replication category is a recomputation-only mechanical reading under E1L §3.

| Model | Estimate | 95 % CI | 90 % CI | p | Holm p | Verdict | Replication category |
|---|---|---|---|---|---|---|---|
| qwen-base | −0.241 | [−0.307, −0.176] | [−0.296, −0.186] | 0.0001* | 0.0007 | difference | vs E1 (−0.253 [−0.325, −0.179]): **Replicated** |
| llama-instruct | +0.079 | [+0.022, +0.139] | [+0.031, +0.130] | 0.0098 | 0.0098 | difference | see below |
| qwen-instruct | −0.323 | [−0.394, −0.252] | [−0.383, −0.263] | 0.0001* | 0.0007 | difference | first estimate |
| llama-base | +0.013 | [−0.047, +0.073] | [−0.038, +0.064] | 0.669 | 0.669 | **equivalent** (±0.2095) | first estimate |

For llama-instruct, compared with E1L (+0.058 [−0.007, +0.121], verdict "equivalent"):
- the same sign;
- the E1L verdict "equivalent" is not reproduced: in-session it is "difference";
- under the E1L §3 categories, stated relative to Qwen's direction: "Not replicated".

**Provenance.** The hypothesis is "the natural error on that path is interchangeable with isotropic noise of its size".

| Model | θ(NAT-K, C1-K) [95 % CI] | θ(NAT-V, C1-V) [95 % CI] | p (both) | Holm p (both) |
|---|---|---|---|---|
| qwen-base | −1.233 [−1.300, −1.162] | +0.831 [+0.766, +0.895] | 0.0001* | 0.0007 |
| llama-instruct | −0.882 [−0.952, −0.812] | +0.890 [+0.826, +0.954] | 0.0001* | 0.0007 |
| qwen-instruct | −1.191 [−1.271, −1.113] | +0.740 [+0.669, +0.814] | 0.0001* | 0.0007 |
| llama-base | −0.932 [−0.999, −0.866] | +0.977 [+0.923, +1.032] | 0.0001* | 0.0006 |

Every CI excludes 0, so interchangeability is **falsified on both paths in all four models**:
- natural K error is less disruptive than isotropic K noise of its size;
- natural V error is more disruptive than isotropic V noise of its size.

**Descriptive contrasts (labelled, not tested):**

| Model | NAT − C2 | C2 − C1 | NAT − CV |
|---|---|---|---|
| qwen-base | +0.036 [−0.039, +0.110] equivalent | −0.277 [−0.346, −0.205] | +0.016 [−0.058, +0.087] equivalent |
| llama-instruct | +0.222 [+0.157, +0.285] | −0.143 [−0.208, −0.078] | +0.212 [+0.142, +0.279] |
| qwen-instruct | −0.103 [−0.172, −0.037] | −0.220 [−0.287, −0.152] | −0.074 [−0.147, −0.0002] |
| llama-base | +0.196 [+0.144, +0.249] | −0.183 [−0.241, −0.126] | +0.144 [+0.088, +0.199] |

**H-A-type predictors** (P_x − P_norm, NAT, descriptive):

| Model | P_val | P_anormK | P_out | Pooled over the six norm-matched arms |
|---|---|---|---|---|
| qwen-base | −0.105 [−0.133, −0.076] | −0.088 [−0.116, −0.059] | −0.091 [−0.117, −0.065] | +0.139 [+0.122, +0.156] |
| llama-instruct | +0.011 [−0.023, +0.044] | +0.016 [−0.016, +0.049] | −0.133 [−0.170, −0.095] | +0.163 [+0.140, +0.186] |
| qwen-instruct | −0.183 [−0.209, −0.158] | −0.186 [−0.213, −0.160] | −0.122 [−0.152, −0.092] | +0.049 [+0.030, +0.067] |
| llama-base | +0.166 [+0.136, +0.195] | +0.158 [+0.130, +0.187] | −0.093 [−0.132, −0.053] | +0.269 [+0.247, +0.291] |

**Sensitivity** (estimates only, as in the locked code). NAT − C1:

| Model | steps 2–8 | steps 9–64 | per-problem medians | excluding the 4 exposed problems | S ≥ 32 |
|---|---|---|---|---|---|
| qwen-base | −0.134 | −0.256 | −0.135 | −0.245 | −0.247 |
| llama-instruct | +0.028 | +0.082 | +0.062 | +0.076 | +0.070 |
| qwen-instruct | −0.191 | −0.322 | −0.295 | −0.318 | −0.323 |
| llama-base | +0.175 | −0.002 | +0.009 | +0.011 | +0.025 |

Every H-B, H-D and H-E sensitivity estimate has the same sign as the primary. They are listed in `v3_report.json`.

## e. Amendment 2: realized post-rounding deviation and truncation sensitivity

**Realized relative per-slot norm deviation (reported, not gated).**
- **Llama models:** every constructed arm has max ≤ 0.012, p99 ≤ 2.0e-3 and **no** step with a slot above 5 %.
- **Qwen models:**

| Model | Arm | mean | p99 ≤ | max | steps with a slot > 5 % | slots > 5 % | dominating slots |
|---|---|---|---|---|---|---|---|
| qwen-base | C1 | 8.2e-4 | 2.1e-2 | 0.225 | 2,507 / 12,235 | 2,685 | L0 K h0 63 %, L0 K h1 37 % |
| qwen-base | QA | 4.8e-4 | 7.5e-3 | 0.133 | 187 / 12,235 | 187 | L0 K h0 100 % |
| qwen-base | QO | 8.5e-4 | 2.2e-2 | 0.250 | 2,996 / 12,235 | 3,236 | L0 K h0 64 %, h1 36 % |
| qwen-base | CV | 1.2e-3 | 3.4e-2 | 0.301 | 5,959 / 12,235 | 7,083 | L0 K h0 52 %, h1 48 % |
| qwen-base | C1-K | 1.5e-3 | 3.6e-2 | 0.226 | 2,628 / 12,235 | 2,813 | L0 K h0 64 %, h1 36 % |
| qwen-base | C2, NAT-K\*, NAT-V\*, C1-V | ≤ 2.1e-4 | ≤ 8.0e-4 | ≤ 0.046 | 0 | 0 | — |
| qwen-instruct | C1 | 8.2e-4 | 2.1e-2 | 0.289 | 2,706 / 12,734 | 2,895 | L0 K h0 64 %, h1 36 % |
| qwen-instruct | QA | 4.7e-4 | 7.5e-3 | 0.127 | 224 / 12,734 | 224 | L0 K h0 99 % |
| qwen-instruct | QO | 8.4e-4 | 2.2e-2 | 0.260 | 3,090 / 12,734 | 3,371 | L0 K h0 62 %, h1 38 % |
| qwen-instruct | CV | 1.3e-3 | 3.6e-2 | 0.307 | 6,609 / 12,734 | 8,002 | L0 K h0 57 %, h1 43 % |
| qwen-instruct | C1-K | 1.5e-3 | 3.6e-2 | 0.298 | 2,667 / 12,734 | 2,862 | L0 K h0 63 %, h1 37 % |
| qwen-instruct | C2, NAT-K\*, NAT-V\*, C1-V | ≤ 2.1e-4 | ≤ 8.0e-4 | ≤ 0.045 | 0 | 0 | — |

NAT, NAT-K, NAT-V, LOC and INH have no constructed slots.

**Truncation sensitivity (EXPLORATORY).**
- **Llama models:** every contrast keeps 200/200 problems, median retained fraction 1.0, so the estimates are the
  primaries (differences only in the CI, from the separate random stream).
- **Qwen models:** the same holds for every contrast whose arms have no flagged slot (H-A in NAT, H-D, Prov-V, H-E,
  NAT − C2, NAT-K − NAT-V). The other contrasts lose most problems:

| Model | Contrast | kept / dropped | median retained fraction | estimate [95 % CI], verdict |
|---|---|---|---|---|
| qwen-base | NAT − C1 | 38 / 162 | 0.18 | −0.314 [−0.503, −0.119], difference |
| qwen-base | H-B QA − QO | 18 / 182 | 0.15 | +2.711 [+2.218, +3.141], difference |
| qwen-base | Prov-K | 36 / 164 | 0.19 | −1.174 [−1.373, −0.977], difference |
| qwen-base | C2 − C1 | 38 / 162 | 0.18 | −0.442 [−0.680, −0.220], difference |
| qwen-base | NAT − CV | 3 / 197 | 0.125 | −0.356 [−0.900, −0.002], "difference" with p 0.24 (n = 3) |
| qwen-base | H-C | 38 / 162 | 0.18 | S +0.021 [−0.916, +0.277]; f +0.08 [−5.65, +0.92]; **inconclusive** (primary: No) |
| qwen-base | H-A pooled over the six arms | 0 / 200 | — | not estimable |
| qwen-instruct | NAT − C1 | 29 / 171 | 0.20 | −0.264 [−0.503, −0.033], difference |
| qwen-instruct | H-B | 10 / 190 | 0.19 | +2.553 [+1.925, +3.132], difference |
| qwen-instruct | Prov-K | 27 / 173 | 0.16 | −1.187 [−1.453, −0.886], difference |
| qwen-instruct | C2 − C1 | 29 / 171 | 0.20 | −0.234 [−0.461, +0.003], **inconclusive** (primary: difference) |
| qwen-instruct | NAT − CV | 3 / 197 | 0.125 | −0.864 [−2.150, −0.068], p 0.26 (n = 3) |
| qwen-instruct | H-C | 29 / 171 | 0.20 | S −0.511 [−1.527, +0.299]; f −1.78 [−10.89, +0.82]; inconclusive |
| qwen-instruct | H-A pooled over the six arms | 0 / 200 | — | not estimable |

**H-F on the truncated θ(NAT, C1):** 9 shared problems. All three contrasts are inconclusive:

| Contrast | Estimate [95 % CI] | p |
|---|---|---|
| family | +0.155 [−0.278, +0.619] | 0.57 |
| tuning | +0.053 [−0.234, +0.332] | 0.74 |
| interaction | +0.009 [−0.268, +0.311] | 0.95 |

In Qwen the retained steps are the early steps (steps 1…f) of a minority of problems.

## f. Deviations and pre-declared fallbacks

**Deviations** (full text in `v3/DEVIATION_LOG_V3.md`):

| ID | What happened | Effect |
|---|---|---|
| V3-A1 | Amendment 1, before any GPU use: write-once Tier A arms, an exactly-once loader, a torn-line stop and repair rule, the ledger heartbeat | None on any estimand. The repair tool was never needed: there is no `repair_log.jsonl`. |
| V3-D04 | The qwen-base gates failed G2/G3 on CV (FP16 rounding in L0 K h0) | Diagnosed on CPU (DIAG_V3_D04) |
| V3-A2 | Amendment 2: the float64 construction check plus a bitwise rounding check; the post-rounding deviation is reported, not gated; the exploratory truncation sensitivity | Section e above; no estimand changed |
| V3-D05 | Manuscript correction of the E1/E1L tolerance wording | None on results |
| V3-D06 | The authors' S2 sensitivity was not adopted, because it cannot be computed from the locked records | None. The CPU proxy bounds the norm ratio at \|log ratio\| ≤ 4.1e-4. |
| V3-D07 | llama-base G6 runtime drift +18.1 % > 15 %; the authors decided to continue via the unlocked launcher | The run completed 200/200 under the locked 25 h ceiling (23.746 h), so the pre-declared truncation rule did not apply |
| V3-D08 (new, Phase 5) | Recomputation tolerances were declared before the recomputation ran | Defines the disagreement criterion only |

**Pre-declared fallbacks:**

| Rule | Triggered? |
|---|---|
| §8.3 H-C_m precondition | **Yes, for llama-base**: β₀ CI includes 0, so H-C is not applicable and is removed from F_m (6 tests). No for the other three models. |
| §8.4 cross-model H-C precondition | No (met). |
| §5 H-D extrapolation rule | No. s_K and s_V lie in [0.5, 2] for all four models. |
| §8.9 25 % SD guard | No. The maximum SD is 0.507, against a limit of 0.621. |
| §9 G6 reductions (CV → C2 → n 160) | No. The design is full, with n = 200. |
| Ceiling truncation | No. |
| V3-D07 truncation rule | No. |
| H-A exclusion for S < 8 | 0 problems in every model. |
| Equivalence informativeness (b̂ ≥ 0.25) | Met in all four models (b̂ 2.20–2.46), so equivalence verdicts were available. |
| Amendment-1 torn-line repair | Not used. |
| Amendment-2 truncation sensitivity | Applied. Its drops are in section e: most Qwen contrasts involving C1, QA/QO, CV or C1-K, and the pooled-arm H-A (not estimable in Qwen). |
