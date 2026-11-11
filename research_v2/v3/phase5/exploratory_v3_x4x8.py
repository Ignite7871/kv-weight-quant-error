"""EXPLORATORY analyses X4-X8 of v3/phase5/EXPLORATORY_PLAN_V3.md (dated entry 2026-10-31, Phase 7a; committed 21e16a5 before computing). Not locked.

Uses the locked analysis primitives (analyze_v3.py, imported unchanged): bootstrap, sign-flip test, Holm, ranking, Spearman, H-C statistics, bands.
Reads the locked records only; cross-checks against the locked report v3/phase5/v3_report_locked.json.
  venv/Scripts/python.exe research_v2/v3/phase5/exploratory_v3_x4x8.py   -> v3/phase5/exploratory_v3_x4x8.json, EXPLORATORY_RESULTS_X4X8.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RV))
import analyze_v3 as L                           # noqa: E402

MODELS = L.ORDER
RUNS = RV / "v3_runs"
HERE = Path(__file__).resolve().parent
SEED = {"X4": 20261204, "X5": 20261205, "X6": 20261206, "X7": 20261207, "X8": 20261208}
LABEL = "EXPLORATORY (EXPLORATORY_PLAN_V3.md, dated entry 2026-10-31, Phase 7a)"
PREDS = ["P_norm", "P_logit", "P_val", "P_anormK"]
ARMS = ["NAT", "C1", "C2", "QA", "QO", "CV", "NAT-K", "NAT-V", "NAT-K*", "NAT-V*", "C1-K", "C1-V", "LOC", "INH"]
LOCKED = json.loads((HERE / "v3_report_locked.json").read_text())
PAIR = ("qwen-base", "llama-instruct")


def load(m, arms):
    return {a: L.load_jsonl(RUNS / m / "tierA" / f"{a.replace('*', 'star')}.jsonl") for a in arms}


def raw_m(r):
    return float(np.mean(r["js"]))                                    # per-problem mean JS over steps 1..S, not floored


def boot_stat(v, rng, fn=np.mean):
    v = np.asarray(v, float)
    b = np.array([fn(v[k]) for k in L.boot_idx(len(v), rng)])
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def weights(n, rng):
    """Problem-clustered bootstrap weights, as in analyze_v3.hc_model."""
    W = np.zeros((L.B_BOOT, n))
    bi = L.boot_idx(n, rng)
    np.add.at(W, (np.repeat(np.arange(L.B_BOOT), n), bi.ravel()), 1.0)
    return W


def with_holm(res, key="p"):
    for (m, v), h in zip(res.items(), L.holm([v[key] for v in res.values()])):
        v["holm_p"] = float(h)


def pooled(per_problem, rng):
    sh = sorted(set.intersection(*[set(d) for d in per_problem.values()]))
    return L.summarize([np.mean([per_problem[m][i] for m in MODELS]) for i in sh], rng) if sh else None


def rho(nat, idx, pred):
    """H-A conventions: NAT, steps 2..S_i, problems with S_i < 8 or an undefined Spearman excluded (and counted)."""
    out, excl = {}, 0
    for i in idx:
        r = nat[i]
        if r["S"] < L.MIN_STEPS_HA:
            excl += 1
            continue
        v = L.spearman(r[pred][1:], r["js"][1:])
        if v != v:
            excl += 1
            continue
        out[i] = v
    return out, excl


# ----------------------------------------------------------------------------- X4, X5
def x4_x5():
    r4, r5 = np.random.default_rng(SEED["X4"]), np.random.default_rng(SEED["X5"])
    x4, x5, pp, checks = {}, {}, {}, {}
    for m in MODELS:
        nat = load(m, ["NAT"])["NAT"]
        idx = sorted(nat)
        R = {p: rho(nat, idx, p) for p in PREDS}
        x4[m] = {}
        for p in PREDS:
            v = np.array(list(R[p][0].values()))
            x4[m][p] = {"n": len(v), "excluded": R[p][1], "mean": float(v.mean()), "ci95": boot_stat(v, r4),
                        "median": float(np.median(v)), "share_positive": float((v > 0).mean())}
        both = sorted(set(R["P_logit"][0]) & set(R["P_anormK"][0]))
        d = {i: R["P_logit"][0][i] - R["P_anormK"][0][i] for i in both}
        s = L.summarize(list(d.values()), r5)
        s["verdict"] = "difference" if s["verdict"] == "difference" else "inconclusive"
        x5[m] = {**s, "excluded_problems": len(idx) - len(both)}
        pp[m] = d
        lk = LOCKED["per_model"][m]
        checks[m] = {"X4 mean(P_logit) - mean(P_norm) minus locked H-A D":
                     x4[m]["P_logit"]["mean"] - x4[m]["P_norm"]["mean"] - lk["HA"]["estimate"],
                     "X5 estimate minus (locked D_logit - D_anormK)":
                     s["estimate"] - (lk["HA"]["estimate"] - lk["descriptive"]["HA-type P_anormK vs P_norm (NAT)"]["estimate"])}
    with_holm(x5)
    return ({"label": LABEL, "per_model": x4},
            {"label": LABEL, "per_model": x5, "pooled_mean_over_models": pooled(pp, r5)}, checks)


# ----------------------------------------------------------------------------- X6
def x6():
    rng = np.random.default_rng(SEED["X6"])
    out = {}
    for m in MODELS:
        A = load(m, ARMS)
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        out[m] = {}
        for a in ARMS:
            mv = np.array([raw_m(A[a][i]) for i in idx])
            lg = np.array([L.logm(A[a][i]) for i in idx])
            out[m][a] = {"n": len(idx), "arith_mean": float(mv.mean()), "arith_ci95": boot_stat(mv, rng),
                         "geo_mean": float(math.exp(lg.mean())), "geo_ci95": [math.exp(x) for x in boot_stat(lg, rng)],
                         "median": float(np.median(mv)), "n_at_floor": int((mv < L.FLOOR).sum())}
    return {"label": LABEL, "unit": "nats", "per_model": out}


# ----------------------------------------------------------------------------- X7
def step_yx(a, b, key):
    y = np.log(np.maximum(np.asarray(a["js"][1:], float), L.FLOOR_STEP_JS)) - np.log(np.maximum(np.asarray(b["js"][1:], float), L.FLOOR_STEP_JS))
    x = np.log(np.maximum(np.asarray(a[key][1:], float), L.FLOOR_P)) - np.log(np.maximum(np.asarray(b[key][1:], float), L.FLOOR_P))
    return y, x


def u_rank(pa, pb):
    """u_t = [R(P_t^NAT) - R(P_t^C1)] / (2(S_i - 1)), average ranks over the pooled 2(S_i - 1) values of steps 2..S_i."""
    pa, pb = np.asarray(pa[1:], float), np.asarray(pb[1:], float)
    assert len(pa) == len(pb) and np.isfinite(pa).all() and np.isfinite(pb).all()
    R = L.rankdata(np.concatenate([pa, pb]))
    k = len(pa)
    return (R[:k] - R[k:]) / (2 * k)


def ols_suff(nat, c1, idx, keys):
    """Per problem: [X'X (row-major), X'y] of the pooled OLS y = b + sum_k g_k u^{P_k}, X = [1, u...]."""
    T = []
    for i in idx:
        y, _ = step_yx(nat[i], c1[i], "P_logit")
        X = np.column_stack([np.ones(len(y))] + [u_rank(nat[i][k], c1[i][k]) for k in keys])
        T.append(np.concatenate([(X.T @ X).ravel(), X.T @ y]))
    return np.asarray(T)


def ols_coef(S, p):
    S = np.atleast_2d(S)
    return np.linalg.solve(S[:, :p * p].reshape(-1, p, p), S[:, p * p:][..., None])[..., 0]


def shrink_block(b0, B0, coef, Coef, applicable):
    b1, B1 = coef[0], Coef[:, 0]
    sh, Sh = abs(b0) - abs(b1), np.abs(B0) - np.abs(B1)
    f = 1 - abs(b1) / abs(b0) if b0 != 0 else float("nan")
    Fr = 1 - np.abs(B1) / np.where(B0 != 0, np.abs(B0), np.nan)
    ci_f = [float(x) for x in np.nanpercentile(Fr, [2.5, 97.5])]
    p = float(min(1.0, max(1.0 / len(Sh), 2 * min((Sh <= 0).mean(), (Sh >= 0).mean()))))
    names = ["beta1", "gamma_L", "gamma_V"][:len(coef)]
    out = {n: float(c) for n, c in zip(names, coef)}
    out.update({f"{n}_ci95": [float(x) for x in np.percentile(Coef[:, k], [2.5, 97.5])] for k, n in enumerate(names)})
    out.update({"shrinkage": float(sh), "shrinkage_ci95": [float(x) for x in np.percentile(Sh, [2.5, 97.5])], "p": p,
                "fraction": float(f), "fraction_ci95": ci_f, "applicable": applicable,
                "band": L.band(f, ci_f) if applicable else "not applicable"})
    return out


def x7():
    rng = np.random.default_rng(SEED["X7"])
    a_, b_, c_, checks, T_joint, idx_m = {}, {}, {}, {}, {}, {}
    for m in MODELS:
        A = load(m, ["NAT", "C1"])
        nat, c1 = A["NAT"], A["C1"]
        idx = sorted(set(nat) & set(c1))
        idx_m[m] = idx
        W = weights(len(idx), rng)
        st = L.hc_stats(nat, c1, idx)                                   # locked sums: n, sum y, sum x, sum xy, sum xx
        Sb = W @ st
        b0, b1 = L.hc_betas(st.sum(0))
        B0, B1 = L.hc_betas(Sb)
        n, sy, sx, sxy, sxx = st.sum(0)
        g = (sxy - sx * sy / n) / (sxx - sx * sx / n)
        Gb = (Sb[:, 3] - Sb[:, 2] * Sb[:, 1] / Sb[:, 0]) / (Sb[:, 4] - Sb[:, 2] ** 2 / Sb[:, 0])
        xs = np.concatenate([step_yx(nat[i], c1[i], "P_logit")[1] for i in idx])
        xv = np.concatenate([step_yx(nat[i], c1[i], "P_val")[1] for i in idx])
        a_[m] = {"beta0": float(b0), "beta0_ci95": [float(x) for x in np.percentile(B0, [2.5, 97.5])],
                 "beta1": float(b1), "beta1_ci95": [float(x) for x in np.percentile(B1, [2.5, 97.5])],
                 "gamma": float(g), "gamma_ci95": [float(x) for x in np.percentile(Gb, [2.5, 97.5])],
                 "x_bar": float(sx / n), "x_bar_ci95": [float(x) for x in np.percentile(Sb[:, 2] / Sb[:, 0], [2.5, 97.5])],
                 "share_steps_x_negative": float((xs < 0).mean()), "x_bar_val": float(xv.mean()), "n_steps": int(n)}
        applicable = bool(LOCKED["per_model"][m]["HC"]["applicable"])
        Tj, Tl = ols_suff(nat, c1, idx, ["P_logit", "P_val"]), ols_suff(nat, c1, idx, ["P_logit"])
        T_joint[m] = Tj
        cj, Cj = ols_coef(Tj.sum(0), 3)[0], ols_coef(W @ Tj, 3)
        cl, Cl = ols_coef(Tl.sum(0), 2)[0], ols_coef(W @ Tl, 2)
        b_[m] = shrink_block(b0, B0, cj, Cj, applicable)
        c_[m] = shrink_block(b0, B0, cl, Cl, applicable)
        lk = LOCKED["per_model"][m]["HC"]
        checks[m] = {"beta0 minus locked": b0 - lk["beta0"], "beta1 minus locked": b1 - lk["beta1"],
                     "joint-OLS beta0 (Sum y / n) minus locked": Tj.sum(0)[9 + 0] / Tj.sum(0)[0] - lk["beta0"]}
    for blk in (b_, c_):
        app = {m: v for m, v in blk.items() if v["applicable"]}
        with_holm(app)
    # X7d: decision pair under the joint adjustment, same problems resampled for both models
    A0, B0m = PAIR
    sh = sorted(set(idx_m[A0]) & set(idx_m[B0m]))
    assert sh == idx_m[A0] == idx_m[B0m]
    W = weights(len(sh), rng)
    DA, DB = load(A0, ["NAT", "C1"]), load(B0m, ["NAT", "C1"])
    sa, sb = L.hc_stats(DA["NAT"], DA["C1"], sh), L.hc_stats(DB["NAT"], DB["C1"], sh)
    g0 = L.hc_betas(sa.sum(0))[0] - L.hc_betas(sb.sum(0))[0]
    G0 = L.hc_betas(W @ sa)[0] - L.hc_betas(W @ sb)[0]
    g1 = ols_coef(T_joint[A0].sum(0), 3)[0][0] - ols_coef(T_joint[B0m].sum(0), 3)[0][0]
    G1 = ols_coef(W @ T_joint[A0], 3)[:, 0] - ols_coef(W @ T_joint[B0m], 3)[:, 0]
    F = 1 - abs(g1) / abs(g0)
    ci_F = [float(x) for x in np.nanpercentile(1 - np.abs(G1) / np.where(G0 != 0, np.abs(G0), np.nan), [2.5, 97.5])]
    pre = bool(LOCKED["HC_cross"][f"{A0} vs {B0m}"]["precondition_met"])
    d_ = {"pair": f"{A0} minus {B0m}", "g0": float(g0), "g0_ci95": [float(x) for x in np.percentile(G0, [2.5, 97.5])],
          "g1J": float(g1), "g1J_ci95": [float(x) for x in np.percentile(G1, [2.5, 97.5])], "F_J": float(F), "F_J_ci95": ci_F,
          "precondition_met_locked": pre, "band": L.band(F, ci_F) if pre else "not applicable",
          "check: g0 minus locked": float(g0 - LOCKED["HC_cross"][f"{A0} vs {B0m}"]["g0"])}
    return {"label": LABEL, "X7a diagnostics (locked H-C, P_logit)": a_, "X7b joint rank-based (P_logit, P_val)": b_,
            "X7c rank-based P_logit only (control)": c_, "X7d decision pair, joint": d_}, checks


# ----------------------------------------------------------------------------- X8
def x8():
    rng = np.random.default_rng(SEED["X8"])
    out, checks = {}, {}
    for m in MODELS:
        A = load(m, ["NAT", "LOC", "INH"])
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        mv = {a: np.array([raw_m(A[a][i]) for i in idx]) for a in A}
        bi = L.boot_idx(len(idx), rng)
        r = {}
        for a in ("INH", "LOC"):
            t = L.theta(A, a, "NAT", idx)
            s = L.summarize(t, rng)
            ar = np.array([mv[a][k].sum() / mv["NAT"][k].sum() for k in bi])
            r[a] = {"geo_ratio": math.exp(s["estimate"]), "geo_ratio_ci95": [math.exp(x) for x in s["ci95"]],
                    "arith_ratio": float(mv[a].sum() / mv["NAT"].sum()), "arith_ratio_ci95": [float(x) for x in np.percentile(ar, [2.5, 97.5])],
                    "median_problem_ratio": float(np.median(mv[a] / mv["NAT"]))}
        shb = np.array([mv["INH"][k].sum() / (mv["INH"][k].sum() + mv["LOC"][k].sum()) for k in bi])
        out[m] = {"n": len(idx), "INH": r["INH"], "LOC": r["LOC"],
                  "INH_share_of_INH_plus_LOC": float(mv["INH"].sum() / (mv["INH"].sum() + mv["LOC"].sum())),
                  "INH_share_ci95": [float(x) for x in np.percentile(shb, [2.5, 97.5])]}
        checks[m] = {"log R_INH - log R_LOC minus locked theta(INH, LOC)":
                     math.log(r["INH"]["geo_ratio"]) - math.log(r["LOC"]["geo_ratio"]) - LOCKED["per_model"][m]["family"]["HE INH-LOC"]["estimate"]}
    return {"label": LABEL, "scope": "INH writes only the stored entries; Tier A logits, queries and tokens are FP16. "
                                     "INH is the cache component of the k/v-in-FP16 configuration, not that configuration.",
            "per_model": out}, checks


# ----------------------------------------------------------------------------- report
def f3(x):
    return f"{x:+.3f}"


def ci3(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def pv(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def to_md(o):
    Lm = [f"# EXPLORATORY results X4–X8 — 2026-10-31", "",
          "**All results here are EXPLORATORY and not blind.** Plan: EXPLORATORY_PLAN_V3.md, dated entry committed 21e16a5 before "
          "computing. Source: `exploratory_v3_x4x8.json` (generated by `exploratory_v3_x4x8.py` from the locked records). "
          "n = 200 per model.", ""]
    Lm += ["## X4. Spearman levels within NAT (steps 2…S_i)", "", "| Model | Predictor | mean ρ_S [95 % CI] | median | share > 0 | excluded |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for p in PREDS:
            v = o["X4"]["per_model"][m][p]
            Lm.append(f"| {m} | {p} | {f3(v['mean'])} {ci3(v['ci95'])} | {f3(v['median'])} | {v['share_positive']:.2f} | {v['excluded']} |")
    Lm += ["", "## X5. Paired ρ_S(P_logit) − ρ_S(P_anormK) within NAT", "", "| Model | estimate [95 % CI] | 90 % CI | p | Holm p | verdict | excluded |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        v = o["X5"]["per_model"][m]
        Lm.append(f"| {m} | {f3(v['estimate'])} {ci3(v['ci95'])} | {ci3(v['ci90'])} | {pv(v['p'])} | {pv(v['holm_p'])} | {v['verdict']} | {v['excluded_problems']} |")
    pm = o["X5"]["pooled_mean_over_models"]
    Lm += ["", f"Pooled mean over the four models: {f3(pm['estimate'])} {ci3(pm['ci95'])} (n = {pm['n']} shared problems)."]
    Lm += ["", "## X6. Absolute per-problem mean JS (nats)", "", "| Model | Arm | arithmetic mean [95 % CI] | geometric mean [95 % CI] | median | at floor |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for a in ARMS:
            v = o["X6"]["per_model"][m][a]
            Lm.append(f"| {m} | {a} | {v['arith_mean']:.3e} [{v['arith_ci95'][0]:.3e}, {v['arith_ci95'][1]:.3e}] | "
                      f"{v['geo_mean']:.3e} [{v['geo_ci95'][0]:.3e}, {v['geo_ci95'][1]:.3e}] | {v['median']:.3e} | {v['n_at_floor']} |")
    X7 = o["X7"]
    Lm += ["", "## X7a. Diagnostics of the locked H-C adjustment (x = log P_logit^NAT − log P_logit^C1)", "",
           "| Model | β0 [95 % CI] | β1 [95 % CI] | γ [95 % CI] | x̄ [95 % CI] | share of steps x < 0 | x̄ for P_val |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        v = X7["X7a diagnostics (locked H-C, P_logit)"][m]
        Lm.append(f"| {m} | {f3(v['beta0'])} {ci3(v['beta0_ci95'])} | {f3(v['beta1'])} {ci3(v['beta1_ci95'])} | {f3(v['gamma'])} {ci3(v['gamma_ci95'])} | "
                  f"{f3(v['x_bar'])} {ci3(v['x_bar_ci95'])} | {v['share_steps_x_negative']:.2f} | {f3(v['x_bar_val'])} |")
    for key, title in (("X7b joint rank-based (P_logit, P_val)", "X7b. Joint rank-based adjustment, y = β1J + γ_L u^logit + γ_V u^val"),
                       ("X7c rank-based P_logit only (control)", "X7c. Control: rank-based adjustment for P_logit alone")):
        joint = key.startswith("X7b")
        Lm += ["", f"## {title}", "", "| Model | β1 [95 % CI] | γ_L [95 % CI] | " + ("γ_V [95 % CI] | " if joint else "") +
               "shrinkage S [95 % CI] | p | Holm p | f [95 % CI] | band |", "|---|---|---|" + ("---|" if joint else "") + "---|---|---|---|---|"]
        for m in MODELS:
            v = X7[key][m]
            Lm.append(f"| {m} | {f3(v['beta1'])} {ci3(v['beta1_ci95'])} | {f3(v['gamma_L'])} {ci3(v['gamma_L_ci95'])} | " +
                      (f"{f3(v['gamma_V'])} {ci3(v['gamma_V_ci95'])} | " if joint else "") +
                      f"{f3(v['shrinkage'])} {ci3(v['shrinkage_ci95'])} | {pv(v['p'])} | {pv(v['holm_p']) if 'holm_p' in v else '—'} | "
                      f"{v['fraction']:+.2f} [{v['fraction_ci95'][0]:+.2f}, {v['fraction_ci95'][1]:+.2f}] | {v['band']} |")
    d = X7["X7d decision pair, joint"]
    Lm += ["", "## X7d. Decision pair under the joint adjustment", "",
           f"{d['pair']}: g0 {f3(d['g0'])} {ci3(d['g0_ci95'])}; g1J {f3(d['g1J'])} {ci3(d['g1J_ci95'])}; "
           f"F_J {d['F_J']:+.2f} [{d['F_J_ci95'][0]:+.2f}, {d['F_J_ci95'][1]:+.2f}]; band {d['band']}."]
    Lm += ["", "## X8. INH and LOC relative to NAT", "", o["X8"]["scope"], "",
           "| Model | R_INH geo [95 % CI] | R_INH arith [95 % CI] | median m_INH/m_NAT | R_LOC geo [95 % CI] | R_LOC arith [95 % CI] | INH share of INH+LOC [95 % CI] |",
           "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        v = o["X8"]["per_model"][m]
        I, Lo = v["INH"], v["LOC"]
        Lm.append(f"| {m} | {I['geo_ratio']:.3f} [{I['geo_ratio_ci95'][0]:.3f}, {I['geo_ratio_ci95'][1]:.3f}] | "
                  f"{I['arith_ratio']:.3f} [{I['arith_ratio_ci95'][0]:.3f}, {I['arith_ratio_ci95'][1]:.3f}] | {I['median_problem_ratio']:.3f} | "
                  f"{Lo['geo_ratio']:.3f} [{Lo['geo_ratio_ci95'][0]:.3f}, {Lo['geo_ratio_ci95'][1]:.3f}] | "
                  f"{Lo['arith_ratio']:.3f} [{Lo['arith_ratio_ci95'][0]:.3f}, {Lo['arith_ratio_ci95'][1]:.3f}] | "
                  f"{v['INH_share_of_INH_plus_LOC']:.3f} [{v['INH_share_ci95'][0]:.3f}, {v['INH_share_ci95'][1]:.3f}] |")
    Lm += ["", "## Consistency checks against the locked report (should be ~0)", ""]
    for k, v in o["checks"].items():
        for m, c in v.items():
            Lm.append(f"- {k} {m}: " + "; ".join(f"{n} = {x:+.2e}" for n, x in c.items()))
    return "\n".join(Lm) + "\n"


def main():
    X4, X5, ck45 = x4_x5()
    X7, ck7 = x7()
    X8, ck8 = x8()
    out = {"X4": X4, "X5": X5, "X6": x6(), "X7": X7, "X8": X8, "checks": {"X4/X5": ck45, "X7": ck7, "X8": ck8}}
    (HERE / "exploratory_v3_x4x8.json").write_text(json.dumps(out, indent=1, default=float))
    md = to_md(out)
    (HERE / "EXPLORATORY_RESULTS_X4X8.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
