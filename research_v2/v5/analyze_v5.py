"""v5 analysis (PROTOCOL_V5.md sections 4, 6-8). CPU only; reads v5_runs and, for the anchor and the A-X references, the locked v4 records and
report, read-only.

  python research_v2/v5/analyze_v5.py calibrate --unit U        -> v5_runs/U/calibration.json (calibration problems only)
  python research_v2/v5/analyze_v5.py anchor                    -> v5_runs/anchor.json (U0 against v4: criteria 1-3 gated, H5 and bitwise share reported)
  python research_v2/v5/analyze_v5.py anchor --author-decision proceed|stop
                                                                record the authors' decision after an anchor failure (needs v5_runs/ANCHOR_AUTHOR_DECISION.txt
                                                                with PROTOCOL_V5, the lock id and the word)
  python research_v2/v5/analyze_v5.py analyze                   -> v5_runs/v5_report.json (refuses before every unit is complete, stopped or not run)

Statistics: the analyze_v3 primitives, imported unchanged (percentile problem bootstrap 10,000, two-sided paired sign-flip test 10,000, Holm, the
verdict rule, log mean-JS with the 1e-9 floor), the v4 calibration rules and readings (analyze_v4, unchanged). Each summary draws from its own stream
numpy.random.default_rng([20261111, key...]). X13a: the X13 functions with the unit's head_dim and the null stream default_rng([20261111, 13, unit, path]).
A-X (section 7.2): the references are the locked v4 estimates of named source models (v4_report.json, SHA-256 checked); the boundary is the midpoint;
a one-sided sign-flip test of theta - b in the direction of the observed side (10,000 flips), Holm over the two A models per contrast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(RV / "v3" / "phase8"))
import analyze_v3 as L                          # noqa: E402  (locked primitives, unchanged)
import analyze_v4 as A4                         # noqa: E402  (locked v4 calibration rules and readings, unchanged)
import analyze_x13 as X13                       # noqa: E402  (X13a functions, unchanged; head_dim passed explicitly)

OUT = Path(os.environ.get("V5_RUNS", RV / "v5_runs"))
V4_RUNS = Path(os.environ.get("V4_RUNS_READONLY", RV / "v4_runs"))
V4_REPORT_SHA256 = "df57a31483160e62351378576d14123893081249e06525ed5be5f5099758dccb"
MANIFEST = HERE / "V5_MODEL_MANIFEST.json"
GSM_SAMPLE = RV / "v4" / "v4_sample.json"
SEED = 20261111
N_NULL, B_FLIP = 200, 10_000
LN4, LN2 = math.log(4.0), math.log(2.0)
UNITS = {"U0": ("qwen2.5-0.5b", "nf4", "gsm8k"), "U1": ("qwen3-0.6b-base", "nf4", "gsm8k"), "U2": ("smollm2-1.7b", "nf4", "gsm8k"),
         "U3": ("qwen2.5-0.5b", "nf4", "wikitext2"), "U4": ("qwen2.5-0.5b", "hqq", "gsm8k"), "U5": ("mistral-7b-v0.3", "nf4", "wikitext2"),
         "U6": ("mistral-7b-v0.3", "hqq", "gsm8k")}
ANCHOR_MODEL = "qwen2.5-0.5b"
ARMS = ("NAT", "C1", "NAT-K", "C1-K", "NAT-V", "C1-V")
CONTRASTS = {"H1": ("NAT-K", "C1-K"), "H2": ("NAT-V", "C1-V"), "H3": ("NAT-K", "NAT-V"), "NAT-C1": ("NAT", "C1")}
STOP_FILES = ("ceiling_stop.json", "drift_stop.json", "integrity_failure.json", "g8_failure.json", "resume_check_failure.json", "hqq_state_failure.json")
FAMILIES = {  # name: (units, contrast or "H5", predicted sign)
    "A-H1": (("U1", "U2"), "H1", -1), "A-H2": (("U1", "U2"), "H2", +1), "A-H3": (("U1", "U2"), "H3", -1), "A-H5": (("U1", "U2"), "H5", None),
    "B-H2": (("U4", "U6"), "H2", +1), "B-H3": (("U4", "U6"), "H3", -1), "B-H5": (("U4", "U6"), "H5", None),
    "D-H2": (("U3", "U5"), "H2", +1), "D-H5": (("U3", "U5"), "H5", None)}
AX_SOURCES = {  # unit: contrast: {"size": v4 source models, "hd": v4 source models}  (section 7.2, D-V5-4)
    "U1": {c: {"size": ["qwen2.5-0.5b"], "hd": ["qwen2.5-1.5b"]} for c in ("NAT-C1", "H1")},
    "U2": {c: {"size": ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-3b"], "hd": ["qwen2.5-0.5b", "llama3.2-1b"]} for c in ("NAT-C1", "H1")}}
AX_FAMILY = {"NAT-C1": "AX-N (primary)", "H1": "AX-1 (secondary)"}
SMOLLM2_NOTE = ("SmolLM2's references are Qwen2.5 / Llama-3.2 estimates (no SmolLM2 model in v4): a 'size pattern' reading is also consistent with a "
                "family offset, and a 'head-dimension pattern' reading with an offset in the other direction (section 7.2)")
ANCHOR_D_MAX, ANCHOR_VERDICT_MIN, CAL_REL_TOL, BHAT_TOL, H5_TOL = 0.10, 0.2, 0.05, 0.25, 0.10
IGNORED_BITWISE = {"wall_seconds", "unit", "quantizer", "task"}


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]


def key_int(k) -> int:
    return k if isinstance(k, int) else int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:4], "big")


def rng_for(*keys):
    return np.random.default_rng([SEED] + [key_int(k) for k in keys])


def summarize(d, keys, margin=None, informative=True, B=None):
    return L.summarize(d, rng_for(*keys), margin, informative, B)


def load_arm(u, split, arm, runs=None):
    return L.load_jsonl((runs or OUT) / u / split / f"{arm}.jsonl")


def load_test(u, runs=None):
    return {a: load_arm(u, "test", a, runs) for a in ARMS}


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ----------------------------------------------------------------------------- calibration (v4 rules)
def calibrate(u):
    import torch
    d = OUT / u / "calib"
    recs = {t: L.load_jsonl(d / f"{t}.jsonl") for t in ("NAT", "C1_a0.5", "C1_a1", "C1_a2")}
    aux = torch.load(d / "aux_state.pt")
    cal = A4.calibrate_core(recs, aux["rhoK"].numpy(), aux["rhoV"].numpy())
    cal.update(unit=u, n_measure_steps=int(aux["n_meas"]), included=len(aux["included"]))
    p = OUT / u / "calibration.json"
    if p.exists():
        sys.exit(f"{p} exists; a frozen calibration is never rewritten")
    p.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(json.dumps(cal, indent=1))
    print(f"next: tools/lock_v5.py --append-calibration {u}")
    return cal


# ----------------------------------------------------------------------------- per-unit estimates
def x13a(u, idx, hd, runs=None):
    out = {}
    for path in ("K", "V"):
        rng_null = np.random.default_rng([SEED, 13, key_int(u), key_int(path)])
        xs = []
        for i in idx:
            with np.load((runs or OUT) / u / "test" / "slots" / f"NAT_{i}.npz") as z:
                norms, S1 = z[f"norms{path}"], z[f"S1{path}"]
            n = norms.shape[0]
            nu = X13.null_lognc(norms, rng_null, N_NULL, hd)
            xs.append(math.log(n * X13.cm_share(norms, S1)) - nu)
        out[path] = xs
    return out


def per_unit(u, cal, idx, exposed, hd):
    A = load_test(u)
    margin, inf = cal["margin_log_js"], cal["dose_response_informative"]
    res = {"unit": u, "model": UNITS[u][0], "quantizer": UNITS[u][1], "task": UNITS[u][2], "n_problems": len(idx), "head_dim": hd, "margin": margin,
           "dose_informative": inf, "rho_K": cal["rho_K"], "rho_V": cal["rho_V"], "rho_V_over_rho_K": cal["rho_V_over_rho_K"], "contrasts": {},
           "sensitivity": {}}
    keep = [i for i in idx if i not in set(exposed)]
    for h, (a, b) in CONTRASTS.items():
        res["contrasts"][h] = summarize(L.theta(A, a, b, idx), (u, h), margin, inf)
        sens = {"steps 2-8": summarize(L.theta(A, a, b, idx, lo=2, hi=8), (u, h, "2-8"))["estimate"],
                "per-problem median": summarize(L.theta(A, a, b, idx, stat="median"), (u, h, "median"))["estimate"]}
        if UNITS[u][2] == "gsm8k":
            sens["excluding exposed"] = summarize(L.theta(A, a, b, keep), (u, h, "exposed"))["estimate"] if keep else None
        long_ = [i for i in idx if A[a][i]["S"] >= 9 and A[b][i]["S"] >= 9]
        sens["steps 9-64"] = summarize(L.theta(A, a, b, long_, lo=9), (u, h, "9-64"))["estimate"] if long_ else None
        tv, dropped = A4.rdev_truncated_theta(A, a, b, idx)
        sens["rounding truncation (>5 %)"] = {"estimate": summarize(tv, (u, h, "rdev"))["estimate"] if tv else None, "dropped": dropped}
        res["sensitivity"][h] = sens
    res["descriptive"] = {"rel_K_mean": float(np.mean([np.mean(A["NAT"][i]["rel_K"]) for i in idx])),
                          "rel_V_mean": float(np.mean([np.mean(A["NAT"][i]["rel_V"]) for i in idx])),
                          "mean_S": float(np.mean([A["NAT"][i]["S"] for i in idx]))}
    return res


# ----------------------------------------------------------------------------- A-X (section 7.2)
def v4_report():
    p = V4_RUNS / "v4_report.json"
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != V4_REPORT_SHA256:
        raise ValueError("v4_report.json is not the locked v4 report")
    return json.loads(b)


def ax_references(rep=None) -> dict:
    """unit -> contrast -> {size, hd, boundary, hd_side (+1: theta above b), sources with their v4 values}."""
    rep = rep or v4_report()
    out = {}
    for u, cs in AX_SOURCES.items():
        out[u] = {}
        for c, src in cs.items():
            vals = {k: {m: rep["per_model"][m]["contrasts"][c]["estimate"] for m in ms} for k, ms in src.items()}
            size, hd = float(np.mean(list(vals["size"].values()))), float(np.mean(list(vals["hd"].values())))
            b = (size + hd) / 2.0
            out[u][c] = {"size_pred": size, "hd_pred": hd, "boundary": b, "hd_side": 1 if hd > b else -1, "sources": vals}
    return out


def one_sided_flip_p(d, keys, B=B_FLIP):
    """One-sided sign-flip p of mean(d) in the direction of the observed mean: (1 + #{flipped mean in that direction >= observed}) / (B + 1)."""
    d = np.asarray(d, float)
    s = 1.0 if d.mean() >= 0 else -1.0
    rng = rng_for(*keys)
    flips = rng.choice([-1.0, 1.0], size=(B, d.size))
    t = (flips * d).mean(axis=1) * s
    return float((1 + int((t >= s * d.mean() - 1e-15).sum())) / (B + 1)), int(s)


def ax_classify(summary, holm_p, ref):
    lo, hi = summary["ci95"]
    b, hs = ref["boundary"], ref["hd_side"]
    on_hd = (lo > b) if hs > 0 else (hi < b)
    on_size = (hi < b) if hs > 0 else (lo > b)
    if holm_p < 0.05 and on_hd:
        return "head-dimension pattern"
    if holm_p < 0.05 and on_size:
        return "size pattern"
    return "unresolved"


def ax(est, thetas, refs):
    """Per A model and contrast: the classification; Holm over the two A models per contrast; the joint NAT-C1 reading."""
    out = {}
    for c in ("NAT-C1", "H1"):
        us = [u for u in ("U1", "U2") if u in est]
        rows = {}
        ps = []
        for u in us:
            ref = refs[u][c]
            d = np.asarray(thetas[u][c], float) - ref["boundary"]
            p, side = one_sided_flip_p(d, (u, "AX", c))
            rows[u] = {"estimate": est[u]["contrasts"][c]["estimate"], "ci95": est[u]["contrasts"][c]["ci95"], "boundary": ref["boundary"],
                       "size_pred": ref["size_pred"], "hd_pred": ref["hd_pred"], "sources": ref["sources"], "p_one_sided": p,
                       "observed_side": "above" if side > 0 else "below"}
            ps.append(p)
        hp = L.holm(ps) if ps else []
        for u, h in zip(us, hp):
            rows[u]["holm_p"] = float(h)
            rows[u]["classification"] = ax_classify(est[u]["contrasts"][c], float(h), refs[u][c])
            if u == "U2":
                rows[u]["note"] = SMOLLM2_NOTE
        out[c] = {"family": AX_FAMILY[c], "units": rows}
    cls = [out["NAT-C1"]["units"][u]["classification"] for u in ("U1", "U2") if u in out["NAT-C1"]["units"]]
    if len(cls) == 2 and all(x == "head-dimension pattern" for x in cls):
        joint = "in these checkpoints the v4 size trend tracks head dimension"
    elif len(cls) == 2 and all(x == "size pattern" for x in cls):
        joint = "in these checkpoints the v4 size trend tracks size"
    else:
        joint = "neither factor alone accounts for it in these checkpoints"
    out["joint_reading_NAT-C1"] = {"reading": joint, "classifications": cls, "wording": "consistent with ... in these checkpoints; not causal (L-A)"}
    return out


# ----------------------------------------------------------------------------- families
def family(units, est, x13s, contrast, sign):
    us = [u for u in units if u in est]
    if contrast == "H5":
        keys = [(u, p) for u in us for p in ("K", "V")]
        ss = {k: summarize(x13s[k[0]][k[1]], (k[0], "H5", k[1])) for k in keys}
        ph = L.holm([ss[k]["p"] for k in keys]) if keys else []
        rd = {f"{u}/{p}": A4.reading_h5(ss[(u, p)], float(h)) for (u, p), h in zip(keys, ph)}
        return {"units": us, "summaries": {f"{u}/{p}": ss[(u, p)] for u, p in keys}, "holm_p": {f"{u}/{p}": float(h) for (u, p), h in zip(keys, ph)},
                "readings": rd, "result": A4.universal(rd), "n_models": len(us)}
    ps = [est[u]["contrasts"][contrast]["p"] for u in us]
    ph = L.holm(ps) if ps else []
    rd = {u: A4.reading_directional(est[u]["contrasts"][contrast], float(h), sign) for u, h in zip(us, ph)}
    return {"units": us, "holm_p": dict(zip(us, map(float, ph))), "readings": rd, "result": A4.universal(rd), "n_models": len(us)}


def unit_status(u):
    if (OUT / u / "not_run.json").exists():
        return "not run"
    if not (OUT / u / "calibration.json").exists():
        return "missing"
    if (OUT / u / "test_complete.json").exists():
        return "complete"
    if any((OUT / u / f).exists() for f in STOP_FILES):
        return "unresolved"                     # section 9: fewer than the design n
    return "running"


def design_n(u):
    if u == "U0":
        return 100
    d = jload(OUT / "design.json") or {}
    return int((d.get("n_test") or {}).get(u, 100))


def analyze():
    s = json.loads(GSM_SAMPLE.read_text(encoding="utf-8"))
    man = manifest()
    status = {u: unit_status(u) for u in UNITS}
    if any(v in ("running", "missing") for v in status.values()) and not (OUT / "STUDY_STOPPED.json").exists():
        sys.exit(f"Refusing: not every unit is complete, stopped or not run: {status}")
    usable = []
    for u in UNITS:
        if status[u] == "complete":
            n = len(set.intersection(*[set(v) for v in load_test(u).values()]))
            if n >= design_n(u):
                usable.append(u)
            else:
                status[u] = "unresolved"
    flagged = [u for u in UNITS if (OUT / u / "precision.json").exists()]
    est, x13s, thetas = {}, {}, {}
    for u in usable:
        cal = json.loads((OUT / u / "calibration.json").read_text(encoding="utf-8"))
        A = load_test(u)
        idx = sorted(set.intersection(*[set(v) for v in A.values()]))
        hd = man[UNITS[u][0]]["geometry"]["head_dim"]
        exposed = s["exposed_test_ids_in_sample"] if UNITS[u][2] == "gsm8k" else []
        est[u] = per_unit(u, cal, idx, exposed, hd)
        thetas[u] = {c: L.theta(A, a, b, idx) for c, (a, b) in CONTRASTS.items()}
        x13s[u] = x13a(u, idx, hd)
        est[u]["H5"] = {p: summarize(x13s[u][p], (u, "H5", p)) for p in ("K", "V")}
        est[u]["H5_C_per_layer_descriptive"] = c_per_layer(u, idx)
    refs = ax_references()

    def families(units_ok):
        e = {u: est[u] for u in units_ok}
        fam = {k: family(us, e, x13s, c, sg) for k, (us, c, sg) in FAMILIES.items()}
        fam["A-X"] = ax(e, thetas, refs)
        return fam
    rep = {"seed": SEED, "status": status, "design": jload(OUT / "design.json"), "flagged_bf16": flagged, "per_unit": est,
           "all_units": families(usable), "without_flagged": families([u for u in usable if u not in flagged]),
           "descriptive_comparisons": descriptive(est), "notes": {"D": "D tests the task and the history source together: the WikiText runs teacher-force "
                                                                     "the real text, v4 and the GSM8K units the model's own greedy history (section 1)",
                                                                "B": "cross-quantizer raw divergence is not like-for-like (HQQ 4.5 vs NF4 about 4.13 bits/weight)"}}
    disagree = {k: (rep["all_units"][k].get("result"), rep["without_flagged"][k].get("result")) for k in FAMILIES
                if rep["all_units"][k].get("result") != rep["without_flagged"][k].get("result")}
    rep["abstract_level"] = {"source": "without_flagged" if flagged else "all_units", "disagreements": disagree}
    (OUT / "v5_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v.get("result") for k, v in rep["all_units"].items() if k != "A-X"}, indent=1))
    print(json.dumps(rep["all_units"]["A-X"]["joint_reading_NAT-C1"], indent=1))
    return rep


def c_per_layer(u, idx):
    slots = {}
    for i in idx:
        with np.load(OUT / u / "test" / "slots" / f"NAT_{i}.npz") as z:
            slots[i] = {k: z[k] for k in z.files}
    return {p: X13.cm_per_layer(slots, p) for p in ("K", "V")}


def descriptive(est):
    """Section 8: the same model under another task or quantizer (descriptive; quantizers not like-for-like)."""
    out = {}
    for name, us in (("Qwen2.5-0.5B: U0 (NF4 GSM8K) / U3 (NF4 WikiText) / U4 (HQQ GSM8K)", ("U0", "U3", "U4")),
                     ("Mistral-7B: U5 (NF4 WikiText) / U6 (HQQ GSM8K)", ("U5", "U6"))):
        out[name] = {u: {c: est[u]["contrasts"][c]["estimate"] for c in CONTRASTS} | {"rho_V_over_rho_K": est[u]["rho_V_over_rho_K"]}
                     for u in us if u in est}
    return out


# ----------------------------------------------------------------------------- the anchor (section 4)
def v4_theta(a, b, idx):
    A = {x: L.load_jsonl(V4_RUNS / ANCHOR_MODEL / "test" / f"{x}.jsonl") for x in (a, b)}
    return L.theta(A, a, b, idx)


def bitwise_share(idx):
    """Reported: the share of U0 records (six arms, NAT slots, history) equal to v4's, ignoring wall time and the v5 label fields."""
    same = total = 0
    for a in ARMS:
        r5, r4 = load_arm("U0", "test", a), L.load_jsonl(V4_RUNS / ANCHOR_MODEL / "test" / f"{a}.jsonl")
        for i in idx:
            x = {k: v for k, v in r5[i].items() if k not in IGNORED_BITWISE}
            y = {k: v for k, v in r4[i].items() if k not in IGNORED_BITWISE}
            same += x == y
            total += 1
    slots_same = 0
    for i in idx:
        with np.load(OUT / "U0" / "test" / "slots" / f"NAT_{i}.npz") as z5, np.load(V4_RUNS / ANCHOR_MODEL / "test" / "slots" / f"NAT_{i}.npz") as z4:
            slots_same += sorted(z5.files) == sorted(z4.files) and all(np.array_equal(z5[k], z4[k]) for k in z5.files)
    h5_ = L.load_jsonl(OUT / "U0" / "test" / "FP16_history.jsonl")
    h4_ = L.load_jsonl(V4_RUNS / ANCHOR_MODEL / "test" / "FP16_history.jsonl")
    hist_same = sum(h5_[i]["ids"] == h4_[i]["ids"] for i in idx)
    return {"records_equal": same, "records_total": total, "share": same / total if total else None, "nat_slots_equal": slots_same,
            "histories_equal": hist_same, "n_problems": len(idx)}


def anchor_core(A, cal5, cal4, rep4_model, idx, x13_5):
    """Criteria 1-3 (gated) and H5 (reported). A: U0 test records; cal5 / cal4: calibrations; rep4_model: v4 report entry of Qwen2.5-0.5B."""
    margin4 = cal4["margin_log_js"]
    out = {"n_paired": len(idx), "v4_margin": margin4, "contrasts": {}}
    ok = True
    for h, (a, b) in CONTRASTS.items():
        t5, t4 = L.theta(A, a, b, idx), v4_theta(a, b, idx)
        dl = summarize(t5 - t4, ("anchor", h))
        c1 = bool(-margin4 < dl["ci90"][0] and dl["ci90"][1] < margin4 and abs(dl["estimate"]) <= ANCHOR_D_MAX)
        s4 = rep4_model["contrasts"][h]
        row = {"delta": dl, "criterion_1": c1, "theta_v4": s4["estimate"], "v4_category": A4.category(s4)}
        if abs(s4["estimate"]) >= ANCHOR_VERDICT_MIN:
            s5 = summarize(t5, ("anchor-v5", h), cal5["margin_log_js"], cal5["dose_response_informative"])
            row.update(v5_category=A4.category(s5), criterion_2=A4.category(s5) == A4.category(s4))
        else:
            row["criterion_2"] = "not applicable (|theta_v4| < 0.2)"
        ok = ok and c1 and row["criterion_2"] is not False
        out["contrasts"][h] = row
    c3 = {"rho_K": abs(cal5["rho_K"] / cal4["rho_K"] - 1) <= CAL_REL_TOL, "rho_V": abs(cal5["rho_V"] / cal4["rho_V"] - 1) <= CAL_REL_TOL,
          "b_hat": abs(cal5["dose_slope_b_hat"] - cal4["dose_slope_b_hat"]) <= BHAT_TOL}
    out["criterion_3"] = {**c3, "v5": {k: cal5[k] for k in ("rho_K", "rho_V", "dose_slope_b_hat")},
                          "v4": {k: cal4[k] for k in ("rho_K", "rho_V", "dose_slope_b_hat")}}
    ok = ok and all(c3.values())
    h5 = {p: {"v5": float(np.mean(x13_5[p])), "v4": rep4_model["H5"][p]["estimate"]} for p in ("K", "V")}
    for p in h5:
        h5[p]["within_0.10"] = abs(h5[p]["v5"] - h5[p]["v4"]) <= H5_TOL
    out["reported_H5"] = h5
    out["pass"] = bool(ok)
    return out


def lock_id():
    r = subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--lock-id"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


def anchor(author_decision=None):
    p = OUT / "anchor.json"
    if author_decision:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["decision"] == "pass":
            sys.exit("the anchor passed; no author decision is needed")
        f = OUT / "ANCHOR_AUTHOR_DECISION.txt"
        txt = f.read_text(encoding="utf-8") if f.exists() else ""
        lid = lock_id()
        if not ("PROTOCOL_V5" in txt and lid and lid in txt and author_decision in txt.split()):
            sys.exit("ANCHOR_AUTHOR_DECISION.txt (written by the corresponding author) must contain PROTOCOL_V5, the lock id and the decision word")
        d["decision"] = f"author: {author_decision}"
        d["author_decision_file_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        if author_decision == "proceed":
            d["label_for_all_v4_comparisons"] = "not agreed"
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(d["decision"])
        return d
    if p.exists():
        sys.exit("anchor.json exists; it is never recomputed")
    if not (OUT / "U0" / "test_complete.json").exists():
        sys.exit("U0 has not completed its test problems")
    rep4 = v4_report()
    A = load_test("U0")
    A4r = {a: L.load_jsonl(V4_RUNS / ANCHOR_MODEL / "test" / f"{a}.jsonl") for a in ARMS}
    idx = sorted(set.intersection(*[set(v) for v in A.values()]) & set.intersection(*[set(v) for v in A4r.values()]))
    cal5 = json.loads((OUT / "U0" / "calibration.json").read_text(encoding="utf-8"))
    cal4 = json.loads((V4_RUNS / ANCHOR_MODEL / "calibration.json").read_text(encoding="utf-8"))
    hd = manifest()[ANCHOR_MODEL]["geometry"]["head_dim"]
    res = anchor_core(A, cal5, cal4, rep4["per_model"][ANCHOR_MODEL], idx, x13a("U0", idx, hd))
    res["reported_bitwise"] = bitwise_share(idx)
    res["decision"] = "pass" if res["pass"] else "fail: stop and report; the authors decide (section 4)"
    p.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"pass": res["pass"], "bitwise": res["reported_bitwise"], "decision": res["decision"]}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "anchor", "analyze"])
    ap.add_argument("--unit")
    ap.add_argument("--author-decision", choices=["proceed", "stop"])
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.unit)
    elif a.cmd == "anchor":
        anchor(a.author_decision)
    else:
        analyze()


if __name__ == "__main__":
    main()
