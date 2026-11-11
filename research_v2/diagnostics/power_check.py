# Planning numbers from stored per-problem outcomes (descriptive; no new inference claims).
import json, sys, numpy as np
from scipy.stats import norm
R = sys.argv[1]
def load(path, key): return {r["index"]: r["status"] == "SOLVED" for r in json.load(open(f"{R}/{path}"))[key]}
b1 = json.load(open(f"{R}/gsm8k_n280_condition_1_and_3_backfill.json")); b2 = json.load(open(f"{R}/gsm8k_n280_condition_2_and_4.json")); a = json.load(open(f"{R}/gsm8k_n650_all_conditions.json"))
print("280 keys:", list(b1), list(b2))
def merge(*lists): return {r["index"]: r["status"] == "SOLVED" for l in lists for r in l}
k1 = [k for k in b1 if "1" in k][0]; k3 = [k for k in b1 if "3" in k][0]; k2 = "condition_2_pure_4bit"; k4 = "condition_4_entropy_override"
C = {"fp16": merge(b1[k1], a["condition_1"]), "nf4": merge(b2[k2], a["condition_2"]), "ent_p50": merge(b1[k3], a["condition_3"]), "override": merge(b2[k4], a["condition_4"]),
     "rand_c5": load("gsm8k_n930_condition_5_random_matched_budget.json", "condition_5_random_matched_budget")}
for p in ["p10", "p30", "p70", "p90"]:
    C[f"ent_{p}"] = load(f"gsm8k_multibudget_{p}_entropy_only.json", "results")
for p in ["p10", "p30", "p50", "p70", "p90"]:
    C[f"rand_{p}"] = load(f"gsm8k_multibudget_{p}_random_matched.json", "results")
idx = sorted(C["fp16"]); assert all(len(v) == 930 and sorted(v) == idx for v in C.values())
def pair(x, y):
    b = sum(C[x][i] and not C[y][i] for i in idx); c = sum(C[y][i] and not C[x][i] for i in idx); n = 930
    d = (b - c) / n; se = np.sqrt((b + c) / n - d * d) / np.sqrt(n)
    pd = (b + c) / n
    mde = (norm.ppf(.975) * np.sqrt(pd) + norm.ppf(.8) * np.sqrt(pd)) / np.sqrt(n)  # approx MDE, 80% power
    return b, c, d, se, mde, pd
print(f"{'pair':28s} b   c   diff(pp) 95%CI           discord  MDE80(pp)")
for x, y in [("fp16","nf4"),("override","rand_c5"),("ent_p10","rand_p10"),("ent_p30","rand_p30"),("ent_p50","rand_p50"),("ent_p70","rand_p70"),("ent_p90","rand_p90"),("ent_p50","nf4"),("rand_p90","nf4"),("ent_p90","nf4")]:
    b, c, d, se, mde, pd = pair(x, y)
    print(f"{x+' vs '+y:28s} {b:3d} {c:3d} {100*d:+6.2f} [{100*(d-1.96*se):+5.2f},{100*(d+1.96*se):+5.2f}]  {100*pd:5.1f}%  {100*mde:4.1f}")
# n needed to detect +2.58pp at 14.2% discordance, 80% power
pd, d = 0.142, 0.0258
n = ((norm.ppf(.975) * np.sqrt(pd) + norm.ppf(.8) * np.sqrt(pd - d * d)) / d) ** 2
print("n for 80% power at d=2.58pp, pd=14.2%:", round(n))
for d in [0.03, 0.05, 0.07]:
    for pd in [0.10, 0.15]:
        print(f"  d={d:.2f} pd={pd:.2f}: n80={((norm.ppf(.975)*np.sqrt(pd)+norm.ppf(.8)*np.sqrt(pd-d*d))/d)**2:.0f}")
# internal consistency of p10 advantage: pilot 280 vs extension 650
s280 = set(r["index"] for r in b1[k1]); 
for name, S in [("280 pilot", s280), ("650 ext", set(idx) - s280)]:
    b = sum(C["ent_p10"][i] and not C["rand_p10"][i] for i in S); c = sum(C["rand_p10"][i] and not C["ent_p10"][i] for i in S)
    print(f"p10 ent vs rand in {name}: b={b} c={c} diff={100*(b-c)/len(S):+.2f}pp (n={len(S)})")
