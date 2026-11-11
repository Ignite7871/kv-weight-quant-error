"""Summarize bench_<label>.json files (CPU only)."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
H = Path(__file__).resolve().parent
def ci(xs):
    m = st.mean(xs); s = st.stdev(xs) if len(xs) > 1 else 0.0
    return m, 1.96 * s / len(xs) ** 0.5 if len(xs) > 1 else 0.0
for lab in sys.argv[1:]:
    r = json.loads((H / f"bench_{lab}.json").read_text())
    print(f"=== {lab}: {r['power_scheme'].split('(')[-1]} -> end {r['power_scheme_end'].split('(')[-1]}  load {r['load_s']:.0f}s")
    for p, v in r["micro"].items():
        for k in ("sync_wall_ms_blocks", "sync_event_ms_blocks", "async_wall_ms_blocks"):
            m, h = ci(v[k]); print(f"  micro {p:4s} {k:22s} {m:6.1f} ± {h:4.1f} ms  blocks={['%.1f'%x for x in v[k]]}")
    g = r["generation"]
    if isinstance(g, str):
        print("  generation:", g)
    else:
        d = defaultdict(list); ident = defaultdict(set)
        for x in g:
            if x["label"] == "timed":
                d[(x["precision"], x["engine"])].append(x["ms_per_step"]); ident[(x["precision"], x["problem"])].add(tuple(x["ids"]))
            else:
                print(f"  warm-up {x['precision']} {x['ms_per_step']:.1f} ms/step")
        for k in sorted(d):
            m, h = ci(d[k]); print(f"  gen {k[0]:4s} {k[1]:12s} {m:6.1f} ± {h:4.1f} ms/step (n={len(d[k])})")
        print("  token identity across engines/reps:", all(len(v) == 1 for v in ident.values()), {str(k): len(v) for k, v in ident.items()})
    cols = r["telemetry_columns"]; T = r["telemetry"]
    by = defaultdict(list)
    for row in T:
        try:
            by[row[1].split("_warm")[0]].append((float(row[3]), float(row[5]), float(row[6]), float(row[7]), row[8], row[9]))
        except (ValueError, IndexError):
            pass
    for ph, v in by.items():
        if ph in ("load", "done"):
            continue
        print(f"  tele {ph:22s} sm={st.mean(x[0] for x in v):6.0f}MHz util={st.mean(x[1] for x in v):4.0f}% temp={max(x[2] for x in v):3.0f}C power={st.mean(x[3] for x in v):5.1f}W pstates={sorted(set(x[4] for x in v))} reasons={sorted(set(x[5] for x in v))[:3]}")
