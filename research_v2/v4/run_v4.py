"""v4 driver (PROTOCOL_V4.md): scale study of weight-induced cache error, core arms, on one A40 pod.

The step and arms are in rv4.py, the ledgers / ceiling / watchdog / redaction / checkpoint packing in v4_ops.py. The prompt, the history
generation and the JS are the locked v3/E1 ones, imported unchanged.

Subcommands (CPU unless marked [POD]; every [POD] command needs --authorized-gpu-run and the authorization file of its tag)
  plan                         write v4/v4_sample.json (ids copied from the locked E1 sample; prompt SHA-256 per id; nothing new is drawn)
  status | ledger              progress per model and tag / GPU hours and USD (in-pod estimate, billing-based, used)
  billing --usd X --utc T      the corresponding author's billing line (cumulative USD from the RunPod billing page)
  download --model M           [POD] pinned snapshot of M (allow_patterns of the manifest), shard SHA-256 check, token-file check
  gm                           [POD, tag GM] CUDA test suite + mutation check + environment record -> v4_runs/gm.json
  gates --model M              [POD] G7, G5, G3 (BF16 rule), G2, G1, G4, cache geometry
  calibrate --model M          [POD] FP16 reference, NAT, C1 at alpha 0.5 / 1 / 2 on the 50 calibration problems
                               then: analyze_v4.py calibrate --model M ; tools/lock_v4.py --append-calibration <model>
  g6 --model qwen2.5-0.5b      [POD] runtime measurement on 3 calibration problems with the six test arms -> v4_runs/g6_measure.json
  g6-probe --model llama3.1-8b [POD] the same timing on the largest model (no records kept) -> v4_runs/g6_probe.json
  g6-design                    project the remaining study in USD; if it does not fit, drop Mistral; if it still does not fit, stop and report
                               -> v4_runs/design.json ; lock --append-design
  test --model M               [POD] the 100 fixed-order test problems, six arms, X13a sums for NAT; G8 on the first 3 problems (and after a restart)
  checkpoint --model M         pack v4_runs/<model> with the ledgers into a tar with SHA256SUMS; secret scan; then fetch with tools/v4_remote.py
  closeout                     on the laptop: is every checkpoint verified here, so that the corresponding author may delete the volume?
  repair --model M --file F --reason R    logged repair of a torn trailing record line (the problem is then rerun)

Order: GM; Step 0 (qwen2.5-0.5b -> llama3.2-1b-instruct -> llama3.2-1b); Step 0 decision (analyze_v4.py step0, lock --append-step0) and the G6
design; then qwen2.5-1.5b -> llama3.2-3b -> qwen2.5-3b -> qwen2.5-7b -> llama3.1-8b; then mistral-7b-v0.3, last and first to drop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
sys.path.insert(0, str(RV))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RV / "tools"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # r4: before anything imports torch; rv4.apply_determinism() completes the settings in main()

import run_controls as E                        # noqa: E402  (locked E1 module, unchanged; imports run_stage1 -> HF before torch)

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

import rv3.core as V                            # noqa: E402
import rv4                                      # noqa: E402
import v4_ops as O                              # noqa: E402
from rv2.boundary import would_stop_at_boundary  # noqa: E402
from rv2.engine import generate                 # noqa: E402
from rv2.policies import StaticFP16             # noqa: E402

RS1 = E.RS1
OUT = Path(os.environ.get("V4_RUNS", RV / "v4_runs"))
SAMPLE = HERE / "v4_sample.json"
LOCK = RV / "V4_PROTOCOL_LOCK.txt"
STEP0 = ["qwen2.5-0.5b", "llama3.2-1b-instruct", "llama3.2-1b"]
DROP_FIRST = "mistral-7b-v0.3"                    # r5: out of sample in H4, so the only model that can be dropped without touching H4
LADDER_NEW = ["qwen2.5-1.5b", "llama3.2-3b", "qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b", DROP_FIRST]
ORDER = STEP0 + LADDER_NEW
N_TEST, N_CAL = 100, 50                           # r5: the first 100 and 50 of the fixed seeded orders of e1_sample.json
G6_MODEL, G6_PROBE = "qwen2.5-0.5b", "llama3.1-8b"
S_STEPS = 64
N_ALPHA0, N_NUM, N_G8, N_G6, N_G5 = 10, 6, 3, 3, 8
TIERS = {"calibration": "V4calib", "test": "V4A", "gates": "V4gate", "g6": "V4g6"}
DRIFT = 0.25
DRIFT_REF_FACTOR = 1.48       # V4-D1 (corresponding author, after the qwen2.5-1.5b drift stop): the drift reference is the G6-design projection times the
                              # factor measured in Step 0 timing (real per-problem time / G6 formula, 1.48 on qwen2.5-0.5b and llama3.2-1b); the 25 % tolerance
                              # is unchanged. Decided from timing data only.
DEVICE = "cuda"
TEST_ARMS = [(a, 1.0, a) for a in rv4.CORE_ARMS]
IGNORED = {"wall_seconds"}


def mdir(tag):
    return OUT / tag


# ----------------------------------------------------------------------------- sample (plan)
def lf_sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def sample_ids(q: dict):
    """r5: the first 100 test and the first 50 calibration ids of the seeded orders. The G6 timing problems stay the last 3 of the original 100, so they
    are not calibration problems."""
    cal_all = q["calibration_train_ids"]
    return q["tierA_ids"][:N_TEST], cal_all[:N_CAL], cal_all[-N_G6:]


def make_sample(data=None) -> dict:
    if SAMPLE.exists():
        return json.loads(SAMPLE.read_text(encoding="utf-8"))
    q = json.loads((RV / "e1_sample.json").read_text(encoding="utf-8"))
    data = data or RS1.Data()
    test_ids, cal_ids, g6_ids = sample_ids(q)
    psha = {"test": {str(i): hashlib.sha256(data.problem("test", i)[1].encode()).hexdigest() for i in test_ids},
            "train": {str(i): hashlib.sha256(data.problem("train", i)[1].encode()).hexdigest() for i in cal_ids}}
    s = {"source": "ids copied from the locked E1 sample (e1_sample.json; the same ids as v3); no new draw, no outcome data",
         "e1_sample_sha256_lf": lf_sha((RV / "e1_sample.json").read_bytes()),
         "order": ORDER, "step0": STEP0, "drop_first": DROP_FIRST, "n_test": N_TEST, "n_calibration": N_CAL,
         "tierA_ids": test_ids, "calibration_train_ids": cal_ids,
         "gate_alpha0_train_ids": cal_ids[:N_ALPHA0], "gate_numerics_train_ids": cal_ids[:N_NUM], "gate_G5_test_ids": test_ids[:N_G5],
         "g6_train_ids": g6_ids, "exposed_test_ids_in_sample": sorted(set(test_ids) & set(q["exposed_test_ids_in_sample"])),
         "seed_tier_labels": TIERS, "master_seed_label": rv4.MASTER_SEED,
         "fewshot_prefix_sha256": hashlib.sha256(data.prefix.encode()).hexdigest(), "prompt_sha256": psha,
         "note": "Each model's teacher-forced history is its in-session high-precision greedy generation (<= 65 tokens, drift-marker stop)."}
    SAMPLE.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
    return s


# ----------------------------------------------------------------------------- state readers
def jload(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def lock_text():
    return LOCK.read_text(encoding="utf-8") if LOCK.exists() else ""


def lock_id():
    r = subprocess.run([sys.executable, str(RV / "tools" / "lock_v4.py"), "--lock-id"], capture_output=True, **O.TEXT)
    return r.stdout.strip() if r.returncode == 0 else None


def lock_verifies():
    return subprocess.run([sys.executable, str(RV / "tools" / "lock_v4.py"), "--verify"], capture_output=True).returncode == 0


def active_deviation():
    """V4-D1: the latest DEVIATION line appended to the lock, or None (as lock_v4.active_deviation)."""
    ids = [l.split()[1].rstrip(":") for l in lock_text().splitlines() if l.startswith("DEVIATION ")]
    return ids[-1] if ids else None


def gm_file():
    """gm.json, or gm_<ID>.json after a deviation (V4-D1: GM runs once more after each deviation)."""
    d = active_deviation()
    return OUT / (f"gm_{d}.json" if d else "gm.json")


def locked_line(prefix, path: Path):
    return path.exists() and any(l.startswith(prefix) and lf_sha(path.read_bytes()) in l for l in lock_text().splitlines())   # r6: as lock_v4.file_sha


def precision(m):
    return (jload(mdir(m) / "precision.json") or {}).get("precision", "fp16")


def gm_passed():
    d = active_deviation()
    return (jload(gm_file()) or {}).get("all_passed") and locked_line(f"GM {d}:" if d else "GM:", gm_file())


def gates_passed(tag):
    return bool((jload(mdir(tag) / "gates" / "gates.json") or {}).get("all_passed"))


def calibration_locked(tag):
    return locked_line(f"CALIBRATION {tag}:", mdir(tag) / "calibration.json")


def flag_locked(m):
    return precision(m) == "fp16" or locked_line(f"FLAG {m}:", mdir(m) / "precision.json")


def step0_ok():
    d = jload(OUT / "step0.json") or {}
    return d.get("decision") in ("pass", "author: proceed") and locked_line("STEP0:", OUT / "step0.json")


def design():
    return jload(OUT / "design.json")


def design_locked():
    return locked_line("DESIGN:", OUT / "design.json")


def complete(tag):
    return (mdir(tag) / "test_complete.json").exists()


def not_run(tag):
    return (mdir(tag) / "not_run.json").exists()


# ----------------------------------------------------------------------------- permission
def refuse(msg):
    sys.exit(f"Refusing to run: {msg}")


def drop_first_check(cmd, est):
    """r5: Mistral runs after the locked design and is the only model that can be dropped. The design may have dropped it; if not, it is dropped at its own
    start when the money left no longer covers its projected cost. Nothing else is ever cut automatically."""
    dz = design()
    if not dz or not design_locked():
        refuse(f"{DROP_FIRST} runs last, after the locked G6 design")
    if dz.get("drop_mistral"):
        refuse(f"the locked design dropped {DROP_FIRST}")
    need = (dz.get("per_tag") or {}).get(DROP_FIRST, {}).get("usd", 0.0)
    if cmd in ("download", "gates") and est["used_usd"] + need > O.stop_usd():
        mdir(DROP_FIRST).mkdir(parents=True, exist_ok=True)
        (mdir(DROP_FIRST) / "not_run.json").write_text(json.dumps({"reason": "dropped at its start: the money left does not cover its projected cost (section 10)",
                                                                   "used_usd": est["used_usd"], "projected_usd": need, "utc": O.utc(time.time())}), encoding="utf-8")
        refuse(f"{DROP_FIRST} is dropped: used ${est['used_usd']:.2f} + projected ${need:.2f} exceeds ${O.stop_usd():.2f}. Fetch the checkpoints and report")


def require(a, m, cmd, tag=None):
    """Every [POD] command: flag, deterministic settings, authorization file of the tag, lock, SKU, no token file, cost config and ceiling, then
    the order rules (GM: the first-boot image record)."""
    tag = tag or m
    if not a.authorized_gpu_run:
        refuse("pod commands need --authorized-gpu-run")
    if not rv4.determinism_ok():
        refuse(f"the deterministic settings are not active: {rv4.determinism_state()}")
    ok, why = O.check_auth(OUT, tag, lock_id())
    if not ok:
        refuse(why)
    if not lock_verifies():
        refuse("the v4 lock does not verify (tools/lock_v4.py --verify)")
    name = O.gpu_name()
    if not O.sku_ok(name):
        refuse(f"the GPU is {name!r}; D5 permits only an {O.SKU} (no GPU-type change mid-study)")
    tf = O.token_files()
    if tf:
        refuse(f"a Hugging Face token file exists ({len(tf)} found); the token must only be the HF_TOKEN variable")
    try:
        cfg = O.load_cost_config(OUT)
    except (FileNotFoundError, ValueError) as e:
        refuse(str(e))
    est = O.CostLedger(OUT, cfg).estimate()
    if est["used_usd"] >= O.stop_usd():
        refuse(f"the ${O.CEILING_USD:.0f} ceiling's stop level of ${O.stop_usd():.2f} is reached (used ${est['used_usd']:.2f})")
    if cmd == "gm" and O.image_record(OUT) is None:
        refuse(f"no first-boot digest record of {O.IMAGE_REPO}:{O.IMAGE_TAG} (cloud/pod_setup.sh writes image_digest.json)")
    if m == DROP_FIRST:
        drop_first_check(cmd, est)
    if cmd == "gm" or cmd == "download":
        return
    if not gm_passed():
        refuse("the GPU mutation check (gm) has not passed and been appended to the lock")
    order_rules(m, cmd)


def order_rules(m, cmd):
    tag = m
    if not (mdir(m) / "download.json").exists():
        refuse(f"{m} is not downloaded and verified (run_v4.py download)")
    if cmd == "g6-probe":
        if m != G6_PROBE or not (OUT / "g6_measure.json").exists():
            refuse("g6-probe runs on llama3.1-8b after the qwen2.5-0.5b G6 measurement")
        return
    k = ORDER.index(m)
    for p in ORDER[:k]:
        if not (complete(p) or not_run(p)):
            refuse(f"fixed run order: {p} has not completed its test problems")
    if m not in STEP0 and not step0_ok():
        refuse("the Step 0 decision (pass, or the authors' 'proceed') is not on file and appended to the lock")
    if m not in STEP0 and not design_locked():
        refuse("the G6 design is not fixed and appended to the lock")
    if not_run(tag):
        refuse(f"{tag} is recorded as not run")
    if cmd in ("calibrate", "g6", "test") and not gates_passed(tag):
        refuse(f"gates for {tag} have not passed")
    if cmd in ("calibrate", "g6", "test") and not flag_locked(m):
        refuse(f"{m} is BF16-flagged but the flag is not appended to the lock")
    if cmd == "test" and not calibration_locked(tag):
        refuse(f"the calibration of {tag} is not frozen and appended to the lock")
    if cmd == "test" and m == G6_MODEL and not (OUT / "g6_measure.json").exists():
        refuse("the G6 runtime measurement of qwen2.5-0.5b comes before its test problems")
    if cmd == "g6" and m != G6_MODEL:
        refuse("g6 is measured on qwen2.5-0.5b (NF4)")


# ----------------------------------------------------------------------------- processes
def ledgered(block, fn, gpu=True):
    cfg = O.load_cost_config(OUT)
    cost = O.CostLedger(OUT, cfg, sku=O.gpu_name())
    g = O.GpuLedger(OUT) if gpu else O.GpuLedger(OUT / "_cost_only")
    if not gpu:
        g.out.mkdir(parents=True, exist_ok=True)
    with O.Ledgers(g, cost, block) as led:
        ok = bool(fn(cost))
        led.completed = ok
    est = O.CostLedger(OUT, cfg).estimate()
    print(f"[ledger] GPU {O.GpuLedger(OUT).seconds() / 3600:.3f} h | USD used {est['used_usd']:.2f} of {O.CEILING_USD:.0f} "
          f"(stop at {O.stop_usd():.2f}; in-pod {est['inpod_usd']:.2f}, billing-based {est['billing_based_usd']})", flush=True)
    return ok


def load(m):
    return rv4.load_model(m, DEVICE, precision(m), "nf4", local_only=True)


class ModelSlot:
    """The one live model of a process that loads more than once. The slot holds the only reference, so the previous model is released before the
    next load starts. Callers read slot.be and slot.tok and keep no local alias or module reference across a load."""

    def __init__(self):
        self.be = self.tok = None

    def drop(self):
        self.be = self.tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, loader, *args, **kwargs):
        self.drop()
        self.be, self.tok = loader(*args, **kwargs)


def run_manifest(tag, extra=None):
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, timeout=120, **O.TEXT).stdout
        smi = subprocess.run(["nvidia-smi"], capture_output=True, timeout=20, **O.TEXT).stdout
    except Exception as e:  # noqa: BLE001
        freeze, smi = "", f"unavailable: {e}"
    head = subprocess.run(["git", "-C", str(RV.parent), "rev-parse", "HEAD"], capture_output=True, **O.TEXT).stdout.strip()
    import bitsandbytes
    import transformers
    d = {"utc": O.utc(time.time()), "tag": tag, "lock_id": lock_id(), "git_head": head, "python": platform.python_version(), "torch": torch.__version__,
         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
         "bitsandbytes": bitsandbytes.__version__, "gpu": O.gpu_name(), "gpu_environment": O.gpu_environment(), "determinism": rv4.determinism_state(),
         "image": O.image_record(OUT), "nvidia_smi": smi, "pip_freeze": freeze.splitlines(),
         "precision": precision(tag), **(extra or {})}
    (mdir(tag)).mkdir(parents=True, exist_ok=True)
    (mdir(tag) / "manifest.json").write_text(json.dumps(d, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- per-problem work
def prompt_ids(tok, data, split, idx, device=DEVICE):
    _, prompt = data.problem(split, idx)
    return tok(prompt, return_tensors="pt").input_ids.to(device)


def fp16_history(be, tok, ids_t):
    """The high-precision greedy history (the BF16 weights for a flagged model)."""
    res = generate(be, ids_t, StaticFP16(), S_STEPS + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids), instrument=False)
    return res.ids


def run_problem(be, tok, data, m, idx, split, arms, tier, dtype, acc=None, slots_for=(), hist=None, device=DEVICE):
    ids_t = prompt_ids(tok, data, split, idx, device)
    ref = hist if hist is not None else fp16_history(be, tok, ids_t)
    S_i = min(S_STEPS, len(ref) - 1)
    rl = rv4.reference_logits(be, ids_t, ref, S_i)
    recs = {}
    for arm, alpha, tag in arms:
        t0 = time.perf_counter()
        out = rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, arm, alpha, rv4.arm_seeds(m, arm, alpha, idx, tier), rl, dtype,
                                        acc if (acc is not None and arm == "NAT") else None, slots=(tag in slots_for))
        recs[tag] = {"index": idx, "split": split, "model": m, "arm": tag, "alpha": alpha, "S": S_i, "ref_len": len(ref), **out,
                     "ref_logits_finite": bool(torch.isfinite(rl).all()), "wall_seconds": time.perf_counter() - t0}
    return recs, ref


def integrity_errors(recs):
    errs = []
    for tag, r in recs.items():
        js = r["js"]
        if not js:
            errs.append(f"{tag}: no decode step (history of length 1)")
        if not all(x == x and abs(x) < 1e3 for x in js):
            errs.append(f"{tag}: non-finite JS")
        if js and js[0] != 0.0:
            errs.append(f"{tag}: step-1 JS is {js[0]!r}, must be exactly 0")
        if js and min(js) < 0:
            errs.append(f"{tag}: negative JS")
        for k, msg in (("construction_ok", "construction (float64, pre-rounding) deviates from the target norm by more than 1e-4"),
                       ("rounding_ok", "stored entry is not the correctly rounded value of the float64 pre-rounding entry"),
                       ("exact_ok", "a path that must equal kappa^F bitwise was changed"), ("natural_ok", "natural entry not stored bitwise"),
                       ("routing_reset", "routing overrides not reset"), ("finite_ok", "a non-finite logit or cache entry"),
                       ("ref_logits_finite", "non-finite reference logits")):
            if not r.get(k, False):
                errs.append(f"{tag}: {msg}")
    return errs


def slot_path(tag, split_dir, arm, idx):
    return mdir(tag) / split_dir / "slots" / f"{arm}_{idx}.npz"


def write_slots(tag, split_dir, recs):
    for arm, r in recs.items():
        if "_slots" in r:
            p = slot_path(tag, split_dir, arm, r["index"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **r.pop("_slots"))
            os.replace(tmp, p)


def done_ids(d: Path, tag):
    return RS1.load_done(d / f"{tag}.jsonl")


def append_new(d: Path, idx, recs):
    """Amendment 1: exactly once. Only the arm files that do not yet hold this problem are appended."""
    E.append_records(d, {t: r for t, r in recs.items() if idx not in done_ids(d, t)})


def torn_record_files(tag):
    bad = []
    for p in sorted(mdir(tag).glob("*/*.jsonl")):
        b = p.read_bytes()
        if not b:
            continue
        ok = b.endswith(b"\n")
        if ok:
            try:
                for line in b.decode("utf-8").splitlines():
                    if line.strip():
                        json.loads(line)
            except (ValueError, UnicodeDecodeError):
                ok = False
        if not ok:
            bad.append(p)
    return bad


def require_intact(tag):
    bad = torn_record_files(tag)
    if bad:
        refuse("incomplete or invalid record file(s): " + ", ".join(str(p) for p in bad)
               + ". Run `run_v4.py repair` on each (logged; the problem is then rerun). Records are never edited by hand.")


def canonical(recs):
    out = {}
    for t, r in recs.items():
        rr = {k: v for k, v in r.items() if k not in IGNORED and k != "_slots"}
        out[t] = json.dumps(rr, sort_keys=True)
        if "_slots" in r:
            out[t + "/slots"] = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for _, v in sorted(r["_slots"].items()))).hexdigest()
    return out


# ----------------------------------------------------------------------------- gates
def snapshot_dir(e):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(e["hf_id"], revision=e["revision"], allow_patterns=e["allow_patterns"], local_files_only=True))


def check_snapshot(e, snap: Path):
    """G7 (files): the snapshot directory is the pinned revision, every manifest shard is present with its SHA-256, no excluded weight file."""
    shards = {n: O.sha256_file(snap / n) == s["sha256"] if (snap / n).exists() else False for n, s in e["shards"].items()}
    excluded = [f for f in e["excluded_files"] if (snap / f).exists()]
    return {"snapshot": snap.name, "revision_pinned": snap.name == e["revision"], "shards_ok": shards, "excluded_present": excluded,
            "pass": bool(snap.name == e["revision"] and all(shards.values()) and not excluded)}


def g5(tok, data, s, e):
    rep = {"prompt_sha_ok": [], "roundtrip_ok": [], "bos_first": []}
    for idx in s["gate_G5_test_ids"]:
        _, p = data.problem("test", idx)
        rep["prompt_sha_ok"].append(hashlib.sha256(p.encode()).hexdigest() == s["prompt_sha256"]["test"][str(idx)])
        ids = tok(p).input_ids
        rep["roundtrip_ok"].append(tok.decode(ids, skip_special_tokens=True) == p)
        rep["bos_first"].append(tok.bos_token_id is not None and ids[0] == tok.bos_token_id)
    want_bos = e["family"] in ("llama", "mistral")
    drift = tok("The answer is 7.\n#### 7\n\nQuestion: Next", add_special_tokens=False).input_ids
    plain = tok("The answer is 7.\n#### 7", add_special_tokens=False).input_ids
    rep.update({"expected_bos": want_bos, "eos_token_id": tok.eos_token_id, "bos_token_id": tok.bos_token_id,
                "drift_marker_detected": bool(would_stop_at_boundary(tok, drift)), "no_false_drift": not would_stop_at_boundary(tok, plain)})
    rep["pass"] = bool(all(rep["prompt_sha_ok"]) and all(rep["roundtrip_ok"]) and all(b == want_bos for b in rep["bos_first"])
                       and tok.eos_token_id is not None and rep["drift_marker_detected"] and rep["no_false_drift"])
    return rep


def g3(be, tok, data, m, s, dtype):
    """Finite values and step-1 JS = 0 on 6 calibration problems, all six arms at alpha 1 (the records double as G2 input)."""
    allrecs, errs = [], []
    for idx in s["gate_numerics_train_ids"]:
        recs, _ = run_problem(be, tok, data, m, idx, "train", TEST_ARMS, TIERS["gates"], dtype)
        allrecs.append(recs)
        for t, r in recs.items():
            if not (r["finite_ok"] and r["ref_logits_finite"] and all(x == x for x in r["js"]) and r["js"] and r["js"][0] == 0.0):
                errs.append(f"{idx}/{t}")
    return not errs, errs, allrecs


def gates(m, s, data):
    tag = m
    e = rv4.manifest()[m]
    rep = {"started_utc": O.utc(time.time()), "model": m, "tag": tag, "hf_id": e["hf_id"], "revision": e["revision"]}
    snap = snapshot_dir(e)
    rep["G7_snapshot"] = check_snapshot(e, snap)
    if not rep["G7_snapshot"]["pass"]:
        return finish_gates(tag, rep, False)
    prec = precision(m)
    slot = ModelSlot()
    slot.load(load, m)
    rep["G7_config"] = {**rv4.config_snapshot(slot.be.model.config), "manifest": {k: e["geometry"][k] for k in rv4.GEOMETRY_KEYS}}
    rep["G7_config"]["pass"] = rv4.geometry_matches(rv4.config_snapshot(slot.be.model.config), e["geometry"])
    rep["G5_format"] = g5(slot.tok, data, s, e)
    ok3, errs3, recs3 = g3(slot.be, slot.tok, data, m, s, rv4.DTYPES[prec])
    rep["G3_finite"] = {"precision": prec, "pass": ok3, "failures": errs3}
    if not ok3 and prec == "fp16":
        (mdir(m) / "precision.json").write_text(json.dumps({"precision": "bf16", "rule": "D6 (section 5): G3 failed in FP16",
                                                            "fp16_failures": errs3, "utc": O.utc(time.time())}, indent=1), encoding="utf-8")
        prec = "bf16"
        slot.load(load, m)
        ok3, errs3, recs3 = g3(slot.be, slot.tok, data, m, s, rv4.DTYPES[prec])
        rep["G3_finite_bf16"] = {"pass": ok3, "failures": errs3}
        if not ok3:
            (mdir(m) / "not_run.json").write_text(json.dumps({"reason": "G3 failed in FP16 and in BF16 (section 5)", "utc": O.utc(time.time())}),
                                                  encoding="utf-8")
            return finish_gates(tag, rep, False)
    if not ok3:
        return finish_gates(tag, rep, False)
    dtype = rv4.DTYPES[prec]
    be, tok = slot.be, slot.tok                                      # no further load in this function
    rep["precision"] = prec
    rep["G2_integrity_errors"] = [x for recs in recs3 for x in integrity_errors(recs)]
    g1 = []
    for idx in s["gate_alpha0_train_ids"]:
        ids_t = prompt_ids(tok, data, "train", idx)
        ref = fp16_history(be, tok, ids_t)
        S_i = min(S_STEPS, len(ref) - 1)
        rl = rv4.reference_logits(be, ids_t, ref, S_i)
        cF = V.fp16_teacher_forced_cache(be, ids_t, ref, S_i)
        for arm in rv4.CORE_ARMS:
            r, c0 = rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, arm, 0.0, rv4.arm_seeds(m, arm, 0.0, idx, TIERS["gates"]), rl, dtype, return_cache=True)
            same = all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(c0.layers, cF.layers))
            g1.append({"index": idx, "arm": arm, "ok": bool(same and all(v == 0.0 for v in r["js"]) and r["routing_reset"])})
    rep["G1_alpha0_bitwise"] = {"n": len(g1), "all_identical": all(x["ok"] for x in g1), "failures": [x for x in g1 if not x["ok"]]}
    g4 = []
    for idx in s["gate_numerics_train_ids"]:
        ids_t = prompt_ids(tok, data, "train", idx)
        ref = fp16_history(be, tok, ids_t)
        S_i = min(S_STEPS, len(ref) - 1)
        rl = rv4.reference_logits(be, ids_t, ref, S_i)
        run = lambda lab: rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, "C1", 2.0, rv4.arm_seeds(m, "C1", 2.0, idx, lab), rl, dtype)["js"]  # noqa: E731
        a1, a2, a3 = run(TIERS["gates"]), run(TIERS["gates"]), run(TIERS["gates"] + "-other")
        g4.append((a1 == a2, a1 != a3))
    rep["G4_seeds"] = {"same_label_identical": all(x for x, _ in g4), "different_label_differs_in_n_of": [sum(y for _, y in g4), len(g4)]}
    geo = rv4.cache_geometry(be, prompt_ids(tok, data, "train", s["gate_alpha0_train_ids"][0]), dtype)
    g = e["geometry"]
    rep["G7_cache"] = {**geo, "n_dynamic_modules": be.n_dynamic,
                       "pass": bool((geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == (g["layers"], g["kv_heads"], g["head_dim"])
                                    and geo["all_mode_dtype"] and be.n_dynamic == 7 * g["layers"])}
    passed = bool(rep["G7_config"]["pass"] and rep["G5_format"]["pass"] and rep["G7_cache"]["pass"] and not rep["G2_integrity_errors"]
                  and rep["G1_alpha0_bitwise"]["all_identical"] and rep["G4_seeds"]["same_label_identical"]
                  and rep["G4_seeds"]["different_label_differs_in_n_of"][0] == len(g4))
    run_manifest(tag)
    return finish_gates(tag, rep, passed)


def finish_gates(tag, rep, passed):
    rep["all_passed"] = bool(passed)
    rep["finished_utc"] = O.utc(time.time())
    (mdir(tag) / "gates").mkdir(parents=True, exist_ok=True)
    (mdir(tag) / "gates" / "gates.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("G1_alpha0_bitwise",)}, indent=1, default=float)[:6000])
    print("ALL PASSED:", rep["all_passed"], flush=True)
    return passed


# ----------------------------------------------------------------------------- calibration
def aux_path(tag):
    return mdir(tag) / "calib" / "aux_state.pt"


def merge_aux(tag, idx, st):
    p = aux_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = torch.load(p) if p.exists() else {"included": []}
    if idx in cur["included"]:
        return
    if not cur["included"]:
        cur.update(st)
    else:
        cur["rhoK"] = torch.cat([cur["rhoK"], st["rhoK"]])
        cur["rhoV"] = torch.cat([cur["rhoV"], st["rhoV"]])
        cur["n_meas"] += st["n_meas"]
    cur["included"] = cur["included"] + [idx]
    tmp = p.with_suffix(".tmp")
    torch.save(cur, tmp)
    os.replace(tmp, p)


def calib_done(tag, idx):
    d = mdir(tag) / "calib"
    p = aux_path(tag)
    inc = set(torch.load(p)["included"]) if p.exists() else set()
    return idx in inc and all(idx in done_ids(d, t) for _, _, t in rv4.CAL_ARMS)


def calib_step(be, tok, data, m, tag, idx, dtype):
    d = mdir(tag) / "calib"
    acc = rv4.RhoAccumulator()
    recs, _ = run_problem(be, tok, data, m, idx, "train", rv4.CAL_ARMS, TIERS["calibration"], dtype, acc=acc)
    errs = integrity_errors(recs)
    if errs:
        (mdir(tag) / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}), encoding="utf-8")
        print("[INTEGRITY FAILURE]", errs[:3], flush=True)
        return False
    merge_aux(tag, idx, acc.state())
    append_new(d, idx, recs)                     # a crash between merge and append leaves no duplicates
    return True


# ----------------------------------------------------------------------------- problem loop with the USD ceiling
def block_loop(name, tag, ids, step_fn, done_fn, cost, expected_problem_s=None):
    import stage1_runtime as SR
    worst, n, tsum = 0.0, 0, 0.0
    for k, idx in enumerate(ids):
        if done_fn(idx):
            continue
        ok, est = cost.can_start(1.5 * worst + 90.0)
        if not ok:
            (mdir(tag) / "ceiling_stop.json").write_text(json.dumps({"before_problem": idx, "estimate": est, "utc": O.utc(time.time())}), encoding="utf-8")
            print(f"[CEILING] stopping before problem #{k}: used ${est['used_usd']:.2f}; the study is TRUNCATED, the ceiling is not raised.", flush=True)
            return False
        tele = SR.telemetry_now()
        why = SR.hardware_stop(tele)
        if why:
            print(f"[HARDWARE STOP] {why}", flush=True)
            return False
        tp = time.perf_counter()
        if not step_fn(idx):
            return False
        dt = time.perf_counter() - tp
        worst, n, tsum = max(worst, dt), n + 1, tsum + dt
        if expected_problem_s and n >= 10 and tsum / n > (1 + DRIFT) * expected_problem_s:
            (mdir(tag) / "drift_stop.json").write_text(json.dumps({"mean_problem_s": tsum / n, "expected_problem_s": expected_problem_s, "n": n,
                                                                   "utc": O.utc(time.time())}), encoding="utf-8")
            print(f"[DRIFT] {tag}: {tsum / n:.1f} s per problem > {1 + DRIFT:.2f} x {expected_problem_s:.1f} s; stop and report.", flush=True)
            return False
        if n <= 3 or n % 10 == 0:
            print(f"[{name}] #{k} idx={idx} {dt:.1f}s | USD {est['used_usd']:.2f} | {tele}", flush=True)
    return True


# ----------------------------------------------------------------------------- test problems
def test_step(be, tok, data, m, tag, idx, dtype, g8=False):
    d = mdir(tag) / "test"
    recs, ref = run_problem(be, tok, data, m, idx, "test", TEST_ARMS, TIERS["test"], dtype, slots_for=("NAT",))
    errs = integrity_errors(recs)
    if g8 and not errs:
        recs2, ref2 = run_problem(be, tok, data, m, idx, "test", TEST_ARMS, TIERS["test"], dtype, slots_for=("NAT",))
        same = ref == ref2 and canonical(recs) == canonical(recs2)
        O.append_jsonl(mdir(tag) / "g8.jsonl", {"index": idx, "bitwise_equal": same, "utc": O.utc(time.time())})
        if not same:
            (mdir(tag) / "g8_failure.json").write_text(json.dumps({"index": idx}), encoding="utf-8")
            print(f"[G8 FAILURE] {tag} problem {idx}: the two in-session runs differ; stop and report", flush=True)
            return False
    if errs:
        (mdir(tag) / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}), encoding="utf-8")
        print("[INTEGRITY FAILURE]", errs[:3], flush=True)
        return False
    write_slots(tag, "test", recs)
    append_new(d, idx, recs)
    if idx not in done_ids(d, "FP16_history"):
        E.append_records(d, {"FP16_history": {"index": idx, "ids": ref}})
    return True


def test_done(tag, idx):
    d = mdir(tag) / "test"
    return all(idx in done_ids(d, t) for _, _, t in TEST_ARMS) and slot_path(tag, "test", "NAT", idx).exists()


def expected_problem_seconds(tag):
    dz = design() or {}
    e = (dz.get("expected_problem_s") or {}).get(tag)
    return e * DRIFT_REF_FACTOR if e else e                            # V4-D1


def stored_canonical(tag, idx):
    """V4-D1: the stored records of a completed test problem in the form of canonical(): every arm's record without IGNORED fields, the NAT slot
    arrays' hash, and the stored FP16 history ids."""
    d = mdir(tag) / "test"
    out = {}
    for _, _, t in TEST_ARMS:
        rows = [json.loads(l) for l in (d / f"{t}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        r = next(x for x in rows if x["index"] == idx)
        out[t] = json.dumps({k: v for k, v in r.items() if k not in IGNORED}, sort_keys=True)
    z = np.load(slot_path(tag, "test", "NAT", idx))
    out["NAT/slots"] = hashlib.sha256(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()
    hist = next(json.loads(l)["ids"] for l in (d / "FP16_history.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["index"] == idx)
    return out, hist


def resume_check(be, tok, data, m, tag, ids, dtype):
    """V4-D1: before a model's test resumes, rerun its last completed problem (no record written) and require it bitwise equal to the stored
    records, FP16 history and NAT slots. Logged in resume_check.jsonl; a difference writes resume_check_failure.json and stops."""
    done = [i for i in ids if test_done(tag, i)]
    if not done:
        return True
    idx = done[-1]
    recs, ref = run_problem(be, tok, data, m, idx, "test", TEST_ARMS, TIERS["test"], dtype, slots_for=("NAT",))
    stored, hist = stored_canonical(tag, idx)
    same = canonical(recs) == stored and list(ref) == list(hist)
    O.append_jsonl(mdir(tag) / "resume_check.jsonl", {"index": idx, "bitwise_equal": same, "n_done": len(done), "utc": O.utc(time.time())})
    print(f"[RESUME CHECK] {tag} problem {idx}: {'bitwise equal to the stored records' if same else 'DIFFERS from the stored records'}", flush=True)
    if not same:
        (mdir(tag) / "resume_check_failure.json").write_text(json.dumps({"index": idx, "utc": O.utc(time.time())}), encoding="utf-8")
    return same


def run_test(m, s, data, cost):
    tag = m
    ids = s["tierA_ids"]
    be, tok = load(m)
    dtype = rv4.DTYPES[precision(m)]
    started = any(done_ids(mdir(tag) / "test", t) for _, _, t in TEST_ARMS)
    pending = [i for i in ids if not test_done(tag, i)]
    g8_ids = ids[:N_G8] if not started else pending[:1]                # G8: first 3 problems; after a restart, the next problem
    if started and pending and not resume_check(be, tok, data, m, tag, ids, dtype):
        return False
    for idx in g8_ids:
        if not test_done(tag, idx) and not test_step(be, tok, data, m, tag, idx, dtype, g8=True):
            return False
    ok = block_loop(f"test:{tag}", tag, ids, lambda i: test_step(be, tok, data, m, tag, i, dtype), lambda i: test_done(tag, i), cost,
                    expected_problem_seconds(tag))
    if ok and all(test_done(tag, i) for i in ids):
        (mdir(tag) / "test_complete.json").write_text(json.dumps({"tag": tag, "n": len(ids), "utc": O.utc(time.time())}), encoding="utf-8")
    return ok


# ----------------------------------------------------------------------------- G6 runtime and design
def g6_time(m, s, data, dtype_override=None):
    be, tok = load(m)
    dtype = dtype_override or rv4.DTYPES[precision(m)]
    per_arm, per_hist, finite = [], [], True
    for idx in s["g6_train_ids"]:
        t = time.perf_counter()
        ids_t = prompt_ids(tok, data, "train", idx)
        ref = fp16_history(be, tok, ids_t)
        per_hist.append(time.perf_counter() - t)
        t = time.perf_counter()
        rl = rv4.reference_logits(be, ids_t, ref, min(S_STEPS, len(ref) - 1))
        per_hist[-1] += time.perf_counter() - t
        recs, _ = run_problem(be, tok, data, m, idx, "train", TEST_ARMS, TIERS["g6"], dtype, hist=ref)
        finite = finite and all(r["finite_ok"] for r in recs.values())
        per_arm.append(sum(r["wall_seconds"] for r in recs.values()) / len(recs))
    e = rv4.manifest()[m]
    return {"model": m, "t_arm_problem_s": float(np.mean(per_arm)), "t_hist_ref_problem_s": float(np.mean(per_hist)), "finite": finite,
            "layers": e["geometry"]["layers"], "precision": str(dtype), "utc": O.utc(time.time())}


def plan_numbers():
    return json.loads((HERE / "plan" / "v4_plan_numbers.json").read_text(encoding="utf-8"))


def weight_bytes(pm):
    """Bytes read per decode pass: FP16 pass (all weights at 2 bytes) + NF4 pass (linear weights at the NF4 rate, the rest at 2 bytes)."""
    nf4 = plan_numbers()["nf4_bytes_per_param"]
    return 2.0 * pm["total_params"] + nf4 * pm["linear_params"] + 2.0 * (pm["total_params"] - pm["linear_params"])


def fit_runtime(meas, probe):
    """t_arm = a * layers + b * bytes, solved exactly from the two measurements; a negative coefficient is set to 0 and the other refitted on
    the larger model (conservative)."""
    pm = plan_numbers()["projection"][O.SKU]["models"]
    x1, x2 =(meas["layers"], weight_bytes(pm[meas["model"]])), (probe["layers"], weight_bytes(pm[probe["model"]]))
    A = np.array([x1, x2], float)
    y = np.array([meas["t_arm_problem_s"], probe["t_arm_problem_s"]])
    a, b = np.linalg.solve(A, y)
    if a < 0:
        a, b = 0.0, y[1] / x2[1]
    if b < 0:
        a, b = y[1] / x2[0], 0.0
    hist = max(meas["t_hist_ref_problem_s"] / meas["t_arm_problem_s"], probe["t_hist_ref_problem_s"] / probe["t_arm_problem_s"])
    return float(a), float(b), float(hist)


def project(a, b, hist_ratio, rate, models):
    """Remaining USD by model: GPU h = (100 + 3 G8) x (6 t_arm + t_hist) + 50 x (4 t_arm + t_hist) + 0.35 h gates; billed h = (GPU h x 1.2 + 0.25 h download) x 1.05."""
    pm = plan_numbers()["projection"][O.SKU]["models"]
    rows, total = {}, 0.0
    for t in models:
        t_arm = a * pm[t]["layers"] + b * weight_bytes(pm[t])
        t_hist = hist_ratio * t_arm
        gpu_h = ((N_TEST + N_G8) * (6 * t_arm + t_hist) + N_CAL * (4 * t_arm + t_hist)) / 3600 + 0.35
        billed_h = (gpu_h * 1.2 + 0.25) * 1.05
        usd = billed_h * rate
        rows[t] = {"t_arm_s": t_arm, "gpu_h": gpu_h, "billed_h": billed_h, "usd": usd, "expected_problem_s": 6 * t_arm + t_hist}
        total += usd
    return total, rows


def design_steps():
    """r5, section 10: the whole reduction order. First the full ladder, then without Mistral; after that, stop and report."""
    return [None, DROP_FIRST]


def g6_design(s):
    meas, probe = jload(OUT / "g6_measure.json"), jload(OUT / "g6_probe.json")
    if not (meas and probe):
        sys.exit("g6-design needs g6_measure.json (qwen2.5-0.5b) and g6_probe.json (llama3.1-8b)")
    if design_locked():
        sys.exit("the design is already locked; it is never revisited")
    cfg = O.load_cost_config(OUT)
    used = O.CostLedger(OUT, cfg).estimate()["used_usd"]
    a, b, hist = fit_runtime(meas, probe)
    steps = []
    for dropped in design_steps():
        models = [m for m in LADDER_NEW if m != dropped]
        tot, rows = project(a, b, hist, cfg["rate_usd_per_h"], models)
        storage = O.storage_allowance(cfg["volume_gb"], sum(r["billed_h"] for r in rows.values()))
        budget = O.stop_usd() - used - storage
        steps.append({"dropped": dropped, "projected_remaining_usd": tot, "storage_allowance_usd": storage, "budget_usd": budget})
        if tot <= budget:
            break
    ok = steps[-1]["projected_remaining_usd"] <= steps[-1]["budget_usd"]
    out = {"pass": ok, "drop_mistral": steps[-1]["dropped"] == DROP_FIRST, "models": models, "n_cal": N_CAL, "n_test": N_TEST,
           "fit": {"host_s_per_layer": a, "device_s_per_byte": b, "hist_ratio": hist}, "used_usd_at_design": used, "projection_steps": steps,
           "expected_problem_s": {t: r["expected_problem_s"] for t, r in rows.items()}, "per_tag": rows, "utc": O.utc(time.time())}
    if not ok:
        out["stop"] = "the projection exceeds the money left even without Mistral: stop and report. Nothing else is cut automatically, because H4 needs all seven Qwen/Llama models"
        (OUT / "design_failed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
        return False
    if out["drop_mistral"]:
        mdir(DROP_FIRST).mkdir(parents=True, exist_ok=True)
        (mdir(DROP_FIRST) / "not_run.json").write_text(json.dumps({"reason": "dropped by the G6 design (section 10)", "utc": O.utc(time.time())}), encoding="utf-8")
    (OUT / "design.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("next: tools/lock_v4.py --append-design")
    return True


# ----------------------------------------------------------------------------- download and checkpoint
def download(m):
    from huggingface_hub import snapshot_download
    e = rv4.manifest()[m]
    tok = O.pod_env("HF_TOKEN") or None                               # never printed or written; passed only to huggingface_hub
    p = Path(snapshot_download(e["hf_id"], revision=e["revision"], allow_patterns=e["allow_patterns"], token=tok))
    rep = check_snapshot(e, p)
    rep["token_files_after"] = len(O.token_files())
    rep["pass"] = rep["pass"] and rep["token_files_after"] == 0
    mdir(m).mkdir(parents=True, exist_ok=True)
    if rep["pass"]:
        (mdir(m) / "download.json").write_text(json.dumps({**rep, "utc": O.utc(time.time())}, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return rep["pass"]


def checkpoint(tag):
    import v4_secret_scan as SS
    tar = O.pack_checkpoint(OUT, tag, extra_files=[LOCK] if LOCK.exists() else [])
    hits = SS.scan_path(tar)
    if hits:
        blocked = tar.with_suffix(".tar.BLOCKED")
        os.replace(tar, blocked)
        for where, name, off in hits:
            print(f"SECRET SCAN HIT: {where}: {name} at byte {off} (value withheld)")
        sys.exit("checkpoint blocked by the secret scan; report to the corresponding author")
    print(f"checkpoint {tar.name} ({tar.stat().st_size / 1e6:.1f} MB), SHA-256 in {tar.name}.sha256, secret scan clean")
    return True


# ----------------------------------------------------------------------------- status
def status():
    s = jload(SAMPLE) or {}
    st = jload(OUT / "status.json") or {}
    for t in ORDER:
        d = mdir(t) / "test"
        n = len(done_ids(d, "NAT")) if d.exists() else 0
        print(f"{t:24s} downloaded {(mdir(t) / 'download.json').exists()!s:5} gates {gates_passed(t)!s:5} "
              f"precision {precision(t):4s} calib_locked {calibration_locked(t)!s:5} test {n:3d}/{len(s.get('tierA_ids', []))} "
              f"complete {complete(t)!s:5} not_run {not_run(t)!s:5} checkpointed {st.get(t, {}).get('checkpointed', False)}")
    print("GM passed:", bool(gm_passed()), "| Step 0 decision ok:", step0_ok(), "| design locked:", design_locked())


def main():
    O.install_redaction()
    rv4.apply_determinism()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "status", "ledger", "billing", "download", "gm", "gates", "calibrate", "g6", "g6-probe", "g6-design",
                                    "test", "checkpoint", "repair", "closeout"])
    ap.add_argument("--model", choices=ORDER)
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--usd", type=float)
    ap.add_argument("--utc")
    ap.add_argument("--file")
    ap.add_argument("--reason")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "plan":
        s = make_sample()
        print({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in s.items() if k not in ("note", "source")})
        return
    if a.cmd == "status":
        return status()
    if a.cmd == "ledger":
        cfg = O.load_cost_config(OUT)
        print(f"GPU ledger: {O.GpuLedger(OUT).seconds() / 3600:.3f} h")
        print(json.dumps(O.CostLedger(OUT, cfg).estimate(), indent=1))
        return
    if a.cmd == "billing":
        if a.usd is None or not a.utc:
            sys.exit("billing --usd X --utc YYYY-MM-DDTHH:MM:SSZ (entered by the corresponding author from the RunPod billing page)")
        O.add_billing_line(OUT, a.usd, a.utc)
        return
    if a.cmd == "g6-design":
        sys.exit(0 if g6_design(make_sample()) else 2)
    if a.cmd == "closeout":
        r = O.closeout_report(OUT, ORDER, DROP_FIRST)
        print(json.dumps(r, indent=1))
        print("every checkpoint is verified on the laptop: the corresponding author may delete the volume, then record volume_deleted_utc and a final billing line"
              if r["ready"] else "NOT READY: do not delete the volume")
        sys.exit(0 if r["ready"] else 1)
    if a.cmd == "gm":
        require(a, None, "gm", tag="GM")
        ok = ledgered("gm", lambda cost: subprocess.run([sys.executable, str(RV / "tools" / "v4_gpu_mutation_check.py"), "--authorized-gpu-run"]).returncode == 0)
        sys.exit(0 if ok else 1)
    if not a.model:
        sys.exit("--model is required")
    m, tag = a.model, a.model
    if a.cmd == "repair":
        import v3_repair_torn_line as RP
        if not (a.file and a.reason):
            sys.exit("repair --model M --file F --reason R")
        p = Path(a.file)
        kept, removed = RP.plan_repair(p.read_bytes())
        p.write_bytes(kept)
        O.append_jsonl(OUT / "repairs.jsonl", {"file": str(p), "removed_sha256": hashlib.sha256(removed).hexdigest(), "removed_bytes": len(removed),
                                               "reason": a.reason, "utc": O.utc(time.time())})
        print("repaired; add a DEVIATION_LOG_V4.md row; the problem is rerun by the next test/calibrate process")
        return
    if a.cmd == "checkpoint":
        checkpoint(tag)
        return
    require(a, m, a.cmd)
    if a.cmd == "download":
        sys.exit(0 if ledgered(f"download:{m}", lambda cost: download(m), gpu=False) else 1)
    require_intact(tag)
    s = make_sample()
    data = RS1.Data()
    mdir(tag).mkdir(parents=True, exist_ok=True)
    if a.cmd == "gates":
        sys.exit(0 if ledgered(f"gates:{tag}", lambda cost: gates(m, s, data)) else 1)
    if a.cmd == "calibrate":
        ids = s["calibration_train_ids"]

        def run(cost):
            be, tok = load(m)
            dtype = rv4.DTYPES[precision(m)]
            return block_loop(f"calib:{tag}", tag, ids, lambda i: calib_step(be, tok, data, m, tag, i, dtype), lambda i: calib_done(tag, i), cost)
        sys.exit(0 if ledgered(f"calibrate:{tag}", run) else 1)
    if a.cmd == "g6":
        def run(cost):
            r = g6_time(m, s, data)
            (OUT / "g6_measure.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            print(json.dumps(r, indent=1))
            return True
        sys.exit(0 if ledgered("g6", run) else 1)
    if a.cmd == "g6-probe":
        def run(cost):
            r = g6_time(m, s, data, dtype_override=torch.float16)
            (OUT / "g6_probe.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            print(json.dumps(r, indent=1))
            return True
        sys.exit(0 if ledgered("g6-probe", run) else 1)
    if a.cmd == "test":
        sys.exit(0 if ledgered(f"test:{tag}", lambda cost: run_test(m, s, data, cost)) else 1)


if __name__ == "__main__":
    main()
