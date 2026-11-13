"""v5 driver (PROTOCOL_V5.md): size vs head dimension (A), a second quantizer (B, HQQ) and a second task (D, WikiText-2), on one A40 pod.

The step and arms are the locked v4 ones (rv4.py), the units, seeds, backends and WikiText layer are in rv5.py, and the ledgers, ceiling, watchdog,
redaction and checkpoint packing in v5_ops.py. The GSM8K prompt, the history generation and the JS are the locked v3/E1 ones, imported unchanged.

Tags (one authorization file each, v5_runs/AUTHORIZED_V5_<tag>.txt): GM, U0, DESIGN, U1 ... U6.

Subcommands (CPU unless marked [POD]; every [POD] command needs --authorized-gpu-run and the authorization file of its tag)
  plan                        write v5/v5_wikitext_sample.json (laptop, before the lock; hashes and row indices only, D-V5-5)
  status | ledger             progress per unit / GPU hours and USD
  billing --usd X --utc T     the corresponding author's billing line (cumulative USD from the RunPod billing page)
  download --tag T            [POD] the pinned snapshot of the unit's model (DESIGN: Mistral) and, for WikiText units, the pinned parquet files
  gm                          [POD, tag GM] CUDA test suite + mutation check + environment record -> v5_runs/gm.json
  gates --tag U               [POD] G7, GW, G5, GH (HQQ), G3 (BF16 rule), G2, G1, G4, cache geometry
  calibrate --tag U           [POD] FP16 reference, NAT, C1 at alpha 0.5 / 1 / 2 on the 50 calibration problems
                              then: analyze_v5.py calibrate --unit U ; tools/lock_v5.py --append-calibration U
  test --tag U                [POD] the unit's test problems (n from the locked design; U0: 100), six arms, X13a sums for NAT; G8 on the first 3
  probe --probe P             [POD, tag DESIGN] design-point timing on the 3 timing problems: qwen-nf4, qwen-hqq, mistral-nf4, mistral-hqq
  design                      project the remaining units in USD; reduction order R1, R2, R3, else stop (R4) -> v5_runs/design.json
  checkpoint --tag T          pack v5_runs/<T> with the ledgers into a tar with SHA256SUMS; secret scan
  closeout [--tag T]          on the laptop: is every checkpoint verified here, so that the corresponding author may delete the volume?
                              (V5-D3: with the pod still running, it first takes a fresh manifest of the pod's checkpoints through
                              tools/v5_remote.py --tag T manifest and needs the pod's newest checkpoint of every tag here)
  d3-calibration-check --tag U1   V5-D3: recompute U1's calibration from the re-executed calibration records with the locked analysis code in a
                              scratch directory and require its hash to equal the lock's CALIBRATION U1 line (else stop)
  repair --tag U --file F --reason R    logged repair of a torn trailing record line (the problem is then rerun)

Order (section 9): GM -> U0 (anchor) -> anchor decision (laptop: analyze_v5.py anchor; lock --append-anchor) -> DESIGN (Mistral download, four
probes, design, lock --append-design) -> U1 -> U2 -> U3 -> U4 -> U5 -> U6. A and the anchor are never cut.

Rehearsal (section 10, standing local rule): V5_REHEARSAL=1 runs the same code on the laptop GPU, in a runs directory other than research_v2/v5_runs,
with authorization files that carry the word REHEARSAL (refused outside rehearsal; rehearsal refuses an A40). V5_REHEARSAL_N sets the number of
calibration and test problems (3 for the rehearsal runs).
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
sys.path.insert(0, str(RV / "v4"))
sys.path.insert(0, str(RV / "tools"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")      # section 10: no token is ever sent

import run_controls as E                        # noqa: E402  (locked E1 module, unchanged; imports run_stage1 -> HF before torch)

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

import rv3.core as V                            # noqa: E402
import rv4                                      # noqa: E402
import rv5                                      # noqa: E402
import v5_ops as O                              # noqa: E402
from rv2.boundary import would_stop_at_boundary  # noqa: E402
from rv2.engine import generate                 # noqa: E402
from rv2.policies import StaticFP16             # noqa: E402

RS1 = E.RS1
OUT = Path(os.environ.get("V5_RUNS", RV / "v5_runs"))
REAL_OUT = RV / "v5_runs"
GSM_SAMPLE = RV / "v4" / "v4_sample.json"
WIKI_SAMPLE = HERE / "v5_wikitext_sample.json"
LOCK = RV / "V5_PROTOCOL_LOCK.txt"
UNITS, ORDER = rv5.UNITS, rv5.ORDER
TAGS = ["GM", "U0", "DESIGN"] + ORDER[1:]
B_D_UNITS = ("U3", "U4", "U5", "U6")
N_TEST, N_TEST_R1, N_CAL = 100, 60, 50
S_STEPS = 64
N_ALPHA0, N_NUM, N_G8, N_TIMING, N_G5 = 10, 6, 3, 3, 8
DRIFT = 0.25
DRIFT_REF_FACTOR_D2 = 1.38   # V5-D2 (corresponding author, after the U1 drift stop, from timing data only: U1 ran 49.95 s against 36.16 s projected)
D3_UNIT, D3_REF = "U1", "d3_reference"     # V5-D3: U1 re-executed from scratch; its 13 surviving test problems are the bitwise reference
FACTOR = 1.48                                   # v4 D1: real per-problem time over the 3-problem timing (section 9)
WIKI_FACTOR = 64 / 61.1
GATES_H = 0.35
BILLED_PER_GPU_H = 19.07 / 15.601               # v4's measured billed / GPU-ledger hours
DOWNLOAD_H, OVERHEAD = 0.25, 1.05
PROBES = {"qwen-nf4": ("qwen2.5-0.5b", "nf4"), "qwen-hqq": ("qwen2.5-0.5b", "hqq"), "mistral-nf4": ("mistral-7b-v0.3", "nf4"),
          "mistral-hqq": ("mistral-7b-v0.3", "hqq")}
REDUCTIONS = [("none", {}, ()), ("R1", {u: N_TEST_R1 for u in B_D_UNITS}, ()), ("R1+R2", {u: N_TEST_R1 for u in B_D_UNITS}, ("U6",)),
              ("R1+R2+R3", {u: N_TEST_R1 for u in B_D_UNITS}, ("U6", "U5"))]
TEST_ARMS = [(a, 1.0, a) for a in rv4.CORE_ARMS]
IGNORED = {"wall_seconds"}
DEVICE = os.environ.get("V5_DEVICE", "cuda")


def rehearsal() -> bool:
    return os.environ.get("V5_REHEARSAL") == "1"


def rehearsal_n():
    v = os.environ.get("V5_REHEARSAL_N")
    return int(v) if (rehearsal() and v) else None


def mdir(tag):
    return OUT / tag


def jload(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def lf_sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------- samples and tasks
def gsm_sample():
    return jload(GSM_SAMPLE)


def wiki_sample():
    return jload(WIKI_SAMPLE)


def make_wiki_sample():
    if WIKI_SAMPLE.exists():
        return wiki_sample()
    toks = {m: rv5.load_tokenizer(m) for m in rv5.D_MODELS}
    lines = {s: rv5.wikitext_lines(s) for s in ("test", "train")}
    s = rv5.build_wikitext_sample(lines, toks)
    WIKI_SAMPLE.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
    return s


class Task:
    """The problem source of a unit: GSM8K (v4's sample and the model's own FP16 greedy history) or WikiText-2 (the locked sample; the real text,
    teacher-forced)."""

    def __init__(self, u, data=None):
        self.u, self.name = u, UNITS[u][2]
        self.wiki = self.name == "wikitext2"
        self.s = wiki_sample() if self.wiki else gsm_sample()
        self.data = data or (rv5.WikiData() if self.wiki else RS1.Data())

    def cal_ids(self):
        ids = [x["index"] for x in self.s["calibration"]] if self.wiki else self.s["calibration_train_ids"][:N_CAL]
        return ids[:rehearsal_n()] if rehearsal_n() else ids

    def test_ids(self, n):
        ids = [x["index"] for x in self.s["test"]] if self.wiki else self.s["tierA_ids"]
        return ids[:min(n, rehearsal_n())] if rehearsal_n() else ids[:n]

    def timing_ids(self):
        return [x["index"] for x in self.s["timing"]] if self.wiki else self.s["g6_train_ids"]

    def gate_alpha0_ids(self):
        return ([x["index"] for x in self.s["calibration"]] if self.wiki else self.s["gate_alpha0_train_ids"])[:N_ALPHA0]

    def gate_numerics_ids(self):
        return ([x["index"] for x in self.s["calibration"]] if self.wiki else self.s["gate_numerics_train_ids"])[:N_NUM]

    def prompt(self, split, idx):
        return self.data.problem(split, idx)[1]

    def prompt_ids(self, tok, split, idx, device=None):
        return tok(self.prompt(split, idx), return_tensors="pt").input_ids.to(device or DEVICE)

    def history(self, be, tok, ids_t, split, idx):
        if self.wiki:
            return rv5.history_ids(tok, self.data.problem(split, idx)[2])
        return fp16_history(be, tok, ids_t)


def fp16_history(be, tok, ids_t):
    """The high-precision greedy history (the BF16 weights for a flagged unit); v4's function."""
    res = generate(be, ids_t, StaticFP16(), S_STEPS + 1, tok.eos_token_id, 0, early_stop_fn=lambda ids: would_stop_at_boundary(tok, ids), instrument=False)
    return res.ids


# ----------------------------------------------------------------------------- state readers
def lock_text():
    return LOCK.read_text(encoding="utf-8") if LOCK.exists() else ""


def lock_id():
    r = subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--lock-id"], capture_output=True, **O.TEXT)
    return r.stdout.strip() if r.returncode == 0 else None


def lock_verifies():
    return subprocess.run([sys.executable, str(RV / "tools" / "lock_v5.py"), "--verify"], capture_output=True).returncode == 0


def active_deviation():
    ids = [l.split()[1].rstrip(":") for l in lock_text().splitlines() if l.startswith("DEVIATION ")]
    return ids[-1] if ids else None


def gm_file():
    d = active_deviation()
    return OUT / (f"gm_{d}.json" if d else "gm.json")


def locked_line(prefix, path: Path):
    return path.exists() and any(l.startswith(prefix) and lf_sha(path.read_bytes()) in l for l in lock_text().splitlines())


def precision(u):
    return (jload(mdir(u) / "precision.json") or {}).get("precision", "fp16")


def d3_active():
    return any(l.startswith("DEVIATION D3:") for l in lock_text().splitlines())


def d3_calibration_reproduced(u):
    return bool((jload(mdir(u) / "d3_calibration_check.json") or {}).get("equal"))


def gm_passed():
    d = active_deviation()
    return bool((jload(gm_file()) or {}).get("all_passed")) and locked_line(f"GM {d}:" if d else "GM:", gm_file())


def gates_passed(u):
    return bool((jload(mdir(u) / "gates" / "gates.json") or {}).get("all_passed"))


def calibration_locked(u):
    return locked_line(f"CALIBRATION {u}:", mdir(u) / "calibration.json")


def flag_locked(u):
    return precision(u) == "fp16" or locked_line(f"FLAG {u}:", mdir(u) / "precision.json")


def hqq_locked(u):
    return UNITS[u][1] != "hqq" or locked_line(f"HQQ {u}:", mdir(u) / "hqq.json")


def anchor_ok():
    d = jload(OUT / "anchor.json") or {}
    return d.get("decision") in ("pass", "author: proceed") and locked_line("ANCHOR:", OUT / "anchor.json")


def design():
    return jload(OUT / "design.json")


def design_locked():
    return locked_line("DESIGN:", OUT / "design.json")


def complete(u):
    return (mdir(u) / "test_complete.json").exists()


def not_run(u):
    return (mdir(u) / "not_run.json").exists()


def downloaded(name):
    return (OUT / "_downloads" / f"{name}.json").exists()


def unit_n(u):
    """The unit's design n: U0 100; the others from the locked design (R1 sets 60 for B and D)."""
    if u == rv5.ANCHOR:
        return N_TEST
    return int(((design() or {}).get("n_test") or {}).get(u, N_TEST))


# ----------------------------------------------------------------------------- permission
def refuse(msg):
    sys.exit(f"Refusing to run: {msg}")


def check_rehearsal(name):
    """(ok, reason). Rehearsal never on an A40 and never in the real runs directory; real runs never accept a rehearsal authorization file."""
    if rehearsal():
        if O.sku_ok(name):
            return False, "rehearsal mode on an A40: the rehearsal runs only on the laptop"
        if OUT.resolve() == REAL_OUT.resolve():
            return False, "rehearsal mode in research_v2/v5_runs: use a separate V5_RUNS directory"
        return True, "rehearsal"
    if not O.sku_ok(name):
        return False, f"the GPU is {name!r}; section 9 permits only an {O.SKU} (no GPU-type change mid-study)"
    return True, "ok"


def auth_ok(tag, lid):
    ok, why = O.check_auth(OUT, tag, lid)
    if not ok:
        return ok, why
    words = O.auth_path(OUT, tag).read_text(encoding="utf-8", errors="replace").split()
    if rehearsal() != ("REHEARSAL" in words):
        return False, ("a rehearsal authorization file must contain the word REHEARSAL" if rehearsal()
                       else "a REHEARSAL authorization file authorizes nothing outside the rehearsal")
    return True, "ok"


def require(a, tag, cmd):
    """Every [POD] command: flag, deterministic settings, authorization file of the tag, lock, SKU (or rehearsal), no token, cost config and the
    $14 stop, then the order rules and, for a unit's first commands, the unit start check."""
    if not a.authorized_gpu_run:
        refuse("pod commands need --authorized-gpu-run")
    if not rv4.determinism_ok():
        refuse(f"the deterministic settings are not active: {rv4.determinism_state()}")
    ok, why = auth_ok(tag, lock_id())
    if not ok:
        refuse(why)
    if not lock_verifies():
        refuse("the v5 lock does not verify (tools/lock_v5.py --verify)")
    ok, why = check_rehearsal(O.gpu_name())
    if not ok:
        refuse(why)
    tp = O.token_problems()
    if tp:
        refuse(f"a Hugging Face token is present ({', '.join(tp)}); v5 runs with no token (section 10)")
    try:
        cfg = O.load_cost_config(OUT)
    except (FileNotFoundError, ValueError) as e:
        refuse(str(e))
    est = O.CostLedger(OUT, cfg).estimate()
    O.set_stop_override(None)
    if tag == O.D1_UNIT:
        d1_stop(est, cfg)
    if d3_active() and tag in ("GM", D3_UNIT):              # V5-D3: phase u1d3 only (GM D3 and U1), the stop is the $15 ceiling
        O.set_stop_override(O.D3_STOP_USD)
    if est["used_usd"] >= O.stop_usd():
        refuse(f"the ${O.CEILING_USD:.0f} ceiling's stop level of ${O.stop_usd():.2f} is reached (used ${est['used_usd']:.2f})")
    if cmd == "gm":
        if O.image_record(OUT) is None:
            refuse(f"no first-boot digest record of {O.IMAGE_REPO}:{O.IMAGE_TAG} (cloud/pod_setup_v5.sh writes image_digest.json)")
        return
    if not gm_passed():
        refuse("the GPU mutation check (gm) has not passed and been appended to the lock")
    order_rules(tag, cmd)
    if tag in UNITS and tag != rv5.ANCHOR and cmd in ("download", "gates"):
        unit_start_check(tag, est)


def d1_stop(est, cfg):
    """Deviation V5-D1, decided by the corresponding author before any U6 data. At U6's start (its first command), if spent + U6's projected cost
    (design.json) + storage until the volume is deleted (the design's storage allowance) exceeds $14, the ledger stop is $16 for U6 only. The
    decision is written once to U6/d1_stop.json and every later U6 command rereads it. Arms, n, gates, analysis and every decision rule (the unit
    start check and the in-run ledger stop included, now read against the stop in force) are unchanged."""
    p = mdir(O.D1_UNIT) / "d1_stop.json"
    rec = jload(p)
    if rec is None:
        row = ((design() or {}).get("per_unit") or {}).get(O.D1_UNIT) or {}
        need = float(row.get("usd", 0.0))
        storage = O.storage_allowance(cfg["volume_gb"], float(row.get("billed_h", 0.0)), cfg["storage_usd_per_gb_month"])
        projection = est["used_usd"] + need + storage
        trig = projection > O.D1_THRESHOLD_USD
        rec = {"deviation": "V5-D1", "unit": O.D1_UNIT, "used_usd": est["used_usd"], "projected_unit_usd": need, "storage_usd": storage,
               "projection_usd": projection, "threshold_usd": O.D1_THRESHOLD_USD, "triggered": trig,
               "stop_usd": O.D1_STOP_USD if trig else O.CEILING_USD - O.CLOSEOUT_RESERVE_USD, "utc": O.utc(time.time())}
        mdir(O.D1_UNIT).mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[D1] at {O.D1_UNIT}'s start: spent ${rec['used_usd']:.2f} + projected ${need:.2f} + storage ${storage:.2f} = ${projection:.2f}; "
              f"{'above' if trig else 'not above'} ${O.D1_THRESHOLD_USD:.0f}, so the stop for {O.D1_UNIT} is ${rec['stop_usd']:.2f}", flush=True)
    O.set_stop_override(O.D1_STOP_USD if rec.get("triggered") else None)


def order_rules(tag, cmd):
    if (OUT / "STUDY_STOPPED.json").exists():
        refuse("the study is stopped (STUDY_STOPPED.json): close out")
    if tag == "DESIGN":
        if not (complete(rv5.ANCHOR) and anchor_ok()):
            refuse("the design point comes after U0 and the anchor decision (pass, or the authors' 'proceed') appended to the lock")
        if design_locked() and cmd in ("probe", "download"):
            refuse("the design is locked; it is never revisited")
        if cmd == "probe" and not (downloaded(PROBES[a_probe()][0])):
            refuse("probe: the model is not downloaded and verified")
        return
    u = tag
    k = ORDER.index(u)
    for p in ORDER[:k]:
        if not (complete(p) or not_run(p)):
            refuse(f"fixed run order: {p} has not completed its test problems")
    if u != rv5.ANCHOR and not anchor_ok():
        refuse("the anchor decision (pass, or the authors' 'proceed') is not on file and appended to the lock")
    if u != rv5.ANCHOR and not design_locked():
        refuse("the design is not fixed and appended to the lock")
    if not_run(u):
        refuse(f"{u} is recorded as not run")
    if cmd == "download":
        return
    m, q, task, _ = UNITS[u]
    if not downloaded(m) or (task == "wikitext2" and not downloaded("wikitext2")):
        refuse(f"{u}: the model{' or the dataset' if task == 'wikitext2' else ''} is not downloaded and verified (run_v5.py download)")
    if cmd in ("calibrate", "test") and not gates_passed(u):
        refuse(f"gates for {u} have not passed")
    if cmd in ("calibrate", "test") and not flag_locked(u):
        refuse(f"{u} is BF16-flagged but the flag is not appended to the lock")
    if cmd in ("calibrate", "test") and not hqq_locked(u):
        refuse(f"the HQQ state record of {u} is not appended to the lock (GH, section 3)")
    if cmd == "test" and not calibration_locked(u):
        refuse(f"the calibration of {u} is not frozen and appended to the lock")
    if cmd == "test" and u == D3_UNIT and d3_active() and not d3_calibration_reproduced(u):
        refuse(f"V5-D3: the re-executed calibration of {u} has not reproduced the locked calibration hash (d3-calibration-check)")


_PROBE = {"name": None}


def a_probe():
    return _PROBE["name"]


def unit_start_check(u, est):
    """Section 9: a unit starts only if its projected cost fits under the $14 stop; otherwise it and every later unit are recorded as not run and
    the study closes out. Units the locked design cut never start."""
    dz = design() or {}
    if u in (dz.get("cut_units") or []):
        refuse(f"the locked design cut {u} ({dz.get('reduction')})")
    need = ((dz.get("per_unit") or {}).get(u) or {}).get("usd", 0.0)
    if est["used_usd"] + need > O.stop_usd():
        rest = ORDER[ORDER.index(u):]
        for w in rest:
            if not (complete(w) or not_run(w)):
                mdir(w).mkdir(parents=True, exist_ok=True)
                (mdir(w) / "not_run.json").write_text(json.dumps({"reason": "unit start check (section 9): the money left below the $14 stop does not "
                                                                  "cover the projected cost", "first_unit": u, "used_usd": est["used_usd"],
                                                                  "projected_usd": need, "utc": O.utc(time.time())}), encoding="utf-8")
        (OUT / "STUDY_STOPPED.json").write_text(json.dumps({"reason": f"unit start check at {u}", "used_usd": est["used_usd"], "projected_usd": need,
                                                            "utc": O.utc(time.time())}), encoding="utf-8")
        refuse(f"{u} does not fit: used ${est['used_usd']:.2f} + projected ${need:.2f} > ${O.stop_usd():.2f}; {', '.join(rest)} recorded as not run, "
               f"study closes out")


# ----------------------------------------------------------------------------- processes
def ledgered(block, fn, gpu=True):
    cfg = O.load_cost_config(OUT)
    cost = O.CostLedger(OUT, cfg, sku=O.gpu_name())
    g = O.GpuLedger(OUT) if gpu else O.GpuLedger(OUT / "_cost_only")
    if not gpu:
        g.out.mkdir(parents=True, exist_ok=True)
    lk = O.GpuProcessLock(OUT, block)
    ok, why = lk.acquire()
    if not ok:
        refuse(why)
    try:
        with O.Ledgers(g, cost, block) as led:
            ok = bool(fn(cost))
            led.completed = ok
    finally:
        lk.release()
    est = O.CostLedger(OUT, cfg).estimate()
    print(f"[ledger] GPU {O.GpuLedger(OUT).seconds() / 3600:.3f} h | USD used {est['used_usd']:.2f} of {O.CEILING_USD:.0f} "
          f"(stop at {O.stop_usd():.2f}; in-pod {est['inpod_usd']:.2f}, billing-based {est['billing_based_usd']})", flush=True)
    return ok


class ModelSlot:
    """The one live model of a process that loads more than once (v4 r4): the previous model is released before the next load starts."""

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


def hqq_frozen_path(u):
    return mdir(u) / "hqq_state.pt"


def load_unit(u, prec=None, check_state=True):
    """The unit's backend. HQQ: a frozen state if GH recorded one, else a fresh quantization; with check_state, the state hash must equal hqq.json."""
    m, q, _, _ = UNITS[u]
    prec = prec or precision(u)
    frozen = None
    rec = jload(mdir(u) / "hqq.json") if q == "hqq" else None
    if rec and rec.get("frozen"):
        frozen = str(hqq_frozen_path(u))
    be, tok = rv5.load_model(m, DEVICE, prec, q, local_only=True, hqq_frozen=frozen)
    if q == "hqq" and check_state:
        import rv5_hqq as H
        h = H.state_hash(be.model)
        if not (rec and rec.get("state_sha256") == h):
            (mdir(u) / "hqq_state_failure.json").write_text(json.dumps({"loaded": h, "recorded": (rec or {}).get("state_sha256"), "utc": O.utc(time.time())}),
                                                            encoding="utf-8")
            refuse(f"{u}: the HQQ state hash {h[:16]} differs from hqq.json; stop and report")
    return be, tok


def run_manifest(u, extra=None):
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, timeout=120, **O.TEXT).stdout
        smi = subprocess.run(["nvidia-smi"], capture_output=True, timeout=20, **O.TEXT).stdout
    except Exception as e:  # noqa: BLE001
        freeze, smi = "", f"unavailable: {e}"
    head = subprocess.run(["git", "-C", str(RV.parent), "rev-parse", "HEAD"], capture_output=True, **O.TEXT).stdout.strip()
    import bitsandbytes
    import transformers
    try:
        hv = rv5.hqq_version()
    except Exception:  # noqa: BLE001
        hv = None
    d = {"utc": O.utc(time.time()), "tag": u, "unit": rv5.unit(u) if u in UNITS else None, "lock_id": lock_id(), "git_head": head,
         "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
         "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__, "hqq": hv, "gpu": O.gpu_name(),
         "gpu_environment": O.gpu_environment(), "determinism": rv4.determinism_state(), "image": O.image_record(OUT), "rehearsal": rehearsal(),
         "nvidia_smi": smi, "pip_freeze": freeze.splitlines(), "precision": precision(u) if u in UNITS else None, **(extra or {})}
    mdir(u).mkdir(parents=True, exist_ok=True)
    (mdir(u) / "manifest.json").write_text(json.dumps(d, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- per-problem work
def run_problem(be, tok, task, u, idx, split, arms, tier, dtype, acc=None, slots_for=(), hist=None, device=None, seeds_fn=None):
    """One problem, every arm. Records carry unit, quantizer and task; for WikiText units, the argmax ids are stored only as a hash (D-V5-5)."""
    device = device or DEVICE
    m, q, tname, _ = UNITS[u]
    ids_t = task.prompt_ids(tok, split, idx, device)
    ref = hist if hist is not None else task.history(be, tok, ids_t, split, idx)
    S_i = min(S_STEPS, len(ref) - 1)
    rl = rv4.reference_logits(be, ids_t, ref, S_i)
    seeds_fn = seeds_fn or (lambda arm, alpha: rv5.arm_seeds(u, arm, alpha, idx, tier))
    recs = {}
    for arm, alpha, tag in arms:
        t0 = time.perf_counter()
        out = rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, arm, alpha, seeds_fn(arm, alpha), rl, dtype,
                                        acc if (acc is not None and arm == "NAT") else None, slots=(tag in slots_for))
        r = {"index": idx, "split": split, "unit": u, "model": m, "quantizer": q, "task": tname, "arm": tag, "alpha": alpha, "S": S_i,
             "ref_len": len(ref), **out, "ref_logits_finite": bool(torch.isfinite(rl).all()), "wall_seconds": time.perf_counter() - t0}
        recs[tag] = rv5.public_record(r, tname)
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


def slot_path(u, split_dir, arm, idx):
    return mdir(u) / split_dir / "slots" / f"{arm}_{idx}.npz"


def write_slots(u, split_dir, recs):
    for arm, r in recs.items():
        if "_slots" in r:
            p = slot_path(u, split_dir, arm, r["index"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **r.pop("_slots"))
            os.replace(tmp, p)


def done_ids(d: Path, tag):
    return RS1.load_done(d / f"{tag}.jsonl")


def append_new(d: Path, idx, recs):
    """Exactly once (v3 amendment 1): only the arm files that do not yet hold this problem are appended."""
    E.append_records(d, {t: r for t, r in recs.items() if idx not in done_ids(d, t)})


def torn_record_files(u):
    bad = []
    for p in sorted(mdir(u).glob("*/*.jsonl")):
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


def require_intact(u):
    bad = torn_record_files(u)
    if bad:
        refuse("incomplete or invalid record file(s): " + ", ".join(str(p) for p in bad)
               + ". Run `run_v5.py repair` on each (logged; the problem is then rerun). Records are never edited by hand.")


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
    shards = {n: O.sha256_file(snap / n) == s["sha256"] if (snap / n).exists() else False for n, s in e["shards"].items()}
    excluded = [f for f in e["excluded_files"] if (snap / f).exists()]
    return {"snapshot": snap.name, "revision_pinned": snap.name == e["revision"], "shards_ok": shards, "excluded_present": excluded,
            "pass": bool(snap.name == e["revision"] and all(shards.values()) and not excluded)}


def check_parquet():
    """GW: the pinned parquet files' SHA-256 equal the manifest (HF-reported LFS hashes at the pinned revision)."""
    d = rv5.dataset_entry()
    res = {}
    for s, f in d["files"].items():
        try:
            res[s] = O.sha256_file(rv5.parquet_path(s, local_only=True)) == f["sha256"]
        except Exception:  # noqa: BLE001
            res[s] = False
    return {"files_ok": res, "pass": all(res.values())}


def gw(task, model_id):
    """GW + G5 (WikiText): every sample paragraph rebuilt from the parquet equals its hashes; the model's prompt ids and 65-token history equal
    the sample's hashes."""
    rep = {"parquet": check_parquet()}
    s = task.s
    tok = rv5.load_tokenizer(model_id)
    bad = []
    n = 0
    for part, split in (("test", "test"), ("calibration", "train"), ("timing", "train")):
        for it in s[part]:
            n += 1
            text, prompt, rest = task.data.problem(split, it["index"])
            want = it["tok"][model_id]
            h = rv5.history_ids(tok, rest)
            p = rv5.prompt_token_ids(tok, prompt)
            if not (rv5.sha_text(text) == it["text_sha256"] and rv5.sha_text(prompt) == it["prompt_sha256"] and len(h) == rv5.WIKI_HIST
                    and rv5.ids_sha(h) == want["history_ids_sha256"] and rv5.ids_sha(p) == want["prompt_ids_sha256"]):
                bad.append(f"{part}/{it['index']}")
    rep["G5_wikitext"] = {"n_items": n, "failures": bad[:20], "pass": not bad and n == len(s["test"]) + len(s["calibration"]) + len(s["timing"])}
    rep["pass"] = bool(rep["parquet"]["pass"] and rep["G5_wikitext"]["pass"])
    return rep


def g5(tok, task, e):
    """GSM8K units: prompt SHA-256 against v4's sample, round trip, BOS as the manifest records for the pinned tokenizer, EOS, drift marker."""
    s = task.s
    rep = {"prompt_sha_ok": [], "roundtrip_ok": [], "bos_first": []}
    for idx in s["gate_G5_test_ids"]:
        p = task.prompt("test", idx)
        rep["prompt_sha_ok"].append(hashlib.sha256(p.encode()).hexdigest() == s["prompt_sha256"]["test"][str(idx)])
        ids = tok(p).input_ids
        rep["roundtrip_ok"].append(tok.decode(ids, skip_special_tokens=True) == p)
        rep["bos_first"].append(tok.bos_token_id is not None and ids[0] == tok.bos_token_id)
    want_bos = bool(e["expects_bos"])
    drift = tok("The answer is 7.\n#### 7\n\nQuestion: Next", add_special_tokens=False).input_ids
    plain = tok("The answer is 7.\n#### 7", add_special_tokens=False).input_ids
    rep.update({"expected_bos": want_bos, "eos_token_id": tok.eos_token_id, "bos_token_id": tok.bos_token_id,
                "drift_marker_detected": bool(would_stop_at_boundary(tok, drift)), "no_false_drift": not would_stop_at_boundary(tok, plain)})
    rep["pass"] = bool(all(rep["prompt_sha_ok"]) and all(rep["roundtrip_ok"]) and all(b == want_bos for b in rep["bos_first"])
                       and tok.eos_token_id is not None and rep["drift_marker_detected"] and rep["no_false_drift"])
    return rep


def gh(u, slot, prec):
    """GH (section 5): the installed hqq version; two quantizations bitwise equal (else the first state is saved and frozen); the independent
    dequantization oracle equals HQQLinear.dequantize() bitwise; the locked configuration. Leaves the unit's backend in slot."""
    import rv5_hqq as H
    m = UNITS[u][0]
    t0 = time.perf_counter()
    slot.load(rv5.load_model, m, DEVICE, prec, "hqq", True)
    t_quant = time.perf_counter() - t0
    h1 = H.state_hash(slot.be.model)
    tmp = mdir(u) / "hqq_state_first.pt.tmp"
    H.save_state(slot.be.model, tmp)
    oracle = rv5.hqq_oracle_ok(slot.be.model)
    cfg_ok = rv5.hqq_config_ok(slot.be.model)
    bpw = H.bits_per_weight(slot.be.model)
    slot.load(rv5.load_model, m, DEVICE, prec, "hqq", True)
    h2 = H.state_hash(slot.be.model)
    det = h1 == h2
    rec = {"unit": u, "precision": prec, "hqq_version": rv5.hqq_version(), "config": H.HQQ_CONFIG, "deterministic": det, "state_sha256": h1,
           "second_state_sha256": h2, "frozen": not det, "bits_per_weight": bpw, "quantize_seconds": t_quant, "utc": O.utc(time.time())}
    if det:
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, hqq_frozen_path(u))
        rec["frozen_file_sha256"] = O.sha256_file(hqq_frozen_path(u))
        slot.load(rv5.load_model, m, DEVICE, prec, "hqq", True, str(hqq_frozen_path(u)))
        if H.state_hash(slot.be.model) != h1:
            rec["frozen_reload_ok"] = False
    rec["pass"] = bool(rec["hqq_version"] == H.HQQ_VERSION and oracle["pass"] and cfg_ok and rec.get("frozen_reload_ok", True))
    rec["oracle"] = oracle
    (mdir(u) / "hqq.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def g3(be, tok, task, u, dtype):
    allrecs, errs = [], []
    for idx in task.gate_numerics_ids():
        recs, _ = run_problem(be, tok, task, u, idx, "train", TEST_ARMS, rv5.tiers(u)["gates"], dtype)
        allrecs.append(recs)
        for t, r in recs.items():
            if not (r["finite_ok"] and r["ref_logits_finite"] and all(x == x for x in r["js"]) and r["js"] and r["js"][0] == 0.0):
                errs.append(f"{idx}/{t}")
    return not errs, errs, allrecs


def gates(u):
    m, q, tname, _ = UNITS[u]
    e = rv5.models()[m]
    task = Task(u)
    rep = {"started_utc": O.utc(time.time()), "unit": rv5.unit(u), "hf_id": e["hf_id"], "revision": e["revision"]}
    snap = snapshot_dir(e)
    rep["G7_snapshot"] = check_snapshot(e, snap)
    if not rep["G7_snapshot"]["pass"]:
        return finish_gates(u, rep, False)
    if task.wiki:
        rep["GW"] = gw(task, m)
        if not rep["GW"]["pass"]:
            return finish_gates(u, rep, False)
    prec = precision(u)
    slot = ModelSlot()

    def load_for(p):
        if q == "hqq":
            rep[f"GH_{p}"] = gh(u, slot, p)
            return rep[f"GH_{p}"]["pass"]
        slot.load(load_unit, u, p, False)
        return True
    if not load_for(prec):
        return finish_gates(u, rep, False)
    snapc = rv4.config_snapshot(slot.be.model.config)
    att = rv5.attention_snapshot(slot.be.model)
    rep["G7_config"] = {**snapc, "attention": att, "manifest": {k: e["geometry"][k] for k in rv4.GEOMETRY_KEYS + ("qk_norm",)},
                        "pass": bool(rv4.geometry_matches(snapc, e["geometry"]) and rv5.attention_matches(att, e["geometry"]))}
    rep["G5_format"] = {"pass": True, "see": "GW.G5_wikitext"} if task.wiki else g5(slot.tok, task, e)
    ok3, errs3, recs3 = g3(slot.be, slot.tok, task, u, rv4.DTYPES[prec])
    rep["G3_finite"] = {"precision": prec, "pass": ok3, "failures": errs3}
    if not ok3 and prec == "fp16":
        (mdir(u) / "precision.json").write_text(json.dumps({"precision": "bf16", "rule": "D6 (v4 section 5, unchanged): G3 failed in FP16",
                                                            "fp16_failures": errs3, "utc": O.utc(time.time())}, indent=1), encoding="utf-8")
        prec = "bf16"
        if not load_for(prec):
            return finish_gates(u, rep, False)
        ok3, errs3, recs3 = g3(slot.be, slot.tok, task, u, rv4.DTYPES[prec])
        rep["G3_finite_bf16"] = {"pass": ok3, "failures": errs3}
        if not ok3:
            (mdir(u) / "not_run.json").write_text(json.dumps({"reason": "G3 failed in FP16 and in BF16", "utc": O.utc(time.time())}), encoding="utf-8")
            return finish_gates(u, rep, False)
    if not ok3:
        return finish_gates(u, rep, False)
    dtype = rv4.DTYPES[prec]
    be, tok = slot.be, slot.tok
    rep["precision"] = prec
    rep["G2_integrity_errors"] = [x for recs in recs3 for x in integrity_errors(recs)]
    tier = rv5.tiers(u)["gates"]
    g1 = []
    for idx in task.gate_alpha0_ids():
        ids_t = task.prompt_ids(tok, "train", idx)
        ref = task.history(be, tok, ids_t, "train", idx)
        S_i = min(S_STEPS, len(ref) - 1)
        rl = rv4.reference_logits(be, ids_t, ref, S_i)
        cF = V.fp16_teacher_forced_cache(be, ids_t, ref, S_i)
        for arm in rv4.CORE_ARMS:
            r, c0 = rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, arm, 0.0, rv5.arm_seeds(u, arm, 0.0, idx, tier), rl, dtype, return_cache=True)
            same = all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(c0.layers, cF.layers))
            g1.append({"index": idx, "arm": arm, "ok": bool(same and all(v == 0.0 for v in r["js"]) and r["routing_reset"])})
    rep["G1_alpha0_bitwise"] = {"n": len(g1), "all_identical": all(x["ok"] for x in g1), "failures": [x for x in g1 if not x["ok"]]}
    g4 = []
    for idx in task.gate_numerics_ids():
        ids_t = task.prompt_ids(tok, "train", idx)
        ref = task.history(be, tok, ids_t, "train", idx)
        S_i = min(S_STEPS, len(ref) - 1)
        rl = rv4.reference_logits(be, ids_t, ref, S_i)
        run = lambda lab: rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, "C1", 2.0, rv5.arm_seeds(u, "C1", 2.0, idx, lab), rl, dtype)["js"]  # noqa: E731
        a1, a2, a3 = run(tier), run(tier), run(tier + "-other")
        g4.append((a1 == a2, a1 != a3))
    rep["G4_seeds"] = {"same_label_identical": all(x for x, _ in g4), "different_label_differs_in_n_of": [sum(y for _, y in g4), len(g4)]}
    geo = rv4.cache_geometry(be, task.prompt_ids(tok, "train", task.gate_alpha0_ids()[0]), dtype)
    g = e["geometry"]
    rep["G7_cache"] = {**geo, "n_dynamic_modules": be.n_dynamic,
                       "pass": bool((geo["n_layers"], geo["n_kv_heads"], geo["head_dim"]) == (g["layers"], g["kv_heads"], g["head_dim"])
                                    and geo["all_mode_dtype"] and be.n_dynamic == 7 * g["layers"])}
    passed = bool(rep["G7_config"]["pass"] and rep["G5_format"]["pass"] and rep["G7_cache"]["pass"] and not rep["G2_integrity_errors"]
                  and rep["G1_alpha0_bitwise"]["all_identical"] and rep["G4_seeds"]["same_label_identical"]
                  and rep["G4_seeds"]["different_label_differs_in_n_of"][0] == len(g4))
    run_manifest(u)
    return finish_gates(u, rep, passed)


def finish_gates(u, rep, passed):
    rep["all_passed"] = bool(passed)
    rep["finished_utc"] = O.utc(time.time())
    (mdir(u) / "gates").mkdir(parents=True, exist_ok=True)
    (mdir(u) / "gates" / "gates.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("G1_alpha0_bitwise",)}, indent=1, default=float)[:6000])
    print("ALL PASSED:", rep["all_passed"], flush=True)
    return passed


# ----------------------------------------------------------------------------- calibration
def aux_path(u):
    return mdir(u) / "calib" / "aux_state.pt"


def merge_aux(u, idx, st):
    p = aux_path(u)
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


def calib_done(u, idx):
    d = mdir(u) / "calib"
    p = aux_path(u)
    inc = set(torch.load(p)["included"]) if p.exists() else set()
    return idx in inc and all(idx in done_ids(d, t) for _, _, t in rv4.CAL_ARMS)


def calib_step(be, tok, task, u, idx, dtype):
    d = mdir(u) / "calib"
    acc = rv4.RhoAccumulator()
    recs, _ = run_problem(be, tok, task, u, idx, "train", rv4.CAL_ARMS, rv5.tiers(u)["calibration"], dtype, acc=acc)
    errs = integrity_errors(recs)
    if errs:
        (mdir(u) / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}), encoding="utf-8")
        print("[INTEGRITY FAILURE]", errs[:3], flush=True)
        return False
    merge_aux(u, idx, acc.state())
    append_new(d, idx, recs)
    return True


# ----------------------------------------------------------------------------- problem loop with the USD ceiling
def block_loop(name, u, ids, step_fn, done_fn, cost, expected_problem_s=None):
    import stage1_runtime as SR
    mdir(u).mkdir(parents=True, exist_ok=True)
    worst, n, tsum = 0.0, 0, 0.0
    for k, idx in enumerate(ids):
        if done_fn(idx):
            continue
        ok, est = cost.can_start(1.5 * worst + 90.0)
        if not ok:
            (mdir(u) / "ceiling_stop.json").write_text(json.dumps({"before_problem": idx, "estimate": est, "utc": O.utc(time.time())}), encoding="utf-8")
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
            first = mdir(u) / "drift_stop.json"                     # V5-D2: an earlier drift-stop record is never overwritten
            (first if not first.exists() else mdir(u) / "drift_stop_D2.json").write_text(json.dumps(
                {"mean_problem_s": tsum / n, "expected_problem_s": expected_problem_s, "n": n, "drift_ref_factor": DRIFT_REF_FACTOR_D2,
                 "utc": O.utc(time.time())}), encoding="utf-8")
            print(f"[DRIFT] {u}: {tsum / n:.1f} s per problem > {1 + DRIFT:.2f} x {expected_problem_s:.1f} s; stop and report.", flush=True)
            return False
        if n <= 3 or n % 10 == 0:
            print(f"[{name}] #{k} idx={idx} {dt:.1f}s | USD {est['used_usd']:.2f} | {tele}", flush=True)
    return True


# ----------------------------------------------------------------------------- test problems
def test_step(be, tok, task, u, idx, dtype, g8=False):
    d = mdir(u) / "test"
    tier = rv5.tiers(u)["test"]
    recs, ref = run_problem(be, tok, task, u, idx, "test", TEST_ARMS, tier, dtype, slots_for=("NAT",))
    errs = integrity_errors(recs)
    if g8 and not errs:
        recs2, ref2 = run_problem(be, tok, task, u, idx, "test", TEST_ARMS, tier, dtype, slots_for=("NAT",))
        same = list(ref) == list(ref2) and canonical(recs) == canonical(recs2)
        O.append_jsonl(mdir(u) / "g8.jsonl", {"index": idx, "bitwise_equal": same, "utc": O.utc(time.time())})
        if not same:
            (mdir(u) / "g8_failure.json").write_text(json.dumps({"index": idx}), encoding="utf-8")
            print(f"[G8 FAILURE] {u} problem {idx}: the two in-session runs differ; stop and report", flush=True)
            return False
    if errs:
        (mdir(u) / "integrity_failure.json").write_text(json.dumps({"index": idx, "errors": errs}), encoding="utf-8")
        print("[INTEGRITY FAILURE]", errs[:3], flush=True)
        return False
    write_slots(u, "test", recs)
    append_new(d, idx, recs)
    if idx not in done_ids(d, "FP16_history"):
        E.append_records(d, {"FP16_history": rv5.history_record(idx, ref, task.name)})
    return d3_reference_ok(u, idx)


def test_done(u, idx):
    d = mdir(u) / "test"
    return all(idx in done_ids(d, t) for _, _, t in TEST_ARMS) and slot_path(u, "test", "NAT", idx).exists()


def expected_problem_seconds(u):
    """Section 9: the drift reference is the design-point projection (it already includes the factor 1.48). U0 runs before the design: none.
    Deviation V5-D2: for U1's remaining problems and every later unit the reference is that projection x 1.38 (tolerance still 25 %)."""
    e = ((design() or {}).get("expected_problem_s") or {}).get(u)
    return e * DRIFT_REF_FACTOR_D2 if e else e


def stored_canonical(u, idx, base=None):
    d = base or (mdir(u) / "test")
    out = {}
    for _, _, t in TEST_ARMS:
        rows = [json.loads(l) for l in (d / f"{t}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        r = next(x for x in rows if x["index"] == idx)
        out[t] = json.dumps({k: v for k, v in r.items() if k not in IGNORED}, sort_keys=True)
    z = np.load(d / "slots" / f"NAT_{idx}.npz")
    out["NAT/slots"] = hashlib.sha256(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()
    hist = next(json.loads(l) for l in (d / "FP16_history.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["index"] == idx)
    return out, hist


def d3_reference_ids(u):
    rec = jload(mdir(u) / D3_REF / "d3_reference.json")
    return [int(i) for i in rec["ids"]] if rec else []


def d3_reference_ok(u, idx):
    """V5-D3: each re-executed problem that survived in the first U1 checkpoint must equal it bitwise, in the resume check's sense (every arm's
    record without wall_seconds, the NAT slots' bytes, the history record). A difference stops the unit (d3_reference_failure.json)."""
    if u != D3_UNIT or idx not in d3_reference_ids(u):
        return True
    new = stored_canonical(u, idx)
    old = stored_canonical(u, idx, base=mdir(u) / D3_REF / "test")
    same = new == old
    O.append_jsonl(mdir(u) / "d3_reference_check.jsonl", {"index": idx, "bitwise_equal": same, "utc": O.utc(time.time())})
    print(f"[D3 REFERENCE] {u} problem {idx}: " + ("bitwise equal to the surviving first-run records" if same
                                                   else "DIFFERS from the surviving first-run records"), flush=True)
    if not same:
        (mdir(u) / "d3_reference_failure.json").write_text(json.dumps({"index": idx, "utc": O.utc(time.time())}), encoding="utf-8")
    return same


def d3_reference_complete(u):
    ids = d3_reference_ids(u)
    if u != D3_UNIT or not ids:
        return True
    p = mdir(u) / "d3_reference_check.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    ok = {r["index"] for r in rows if r["bitwise_equal"]}
    return set(ids) <= ok and all(r["bitwise_equal"] for r in rows)


def d3_calibration_check(u):
    """V5-D3: the locked analyze_v5.calibrate, unchanged, on a copy of the re-executed calibration records in a scratch runs directory; the
    result's LF hash must equal the CALIBRATION line already in the lock. Writes U/d3_calibration_check.json (and a stop file if it differs)."""
    import shutil
    import analyze_v5 as AV
    scratch = OUT / "_d3_calibration"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / u).mkdir(parents=True)
    shutil.copytree(mdir(u) / "calib", scratch / u / "calib")
    saved = AV.OUT
    AV.OUT = scratch
    try:
        AV.calibrate(u)
    finally:
        AV.OUT = saved
    new = lf_sha((scratch / u / "calibration.json").read_bytes())
    line = next((l for l in lock_text().splitlines() if l.startswith(f"CALIBRATION {u}:")), "")
    locked = line.split(": ", 1)[1].split()[0] if line else None
    rec = {"deviation": "V5-D3", "unit": u, "recomputed_sha256": new, "locked_sha256": locked, "equal": bool(locked) and new == locked,
           "utc": O.utc(time.time())}
    (mdir(u) / "d3_calibration_check.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[D3 CALIBRATION] {u}: recomputed {new[:16]} locked {str(locked)[:16]} " + ("EQUAL" if rec["equal"] else "DIFFERENT: stop and report"),
          flush=True)
    if not rec["equal"]:
        (mdir(u) / "d3_calibration_mismatch.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec["equal"]


def resume_check(be, tok, task, u, ids, dtype):
    """V4-D1 rule, unchanged: before a unit's test resumes, rerun its last completed problem (no record written) and require it bitwise equal to
    the stored records, history and NAT slots."""
    done = [i for i in ids if test_done(u, i)]
    if not done:
        return True
    idx = done[-1]
    recs, ref = run_problem(be, tok, task, u, idx, "test", TEST_ARMS, rv5.tiers(u)["test"], dtype, slots_for=("NAT",))
    stored, hist = stored_canonical(u, idx)
    same = canonical(recs) == stored and rv5.history_matches(hist, ref, task.name)
    O.append_jsonl(mdir(u) / "resume_check.jsonl", {"index": idx, "bitwise_equal": same, "n_done": len(done), "utc": O.utc(time.time())})
    print(f"[RESUME CHECK] {u} problem {idx}: {'bitwise equal to the stored records' if same else 'DIFFERS from the stored records'}", flush=True)
    if not same:
        (mdir(u) / "resume_check_failure.json").write_text(json.dumps({"index": idx, "utc": O.utc(time.time())}), encoding="utf-8")
    return same


def run_test(u, cost):
    task = Task(u)
    ids = task.test_ids(unit_n(u))
    be, tok = load_unit(u)
    dtype = rv4.DTYPES[precision(u)]
    started = any(done_ids(mdir(u) / "test", t) for _, _, t in TEST_ARMS)
    pending = [i for i in ids if not test_done(u, i)]
    g8_ids = ids[:N_G8] if not started else pending[:1]
    if started and pending and not resume_check(be, tok, task, u, ids, dtype):
        return False
    for idx in g8_ids:
        if not test_done(u, idx) and not test_step(be, tok, task, u, idx, dtype, g8=True):
            return False
    ok = block_loop(f"test:{u}", u, ids, lambda i: test_step(be, tok, task, u, i, dtype), lambda i: test_done(u, i), cost, expected_problem_seconds(u))
    if ok and all(test_done(u, i) for i in ids) and not d3_reference_complete(u):
        (mdir(u) / "d3_reference_failure.json").write_text(json.dumps({"reason": "not every surviving first-run problem was checked equal",
                                                                       "utc": O.utc(time.time())}), encoding="utf-8")
        print(f"[D3 REFERENCE] {u}: not every surviving first-run problem was checked equal; stop and report", flush=True)
        return False
    if ok and all(test_done(u, i) for i in ids):
        (mdir(u) / "test_complete.json").write_text(json.dumps({"unit": u, "n": len(ids), "rehearsal": rehearsal(), "utc": O.utc(time.time())}),
                                                    encoding="utf-8")
    return ok


# ----------------------------------------------------------------------------- design point (section 9)
def probe(p):
    """3 timing problems (GSM8K, v4's G6 ids) with the six test arms; no record kept. HQQ: the load includes the quantization."""
    m, q = PROBES[p]
    task = Task("U0")
    t0 = time.perf_counter()
    be, tok = rv5.load_model(m, DEVICE, "fp16", q, local_only=True)
    t_load = time.perf_counter() - t0
    per_arm, per_hist, finite = [], [], True
    for idx in task.timing_ids():
        t = time.perf_counter()
        ids_t = task.prompt_ids(tok, "train", idx)
        ref = fp16_history(be, tok, ids_t)
        rl = rv4.reference_logits(be, ids_t, ref, min(S_STEPS, len(ref) - 1))
        per_hist.append(time.perf_counter() - t)
        S_i = min(S_STEPS, len(ref) - 1)
        arms_t = []
        for arm, alpha, _ in TEST_ARMS:
            t = time.perf_counter()
            r = rv4.teacher_forced_arm_v4(be, ids_t, ref, S_i, arm, alpha, rv5.probe_seeds(m, q, arm, alpha, idx), rl, torch.float16)
            arms_t.append(time.perf_counter() - t)
            finite = finite and r["finite_ok"]
        per_arm.append(float(np.mean(arms_t)))
    e = rv5.models()[m]
    return {"probe": p, "model": m, "quantizer": q, "t_arm_problem_s": float(np.mean(per_arm)), "t_hist_ref_problem_s": float(np.mean(per_hist)),
            "t_load_s": t_load, "finite": finite, "layers": e["geometry"]["layers"], "params": e["params"], "rehearsal": rehearsal(),
            "gpu": O.gpu_name(), "utc": O.utc(time.time())}


def nf4_bytes_per_param():
    return json.loads((RV / "v4" / "plan" / "v4_plan_numbers.json").read_text(encoding="utf-8"))["nf4_bytes_per_param"]


def weight_bytes(params):
    """Bytes read per decode pass (v4 G6 model): FP16 pass (all weights at 2 bytes) + NF4 pass (linear weights at the NF4 rate, the rest at 2)."""
    tot, lin = params["total"], params["linear"]
    return 2.0 * tot + nf4_bytes_per_param() * lin + 2.0 * (tot - lin)


def fit_runtime(meas, probe_):
    """v4's t_arm = a * layers + b * bytes, solved exactly from the two NF4 probes; a negative coefficient is set to 0 and the other refitted on
    the larger model (conservative)."""
    x1, x2 = (meas["layers"], weight_bytes(meas["params"])), (probe_["layers"], weight_bytes(probe_["params"]))
    y = np.array([meas["t_arm_problem_s"], probe_["t_arm_problem_s"]])
    a, b = np.linalg.solve(np.array([x1, x2], float), y)
    if a < 0:
        a, b = 0.0, y[1] / x2[1]
    if b < 0:
        a, b = y[1] / x2[0], 0.0
    hist = max(meas["t_hist_ref_problem_s"] / meas["t_arm_problem_s"], probe_["t_hist_ref_problem_s"] / probe_["t_arm_problem_s"])
    return float(a), float(b), float(hist)


def unit_timing(pr: dict, fit) -> dict:
    """Per unit (U1-U6): (t_arm, t_hist) per problem in seconds, times the factor 1.48; WikiText units times 64/61.1."""
    a, b, hist = fit
    out = {}
    for u in ORDER[1:]:
        m, q, task, _ = UNITS[u]
        if u in ("U1", "U2"):
            t_arm = a * rv5.models()[m]["geometry"]["layers"] + b * weight_bytes(rv5.models()[m]["params"])
            t_hist = hist * t_arm
        else:
            key = ("qwen" if m == "qwen2.5-0.5b" else "mistral") + "-" + q
            t_arm, t_hist = pr[key]["t_arm_problem_s"], pr[key]["t_hist_ref_problem_s"]
        f = FACTOR * (WIKI_FACTOR if task == "wikitext2" else 1.0)
        out[u] = (t_arm * f, t_hist * f)
    return out


def project(times: dict, n_test: dict, cut, rate, quant_s: dict) -> dict:
    """GPU h = (n + 3 G8) x (6 t_arm + t_hist) + 50 x (4 t_arm + t_hist) + gates (0.35 h, plus two HQQ quantizations); billed h = (GPU h x 1.222
    + 0.25 h per model not yet downloaded) x 1.05; USD = billed x rate."""
    rows, new_models = {}, {"qwen3-0.6b-base", "smollm2-1.7b"}
    for u in ORDER[1:]:
        if u in cut:
            continue
        t_arm, t_hist = times[u]
        n = n_test.get(u, N_TEST)
        m, q, _, _ = UNITS[u]
        gates_h = GATES_H + (2 * quant_s.get(m, 0.0) / 3600 if q == "hqq" else 0.0)
        gpu_h = ((n + N_G8) * (6 * t_arm + t_hist) + N_CAL * (4 * t_arm + t_hist)) / 3600 + gates_h
        dl = DOWNLOAD_H if m in new_models else 0.0
        new_models.discard(m)
        billed = (gpu_h * BILLED_PER_GPU_H + dl) * OVERHEAD
        rows[u] = {"n_test": n, "t_arm_s": t_arm, "t_hist_s": t_hist, "expected_problem_s": 6 * t_arm + t_hist, "gpu_h": gpu_h, "billed_h": billed,
                   "usd": billed * rate}
    return rows


def load_probes():
    pr = {}
    for p in PROBES:
        d = jload(OUT / "probes" / f"{p}.json")
        if d is None:
            sys.exit(f"design needs probes/{p}.json")
        if d.get("rehearsal_synthetic") and not rehearsal():
            sys.exit(f"probes/{p}.json is a rehearsal stand-in; it cannot enter a real design")
        if not d.get("finite"):
            sys.exit(f"probe {p} had a non-finite value")
        pr[p] = d
    return pr


def u0_arms_seconds():
    """Mean over U0's test problems of the summed wall time of the six arms (records' wall_seconds; history and reference not included)."""
    tot = {}
    for _, _, t in TEST_ARMS:
        p = mdir("U0") / "test" / f"{t}.jsonl"
        for l in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
            if l.strip():
                r = json.loads(l)
                tot[r["index"]] = tot.get(r["index"], 0.0) + r["wall_seconds"]
    return float(np.mean(list(tot.values()))) if tot else None


def make_design():
    if design_locked():
        sys.exit("the design is already locked; it is never revisited")
    cfg = O.load_cost_config(OUT)
    used = O.CostLedger(OUT, cfg).estimate()["used_usd"]
    pr = load_probes()
    fit = fit_runtime(pr["qwen-nf4"], pr["mistral-nf4"])
    times = unit_timing(pr, fit)
    quant_s = {"qwen2.5-0.5b": max(0.0, pr["qwen-hqq"]["t_load_s"] - pr["qwen-nf4"]["t_load_s"]),
               "mistral-7b-v0.3": max(0.0, pr["mistral-hqq"]["t_load_s"] - pr["mistral-nf4"]["t_load_s"])}
    steps = []
    for name, n_test, cut in REDUCTIONS:
        rows = project(times, n_test, cut, cfg["rate_usd_per_h"], quant_s)
        tot = sum(r["usd"] for r in rows.values())
        storage = O.storage_allowance(cfg["volume_gb"], sum(r["billed_h"] for r in rows.values()), cfg["storage_usd_per_gb_month"])
        budget = O.stop_usd() - used - storage
        steps.append({"reduction": name, "cut_units": list(cut), "projected_remaining_usd": tot, "storage_allowance_usd": storage, "budget_usd": budget,
                      "rows": rows})
        if tot <= budget:
            break
    last = steps[-1]
    ok = last["projected_remaining_usd"] <= last["budget_usd"]
    u0_arms = u0_arms_seconds()
    out = {"pass": ok, "reduction": last["reduction"] if ok else "R4", "cut_units": last["cut_units"] if ok else [u for u in ORDER[1:] if u not in rv5.NEVER_CUT],
           "n_test": {u: r["n_test"] for u, r in last["rows"].items()}, "n_cal": N_CAL, "factor": FACTOR, "wiki_factor": WIKI_FACTOR,
           "fit": {"host_s_per_layer": fit[0], "device_s_per_byte": fit[1], "hist_ratio": fit[2]},
           "hqq_ratio": {"qwen2.5-0.5b": pr["qwen-hqq"]["t_arm_problem_s"] / pr["qwen-nf4"]["t_arm_problem_s"],
                         "mistral-7b-v0.3": pr["mistral-hqq"]["t_arm_problem_s"] / pr["mistral-nf4"]["t_arm_problem_s"]},
           "hqq_quantize_s": quant_s, "used_usd_at_design": used, "projection_steps": steps,
           "expected_problem_s": {u: r["expected_problem_s"] for u, r in last["rows"].items()}, "per_unit": last["rows"],
           "u0_check": {"what": "reported only: U0's measured six-arm time per test problem against the qwen-nf4 probe's six-arm time x 1.48",
                        "probe_six_arms_s_x1.48": FACTOR * 6 * pr["qwen-nf4"]["t_arm_problem_s"], "u0_six_arms_s_mean": u0_arms},
           "rehearsal": rehearsal(), "utc": O.utc(time.time())}
    if not ok:
        out["stop"] = "R4: the projection exceeds the money left below the $14 stop even with R1-R3: stop and report (section 9)"
        (OUT / "design_failed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "projection_steps"}, indent=1))
        return False
    for u in out["cut_units"]:
        mdir(u).mkdir(parents=True, exist_ok=True)
        (mdir(u) / "not_run.json").write_text(json.dumps({"reason": f"cut by the locked design ({out['reduction']}, section 9)", "utc": O.utc(time.time())}),
                                              encoding="utf-8")
    (OUT / "design.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    mdir("DESIGN").mkdir(parents=True, exist_ok=True)
    print(json.dumps({k: v for k, v in out.items() if k != "projection_steps"}, indent=1))
    print("next: tools/lock_v5.py --append-design")
    return True


# ----------------------------------------------------------------------------- download and checkpoint
def download_model(m):
    from huggingface_hub import snapshot_download
    e = rv5.models()[m]
    p = Path(snapshot_download(e["hf_id"], revision=e["revision"], allow_patterns=e["allow_patterns"], token=False))
    rep = check_snapshot(e, p)
    return rep


def download(tag):
    if tag in UNITS:
        names = [UNITS[tag][0]] + (["wikitext2"] if UNITS[tag][2] == "wikitext2" else [])
    else:
        names = ["mistral-7b-v0.3"]                                    # DESIGN: the Mistral probes
    reps = {}
    for nme in names:
        if downloaded(nme):
            reps[nme] = {"pass": True, "skipped": "already downloaded and verified"}
            continue
        if nme == "wikitext2":
            for s in ("test", "train"):
                rv5.parquet_path(s, local_only=False)
            rep = check_parquet()
        else:
            rep = download_model(nme)
        tp = O.token_problems()
        rep["token_problems_after"] = tp
        rep["pass"] = bool(rep["pass"] and not tp)
        if rep["pass"]:
            (OUT / "_downloads").mkdir(parents=True, exist_ok=True)
            (OUT / "_downloads" / f"{nme}.json").write_text(json.dumps({**rep, "utc": O.utc(time.time())}, indent=1), encoding="utf-8")
        reps[nme] = rep
    mdir(tag).mkdir(parents=True, exist_ok=True)
    ok = all(r["pass"] for r in reps.values())
    if ok:
        (mdir(tag) / "download.json").write_text(json.dumps({"downloads": reps, "utc": O.utc(time.time())}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(reps, indent=1, default=str))
    return ok


def checkpoint(tag):
    import v4_secret_scan as SS
    if tag == "DESIGN" and design_locked():
        (mdir("DESIGN") / "design_complete.json").write_text(json.dumps({"utc": O.utc(time.time())}), encoding="utf-8")
    extra = [LOCK] if LOCK.exists() else []
    extra += sorted((OUT / "_downloads").glob("*.json")) if (OUT / "_downloads").exists() else []
    tar = O.pack_checkpoint(OUT, tag, extra_files=extra)
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
    st = jload(OUT / "status.json") or {}
    for u in ORDER:
        d = mdir(u) / "test"
        n = len(done_ids(d, "NAT")) if d.exists() else 0
        m = UNITS[u][0]
        print(f"{u} {m:16s} {UNITS[u][1]:3s} {UNITS[u][2]:9s} downloaded {downloaded(m)!s:5} gates {gates_passed(u)!s:5} precision {precision(u):4s} "
              f"hqq_locked {hqq_locked(u)!s:5} calib_locked {calibration_locked(u)!s:5} test {n:3d}/{unit_n(u)} complete {complete(u)!s:5} "
              f"not_run {not_run(u)!s:5} checkpointed {st.get(u, {}).get('checkpointed', False)}")
    print("GM passed:", gm_passed(), "| anchor ok:", anchor_ok(), "| design locked:", design_locked(), "| rehearsal:", rehearsal())


def main():
    O.install_redaction()
    rv4.apply_determinism()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "status", "ledger", "billing", "download", "gm", "gates", "calibrate", "probe", "design", "test",
                                    "checkpoint", "repair", "closeout", "d3-calibration-check"])
    ap.add_argument("--tag", choices=TAGS)
    ap.add_argument("--probe", choices=list(PROBES))
    ap.add_argument("--authorized-gpu-run", action="store_true")
    ap.add_argument("--usd", type=float)
    ap.add_argument("--utc")
    ap.add_argument("--file")
    ap.add_argument("--reason")
    a = ap.parse_args()
    if a.cmd == "plan":
        s = make_wiki_sample()
        print(json.dumps(s["counts"]), len(s["test"]), len(s["calibration"]), len(s["timing"]))
        return
    OUT.mkdir(parents=True, exist_ok=True)
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
    if a.cmd == "design":
        sys.exit(0 if make_design() else 2)
    if a.cmd == "closeout":
        # V5-D3: the pod must be up; a fresh manifest of its checkpoints is taken first (tools/v5_remote.py manifest, any authorized tag)
        mr = subprocess.run([sys.executable, str(RV / "tools" / "v5_remote.py"), "--tag", a.tag or "U0", "manifest"], capture_output=True, **O.TEXT)
        if mr.returncode != 0:
            print(f"could not take the pod checkpoint manifest (the pod must be running): {(mr.stderr or mr.stdout).strip()[-300:]}")
        r = O.closeout_report(OUT, ["U0", "DESIGN"] + ORDER[1:])
        print(json.dumps(r, indent=1))
        print("every checkpoint is verified on the laptop: the corresponding author may delete the volume, then record volume_deleted_utc and a final billing line"
              if r["ready"] else "NOT READY: do not delete the volume")
        sys.exit(0 if r["ready"] else 1)
    if a.cmd == "gm":
        require(a, "GM", "gm")
        ok = ledgered("gm", lambda cost: subprocess.run([sys.executable, str(RV / "tools" / "v5_gpu_mutation_check.py"), "--authorized-gpu-run"]).returncode == 0)
        sys.exit(0 if ok else 1)
    if a.cmd == "probe":
        if not a.probe:
            sys.exit("--probe is required")
        _PROBE["name"] = a.probe
        require(a, "DESIGN", "probe")

        def run(cost):
            r = probe(a.probe)
            (OUT / "probes").mkdir(parents=True, exist_ok=True)
            (OUT / "probes" / f"{a.probe}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            print(json.dumps(r, indent=1))
            return r["finite"]
        sys.exit(0 if ledgered(f"probe:{a.probe}", run) else 1)
    if not a.tag:
        sys.exit("--tag is required")
    tag = a.tag
    if a.cmd == "repair":
        import v3_repair_torn_line as RP
        if not (a.file and a.reason):
            sys.exit("repair --tag U --file F --reason R")
        p = Path(a.file)
        kept, removed = RP.plan_repair(p.read_bytes())
        p.write_bytes(kept)
        O.append_jsonl(OUT / "repairs.jsonl", {"file": str(p), "removed_sha256": hashlib.sha256(removed).hexdigest(), "removed_bytes": len(removed),
                                               "reason": a.reason, "utc": O.utc(time.time())})
        print("repaired; add a DEVIATION_LOG_V5.md row; the problem is rerun by the next test/calibrate process")
        return
    if a.cmd == "checkpoint":
        checkpoint(tag)
        return
    if tag == "GM" or (tag == "DESIGN" and a.cmd != "download"):
        sys.exit(f"{a.cmd} does not take --tag {tag}")
    require(a, tag, a.cmd)
    if a.cmd == "download":
        sys.exit(0 if ledgered(f"download:{tag}", lambda cost: download(tag), gpu=False) else 1)
    require_intact(tag)
    u = tag
    mdir(u).mkdir(parents=True, exist_ok=True)
    if a.cmd == "gates":
        sys.exit(0 if ledgered(f"gates:{u}", lambda cost: gates(u)) else 1)
    if a.cmd == "calibrate":
        def run(cost):
            task = Task(u)
            be, tok = load_unit(u)
            dtype = rv4.DTYPES[precision(u)]
            return block_loop(f"calib:{u}", u, task.cal_ids(), lambda i: calib_step(be, tok, task, u, i, dtype), lambda i: calib_done(u, i), cost)
        sys.exit(0 if ledgered(f"calibrate:{u}", run) else 1)
    if a.cmd == "test":
        sys.exit(0 if ledgered(f"test:{u}", lambda cost: run_test(u, cost)) else 1)
    if a.cmd == "d3-calibration-check":
        if u != D3_UNIT or not d3_active():
            sys.exit(f"d3-calibration-check runs only for {D3_UNIT} with V5-D3 appended to the lock")
        sys.exit(0 if d3_calibration_check(u) else 3)


if __name__ == "__main__":
    main()
