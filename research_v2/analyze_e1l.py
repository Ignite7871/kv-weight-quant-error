"""Llama E1 Tier A analysis (E1L_SPEC.md): the locked Qwen E1 analysis code (analyze_controls.py, unchanged) applied to the Llama records. Tier A only; the
secondary Holm family therefore has FIVE tests (the Qwen family also contained two Tier B tests).
  venv/Scripts/python.exe research_v2/analyze_e1l.py calibrate | analyze"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_controls as A  # noqa: E402

OUT = HERE / "e1l_runs"


def main():
    cmd = sys.argv[1]
    if cmd == "calibrate":
        calib = {k: A.load_jsonl(OUT / "calib" / f"{k}.jsonl") for k in ("NAT", "C1_a0.5", "C1_a1", "C1_a2", "C1_a4")}
        res = A.calibrate(calib)
        (OUT / "e1l_calibration.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    calib = json.loads((OUT / "e1l_calibration.json").read_text())
    sample = json.loads((HERE / "e1l_sample.json").read_text())
    rep = A.analyze(A.tier_a_arms(OUT), {}, calib, set(sample.get("exposed_test_ids_in_sample", [])))
    (OUT / "e1l_report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "e1l_report.md").write_text(A.to_markdown(rep), encoding="utf-8")
    print(A.to_markdown(rep))


if __name__ == "__main__":
    main()
