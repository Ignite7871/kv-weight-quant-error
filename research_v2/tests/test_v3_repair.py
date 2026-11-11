"""Tests of tools/v3_repair_torn_line.py (amendment 1). No model; temporary folders.
  venv/Scripts/python.exe research_v2/tests/test_v3_repair.py"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v3_repair_torn_line as RP  # noqa: E402

GOOD = [json.dumps({"index": i, "js": [0.0, 0.1 * i]}) for i in range(3)]
TESTS = []


def t(fn):
    TESTS.append(fn)
    return fn


def setup(content: bytes):
    root = Path(tempfile.mkdtemp())
    p = root / "qwen-base" / "tierA" / "NAT.jsonl"
    p.parent.mkdir(parents=True)
    p.write_bytes(content)
    dev = root / "dev.md"
    dev.write_text("", encoding="utf-8")
    return root, p, dev


def run(root, p, dev):
    return RP.main([str(p), "--reason", "test", "--root", str(root), "--deviation-log", str(dev)])


@t
def r01_torn_without_newline_is_removed_and_logged():
    good = ("\n".join(GOOD) + "\n").encode()
    root, p, dev = setup(good + b'{"index": 3, "js": [0.0, 0.')
    assert run(root, p, dev) == 0
    assert p.read_bytes() == good, "kept part must be the complete lines, byte-identical"
    log = [json.loads(x) for x in (root / "repair_log.jsonl").read_text().splitlines()]
    assert len(log) == 1 and log[0]["bytes_removed"] == len(b'{"index": 3, "js": [0.0, 0.')
    assert (root / log[0]["removed_bytes_saved_to"]).read_bytes() == b'{"index": 3, "js": [0.0, 0.', "removed bytes must be kept"
    assert "tools/v3_repair_torn_line.py" in dev.read_text(encoding="utf-8"), "deviation must be recorded"


@t
def r02_invalid_last_line_with_newline_is_removed():
    good = ("\n".join(GOOD) + "\n").encode()
    root, p, dev = setup(good + b'{"index": 3, "js": [0.0\n')
    assert run(root, p, dev) == 0 and p.read_bytes() == good


@t
def r03_intact_file_is_refused_and_untouched():
    b = ("\n".join(GOOD) + "\n").encode()
    root, p, dev = setup(b)
    assert run(root, p, dev) == 2 and p.read_bytes() == b and not (root / "repair_log.jsonl").exists() and dev.read_text() == ""


@t
def r04_invalid_middle_line_is_refused():
    b = (GOOD[0] + "\n" + '{"index": 1, "js": [0.0' + "\n" + GOOD[2] + "\n" + '{"inde').encode()
    root, p, dev = setup(b)
    assert run(root, p, dev) == 2 and p.read_bytes() == b, "corruption before the last line is not a torn write: stop and report"


@t
def r05_outside_root_is_refused():
    root, p, dev = setup(b'{"index": 0}\n{"ind')
    other = Path(tempfile.mkdtemp()) / "x.jsonl"
    other.write_bytes(b'{"index": 0}\n{"ind')
    assert RP.main([str(other), "--reason", "t", "--root", str(root), "--deviation-log", str(dev)]) == 2 and other.read_bytes() == b'{"index": 0}\n{"ind'


@t
def r06_single_torn_line_file():
    root, p, dev = setup(b'{"index": 0, "js"')
    assert run(root, p, dev) == 0 and p.read_bytes() == b""


@t
def r07_repair_removes_only_the_trailing_line():
    good = ("\n".join(GOOD) + "\n").encode()
    for tail in (b"{", b'{"index": 9', b"\xff\xfe", b'{"index": 9, "js": [1]}'):
        root, p, dev = setup(good + tail)
        assert run(root, p, dev) == 0 and p.read_bytes() == good, tail


def run_all():
    fails = []
    for fn in TESTS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails.append((fn.__name__, f"{type(e).__name__}: {e}"))
    return fails


if __name__ == "__main__":
    f = run_all()
    for fn in TESTS:
        print(("FAIL " if fn.__name__ in dict(f) else "PASS ") + fn.__name__)
    for x in f:
        print(*x)
    print(f"{len(TESTS) - len(f)}/{len(TESTS)} passed")
    sys.exit(1 if f else 0)
