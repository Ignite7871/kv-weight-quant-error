"""research_v2 instrumentation: policy-pluggable dual-precision generation with full tracing.

Imports the historical project code (src/, scripts/) read-only; nothing under src/ or scripts/ is
modified. See research_v2/PILOT_SPEC.md for the definitions this package implements.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
