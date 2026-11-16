"""CPU unit tests of v6/rv6.py (gates G4 and G5 of PROTOCOL_V6). python research_v2/v6/tests/test_rv6.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rv6  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
g = np.random.default_rng(0)


def test_codec():
    x = torch.from_numpy(g.standard_normal((10_000, 64)) * 3).half().double()
    a = rv6.codec_per_token(x).numpy()
    b = rv6.codec_per_token_np(x.numpy())
    assert np.array_equal(a, b), "G4: codec differs from the numpy implementation"
    c = torch.zeros(4, 64, dtype=torch.float64)
    assert torch.equal(rv6.codec_per_token(c), c), "constant group must pass through"
    xg = x.reshape(-1, 2, 32)
    mn = rv6._fp16(xg.amin(-1, keepdim=True))
    step = rv6._fp16((xg.amax(-1, keepdim=True) - mn) / rv6.QMAX)
    err = (rv6.codec_per_token(x).reshape(-1, 2, 32) - xg).abs()
    slack = (xg.amax(-1, keepdim=True) - mn - rv6.QMAX * step).abs() + 1e-12      # FP16 rounding of the scale at the group maximum
    assert bool((err <= step / 2 + slack).all()), "codec error exceeds half a step"


def test_inverse_rope():
    d = 64
    x = torch.randn(3, 2, d, dtype=torch.float64)
    th = torch.rand(d // 2, dtype=torch.float64) * 6
    cos = torch.cat([th.cos(), th.cos()])
    sin = torch.cat([th.sin(), th.sin()])
    rot = x * cos + rv6.rotate_half(x) * sin
    assert torch.allclose(rv6.inverse_rope(rot, cos, sin), x, atol=1e-12)


class FakeLayer:
    def __init__(self, K, Vv):
        self.keys, self.values = K[None], Vv[None]


def make(n_q=4, n_kv=2, d=64, L=2, P0=20, n=6, scale_k=0.0, scale_v=0.0, seed=1):
    gg = torch.Generator().manual_seed(seed)
    T = P0 + n + 1
    mi = SimpleNamespace(n_layers=L, n_q=n_q, n_kv=n_kv, hd=d, hidden=32, scaling=d ** -0.5,
                         hmap=torch.arange(n_q) // (n_q // n_kv),
                         G=torch.eye(d, dtype=torch.float64).expand(L, n_kv, d, d).clone(), trG=torch.full((L, n_kv), float(d), dtype=torch.float64))
    pv = torch.randn(L, n_kv, P0, d, generator=gg, dtype=torch.float64)
    log = rv6.Log6(mi, P0, 16, pv, "cpu")
    layers, qs = [], []
    for l in range(L):
        Ksh = torch.randn(n_kv, T, d, generator=gg, dtype=torch.float64)
        Vsh = torch.randn(n_kv, T, d, generator=gg, dtype=torch.float64)
        dK = scale_k * torch.randn(n_kv, n, d, generator=gg, dtype=torch.float64)
        dV = scale_v * torch.randn(n_kv, n, d, generator=gg, dtype=torch.float64)
        log.D["K"][l, :, :n] = dK
        log.D["V"][l, :, :n] = dV
        Kst, Vst = Ksh.clone(), Vsh.clone()
        Kst[:, P0:P0 + n] += dK
        Vst[:, P0:P0 + n] += dV
        layers.append(FakeLayer(Kst, Vst))
        qs.append(torch.randn(n_q, d, generator=gg, dtype=torch.float64) * 2)
    log.read_time(SimpleNamespace(layers=layers), n + 1, qs)
    return log, n


def test_value_identity():
    log, n = make(scale_v=0.3)
    s = n
    rel = ((log.read["lin_V"][s] - log.read["meas"][s]).abs() / log.read["meas"][s]).max()
    assert rel < 1e-10, f"G5: lin_V != meas (V only), rel {rel}"


def test_key_first_order():
    errs = []
    for sc in (1e-2, 1e-3, 1e-4):
        log, n = make(scale_k=sc)
        r = (log.read["lin_K"][n] / log.read["meas"][n]).flatten()
        errs.append(float((r - 1).abs().max()))
    assert errs[2] < errs[0] and errs[2] < 1e-3, f"G5: lin_K/meas does not converge to 1: {errs}"


def test_alignment_null():
    mi = SimpleNamespace(n_layers=1, n_q=2, n_kv=1, hd=64, hidden=8, scaling=0.125, hmap=torch.zeros(2, dtype=torch.long),
                         G=torch.diag(torch.linspace(0.1, 3, 64, dtype=torch.float64))[None, None], trG=None)
    mi.trG = torch.diagonal(mi.G, dim1=-2, dim2=-1).sum(-1)
    pv = torch.randn(1, 1, 50, 64, dtype=torch.float64)
    log = rv6.Log6(mi, 50, 2000, pv, "cpu")
    rv4 = rv6.rv4
    for s in range(1, 2001):
        dv = torch.randn(1, 1, 64, dtype=torch.float64)
        new = torch.zeros(2, 1, 64, dtype=torch.float64)
        new[1] = dv[0]
        kF = torch.zeros_like(new)
        log.write_time(s, new.half(), kF.half(), kF.half(), (torch.ones(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)), torch.zeros(1, 8))
    mo, mc = float(np.nanmean(log.A_o.numpy())), float(np.nanmean(log.A_cov.numpy()))
    assert abs(mo - 1) < 0.05 and abs(mc - 1) < 0.08, (mo, mc)


if __name__ == "__main__":
    for f in (test_codec, test_inverse_rope, test_value_identity, test_key_first_order, test_alignment_null):
        f()
        print("ok", f.__name__)
    print("ALL OK")
