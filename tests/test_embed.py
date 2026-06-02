from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import EMBED_DIM, pack, unpack


def test_pack_unpack_roundtrip():
    arr = np.arange(EMBED_DIM, dtype=np.float32)
    blob = pack(arr)
    assert len(blob) == EMBED_DIM * 4
    out = unpack(blob)
    assert out.shape == (EMBED_DIM,)
    # pack() normalises to unit length; check proportionality and unit norm
    assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.dot(out, arr / np.linalg.norm(arr))) == pytest.approx(1.0, abs=1e-5)


def test_pack_rejects_wrong_shape():
    with pytest.raises(ValueError):
        pack(np.zeros(10, dtype=np.float32))


def test_unpack_returns_independent_copy():
    arr = np.full(EMBED_DIM, 3.5, dtype=np.float32)
    blob = pack(arr)
    expected = float(unpack(blob)[0])  # normalized value
    a = unpack(blob)
    a[0] = 99.0
    b = unpack(blob)
    assert b[0] == pytest.approx(expected)
