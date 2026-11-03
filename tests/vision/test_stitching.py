"""Re-linking track fragments by place, time and appearance (constructed fragments, no data)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from varlab_vision.stitching import StitchConfig, Tracklet, mean_embeddings, stitch  # noqa: E402


def _frag(tid, f0, f1, start_xy, vel, emb=None, team=None, shot=0, height=60.0):
    f = np.arange(f0, f1 + 1)
    feet = np.asarray(start_xy, float) + np.outer(f - f0, vel)
    e = None if emb is None else np.asarray(emb, float) / np.linalg.norm(emb)
    return Tracklet(tid=tid, frames=f, feet=feet, heights=np.full(len(f), height), shot=shot, embedding=e, team=team)


def _continue(frag, tid, f0, f1, **kw):
    """The same player reappearing at frame ``f0`` where their motion would have taken them."""
    p, v, _h = frag.end_state()
    return _frag(tid, f0, f1, p + v * (f0 - frag.end), v, **kw)


def test_a_broken_track_is_joined_and_a_passer_by_is_not():
    red, blue = [1, 0, 0], [0, 1, 0]
    a = _frag(1, 0, 40, (100, 300), (2.0, 0.0), emb=red)
    a2 = _continue(a, 2, 55, 90, emb=red)
    b = _frag(3, 0, 90, (700, 300), (-6.0, 0.0), emb=blue)  # another player running the other way
    b_late = _frag(4, 52, 90, (a.feet[-1][0] + 25, 300), (2.0, 0.0), emb=blue)  # starts near the gap, looks different
    out = stitch([a, a2, b, b_late])
    assert out == {1: 1, 2: 1, 3: 3, 4: 4}


def test_chains_form_over_several_breaks():
    frags = [_frag(1, 0, 20, (0, 0), (3.0, 1.0), emb=[1, 1, 0])]
    for tid, (f0, f1) in zip((2, 3, 4), ((30, 50), (58, 80), (100, 120))):
        frags.append(_continue(frags[-1], tid, f0, f1, emb=[1, 1, 0]))
    assert set(stitch(frags).values()) == {1}


@pytest.mark.parametrize("change", ["team", "shot", "far", "overlap", "gap"])
def test_what_keeps_two_fragments_apart(change):
    a = _frag(1, 0, 40, (100, 300), (2.0, 0.0), emb=[1, 0, 0], team="left")
    kw = dict(emb=[1, 0, 0], team="left")
    if change == "team":
        kw["team"] = "right"
    if change == "shot":
        kw["shot"] = 1
    b = _continue(a, 2, 50, 80, **kw)
    if change == "far":
        b.feet = b.feet + np.array([400.0, 0.0])  # 6.7 body heights away after 10 frames
    if change == "overlap":
        b = _continue(a, 2, 30, 80, **kw)
    if change == "gap":
        f0 = 40 + StitchConfig().max_gap + 1
        b = _continue(a, 2, f0, f0 + 30, **kw)
    assert stitch([a, b]) == {1: 1, 2: 2}


def test_mean_embeddings_are_unit_length():
    m = mean_embeddings([1, 1, 2], np.array([[1.0, 0], [0, 1.0], [3.0, 4.0]]))
    assert np.allclose(m[1], [np.sqrt(0.5), np.sqrt(0.5)]) and np.allclose(m[2], [0.6, 0.8])
