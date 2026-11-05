"""Jersey numbers: label encoding, combining the heads, the per-track decision (no data, no torch)."""
from __future__ import annotations

import numpy as np
import pytest

from varlab_vision import jersey as js


def _log_softmax(x):
    x = np.asarray(x, float)
    return x - np.log(np.exp(x - x.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) - x.max(axis=1, keepdims=True)


@pytest.mark.parametrize("label,code", [("7", (1, 7, -1)), ("10", (2, 1, 0)), ("99", (2, 9, 9)), ("0", (1, 0, -1)),
                                        (None, (0, -1, -1)), ("", (0, -1, -1)), ("100", (0, -1, -1)), ("A", (0, -1, -1))])
def test_encode(label, code):
    assert js.encode(label) == code


def test_number_log_probs_is_a_distribution_over_none_and_0_to_99():
    rng = np.random.default_rng(0)
    ln, d1, d2 = (_log_softmax(rng.normal(size=(5, k))) for k in (3, 10, 10))
    lp = js.number_log_probs(ln, d1, d2)
    # two-digit numbers with a leading zero are excluded, so the mass is a little under one
    total = np.exp(lp).sum(axis=1)
    lost = np.exp(ln[:, 2] + d1[:, 0])  # P(two digits, tens digit 0)
    assert np.allclose(total + lost, 1.0)
    assert np.allclose(lp[:, 1 + 23], ln[:, 2] + d1[:, 2] + d2[:, 3])  # "23"
    assert np.allclose(lp[:, 1 + 7], ln[:, 1] + d1[:, 7])  # "7"


def _crop_lp(number, conf=0.9, n=1):
    lp = np.full((n, 101), np.log((1 - conf) / 100))
    lp[:, 0 if number is None else 1 + int(number)] = np.log(conf)
    return lp


def test_decide_sums_the_crops_and_needs_a_margin():
    seen = np.concatenate([_crop_lp("10", 0.6, 4), _crop_lp("16", 0.3, 2)])  # mostly 10, a couple of blurred reads
    assert js.decide(seen) == "10"
    tie = np.concatenate([_crop_lp("10", 0.5, 3), _crop_lp("16", 0.5, 3)])
    assert js.decide(tie) is None
    assert js.decide(_crop_lp("10", 0.9, 2)) is None  # too few crops
    assert js.decide(_crop_lp(None, 0.9, 6)) is None  # a referee: no number
