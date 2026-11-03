"""Named pitch lines: label maps from GSR polylines, naming extracted lines, the flip table (no data)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from varlab_vision import line_seg as ls  # noqa: E402
from varlab_vision.calibration import ImageLine  # noqa: E402
from varlab_vision.pitch_model import DEFAULT_PITCH  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _line(p0, p1) -> ImageLine:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return ImageLine(p0=p0, p1=p1, abc=np.array([n[0], n[1], -n @ p0]), n_pixels=int(np.linalg.norm(d)))


def test_every_model_name_is_a_pitch_model_line_or_circle():
    lines = {ln.name for ln in DEFAULT_PITCH.lines()}
    for sn, name in ls.MODEL_NAME.items():
        assert sn in ls.CLASSES
        assert name in lines or name in ls.CIRCLES
    assert ls.N_CLASSES == 27 and not any(c.startswith("Goal") for c in ls.MODEL_NAME)


def test_rasterise_draws_each_polyline_with_its_class_and_lines_over_circles():
    lines = {"Side line top": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}],
             "Circle central": [{"x": 0.5 + 0.1 * np.cos(t), "y": 0.2 + 0.1 * np.sin(t)} for t in np.linspace(0, 2 * np.pi, 60)],
             "Not a marking": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}
    lab = ls.rasterise_lines(lines, 200, 100, thickness=3)
    top = ls.CLASSES.index("Side line top") + 1
    circ = ls.CLASSES.index("Circle central") + 1
    assert lab[20, 50] == top and lab[20, 100] == top  # where the circle touches the line, the line wins
    assert lab[30, 100] == circ  # bottom of the circle (an ellipse in normalised coordinates)
    assert set(np.unique(lab)) == {0, top, circ}


def test_name_lines_reads_the_label_map_along_each_line():
    lab = np.zeros((176, 320), np.uint8)  # half the frame resolution
    lab[49:52, :] = ls.CLASSES.index("Side line top") + 1
    lab[:, 159:162] = ls.CLASSES.index("Middle line") + 1
    lab[120:123, 20:120] = ls.CLASSES.index("Goal left crossbar") + 1  # names nothing
    lines = [_line((10, 100), (600, 100)), _line((320, 120), (320, 340)), _line((60, 242), (220, 242)),
             _line((400, 300), (600, 330))]
    names = ls.name_lines(lines, lab, frame_size=(640, 352))
    assert names == {0: "touchline_far", 1: "halfway"}


def test_two_lines_with_one_name_keep_it_only_when_they_are_collinear():
    lab = np.zeros((352, 640), np.uint8)
    k = ls.CLASSES.index("Side line top") + 1
    lab[99:102, :] = k
    lab[199:202, 300:600] = k  # the network also paints a second, parallel stroke with the same name
    collinear = [_line((10, 100), (250, 100)), _line((300, 100), (600, 100))]
    assert ls.name_lines(collinear, lab, frame_size=(640, 352)) == {0: "touchline_far", 1: "touchline_far"}
    assert ls.name_lines(collinear[:1] + [_line((300, 200), (600, 200))], lab, frame_size=(640, 352)) == {}


def test_the_flip_table_mirrors_left_and_right_and_is_its_own_inverse():
    spec = importlib.util.spec_from_file_location("line_seg_worker", ROOT / "colab" / "line_seg_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    table = worker._flip_table()
    idx = {c: k + 1 for k, c in enumerate(ls.CLASSES)}
    assert table[0] == 0
    assert table[idx["Big rect. left main"]] == idx["Big rect. right main"]
    assert table[idx["Goal left post left"]] == idx["Goal right post right"]
    assert table[idx["Side line top"]] == idx["Side line top"]
    assert table[idx["Side line left"]] == idx["Side line right"]
    assert all(table[table[k]] == k for k in range(len(table)))


def test_the_network_keeps_the_input_size():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    model = ls.build_model(pretrained=False).eval()
    with torch.inference_mode():
        y = model(torch.from_numpy(ls.to_input(np.zeros((1080, 1920, 3), np.uint8)))[None])
    assert tuple(y.shape) == (1, ls.N_CLASSES, ls.INPUT_SIZE[1], ls.INPUT_SIZE[0])
