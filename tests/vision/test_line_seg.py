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


# ----------------------------------------------------------------------------
# Named lines in the calibration: a perfect label map drawn from the true camera
# ----------------------------------------------------------------------------

_CLASS_OF = {"touchline_far": "Side line top", "touchline_near": "Side line bottom", "halfway": "Middle line",
             "goal_line_left": "Side line left", "goal_line_right": "Side line right"}


def _class_for(poly_name: str, mid_x: float) -> str:
    base = poly_name.split("#")[0]
    if base in _CLASS_OF:
        return _CLASS_OF[base]
    side = "left" if mid_x < 0 else "right"
    rect = "Big rect." if base.startswith("penalty") else "Small rect."
    part = "main" if "front" in base else "top" if base.endswith("far") else "bottom"
    return f"{rect} {side} {part}"


def _perfect_labels(cam, size=ls.INPUT_SIZE):
    from varlab_vision.synthetic import project

    lab = np.zeros((size[1], size[0]), np.uint8)
    sx, sy = size[0] / cam.width, size[1] / cam.height
    for name, poly in DEFAULT_PITCH.polylines(arc_step_deg=4.0):
        if name.startswith(("centre_circle", "penalty_arc", "corner")):
            continue
        dense = np.concatenate([np.linspace(a, b, 50) for a, b in zip(poly[:-1], poly[1:])])
        px = project(cam.homography_m2px(), dense) * np.array([sx, sy])
        k = ls.CLASSES.index(_class_for(name, float(poly[:, 0].mean()))) + 1
        fp = np.round(px * 16).astype(np.int64).reshape(-1, 1, 2)
        cv2.polylines(lab, [fp], False, k, thickness=3, shift=4)
    return lab


def _scene():
    from varlab_vision.synthetic import BroadcastCamera, render_textured, textured_world

    cam = BroadcastCamera(width=640, height=360, hfov_deg=40.0, position=(0.0, 60.0, 22.0), look_at=(-30.0, 0.0))
    return cam, render_textured(cam, textured_world(seed=3))


def _max_error_m(cam, res):
    xs, ys = np.meshgrid(np.linspace(-52, 52, 27), np.linspace(-34, 34, 18))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    px, depth = cam.project(pts)
    vis = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < cam.width) & (px[:, 1] >= 0) & (px[:, 1] < cam.height)
    return float(np.linalg.norm(res.project(px[vis]) - pts[vis], axis=1).max())


def test_named_lines_give_the_true_homography_and_survive_one_wrong_name():
    from varlab_vision import calibration as cal

    cam, img = _scene()
    lines = cal.extract_lines(cal.marking_mask(img))
    names = ls.name_lines(lines, _perfect_labels(cam), (cam.width, cam.height))
    assert len(names) >= 4
    hs = cal.named_line_homographies(lines, names, 640, 360)
    assert hs and min(cal.homography_shift_px(h, cam.homography_m2px(), 640, 360) for h in hs) < 3.0  # a start: the lines sit around one box
    wrong = dict(names)
    i = next(i for i, nm in names.items() if nm == "penalty_front_left")
    wrong[i] = "halfway"  # one line misnamed (to a name no other line has): a leave-one-out fit is still right
    shifts = [cal.homography_shift_px(h, cam.homography_m2px(), 640, 360) for h in cal.named_line_homographies(lines, wrong, 640, 360)]
    assert min(shifts) < 4.0 and max(shifts) > 50.0


def test_calibrate_frame_tries_the_named_lines_before_the_search():
    from varlab_vision import calibration as cal

    cam, img = _scene()
    calls = []

    def labels():
        calls.append(1)
        return _perfect_labels(cam)

    res = cal.calibrate_frame(img, line_labels=labels)
    assert res.ok and "named lines" in res.note and calls == [1]
    assert _max_error_m(cam, res) < 0.15
    fallback = cal.calibrate_frame(img, line_labels=np.zeros((352, 640), np.uint8))  # names nothing: the search decides
    assert fallback.ok and "global search" in fallback.note
