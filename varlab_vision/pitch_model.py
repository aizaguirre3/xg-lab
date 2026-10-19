"""The pitch model: every marking of a football pitch in metres, plus the named
lines and points the calibration stage matches against.

Coordinate frame (the GSR convention from Figure S3 of arXiv:2404.11335, which
``gsr_format.py`` documents): origin at the centre mark, ``x`` toward the right
goal, ``y`` along the halfway line toward the main camera, so the near touchline
is ``y = +width / 2`` and the far touchline ``y = -width / 2``. Metres. The
default pitch is 105 x 68 m (FIFA recommended; the GSR pitch).

Mapping to the R engine (``R/09_offside_engine.R``): ``to_attacking_frame(x, y,
attacking_right, pitch_length = 105, pitch_width = 68, origin = "centre")``
takes these coordinates unchanged (it adds ``length / 2`` and ``width / 2`` to
move the origin to a corner, then mirrors when the team attacks left). StatsBomb
coordinates (120 x 80 yard-like units, corner origin) are a different frame;
scale by ``105 / 120`` and ``68 / 80`` after subtracting the half sizes if the
two must ever be compared.

Marking dimensions are the Laws of the Game constants: penalty area 16.5 m deep
and 40.32 m wide, goal area 5.5 m deep and 18.32 m wide, penalty mark 11 m from
the goal line, centre circle and penalty arcs of radius 9.15 m, corner arcs of
radius 1 m, goal 7.32 m wide.

The calibration stage uses two line *families*: ``x_const`` lines (``x = c``:
goal lines, halfway line, penalty- and goal-area fronts) and ``y_const`` lines
(``y = c``: touchlines, penalty- and goal-area sides). Each family's members are
ordered by their constant, which is the ordering constraint the hypothesis
search exploits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PENALTY_DEPTH_M = 16.5
PENALTY_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
PENALTY_MARK_M = 11.0
CIRCLE_RADIUS_M = 9.15
CORNER_RADIUS_M = 1.0
GOAL_WIDTH_M = 7.32

X_CONST = "x_const"
Y_CONST = "y_const"


@dataclass(frozen=True)
class ModelLine:
    """An infinite line of the pitch model with the segment(s) it is drawn as."""

    name: str
    kind: str  # X_CONST or Y_CONST
    value: float  # the constant: x for X_CONST, y for Y_CONST
    segments: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]

    @property
    def homogeneous(self) -> np.ndarray:
        """``(a, b, c)`` with ``a x + b y + c = 0`` in metres."""
        if self.kind == X_CONST:
            return np.array([1.0, 0.0, -self.value])
        return np.array([0.0, 1.0, -self.value])


@dataclass(frozen=True)
class ModelArc:
    name: str
    centre: Tuple[float, float]
    radius: float
    start_deg: float  # counter-clockwise from +x, in the pitch frame
    end_deg: float


@dataclass
class PitchModel:
    length: float = 105.0
    width: float = 68.0
    penalty_depth: float = PENALTY_DEPTH_M
    penalty_width: float = PENALTY_WIDTH_M
    goal_area_depth: float = GOAL_AREA_DEPTH_M
    goal_area_width: float = GOAL_AREA_WIDTH_M
    penalty_mark: float = PENALTY_MARK_M
    circle_radius: float = CIRCLE_RADIUS_M
    corner_radius: float = CORNER_RADIUS_M
    goal_width: float = GOAL_WIDTH_M
    _lines: List[ModelLine] = field(default_factory=list, init=False, repr=False)

    # -- geometry ------------------------------------------------------------
    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def lines(self) -> List[ModelLine]:
        """Every straight marking, as named infinite lines with their drawn segments."""
        if self._lines:
            return list(self._lines)
        L, W = self.half_length, self.half_width
        pd, pw = self.penalty_depth, self.penalty_width / 2.0
        gd, gw = self.goal_area_depth, self.goal_area_width / 2.0
        out: List[ModelLine] = []

        def add(name: str, kind: str, value: float, segs) -> None:
            out.append(ModelLine(name=name, kind=kind, value=value, segments=tuple(segs)))

        # y = const (touchlines and the box sides); the sides of both boxes at the
        # same |y| are one infinite line each, drawn as two segments.
        add("touchline_far", Y_CONST, -W, [((-L, -W), (L, -W))])
        add("touchline_near", Y_CONST, W, [((-L, W), (L, W))])
        add("penalty_side_far", Y_CONST, -pw, [((-L, -pw), (-L + pd, -pw)), ((L - pd, -pw), (L, -pw))])
        add("penalty_side_near", Y_CONST, pw, [((-L, pw), (-L + pd, pw)), ((L - pd, pw), (L, pw))])
        add("goal_area_side_far", Y_CONST, -gw, [((-L, -gw), (-L + gd, -gw)), ((L - gd, -gw), (L, -gw))])
        add("goal_area_side_near", Y_CONST, gw, [((-L, gw), (-L + gd, gw)), ((L - gd, gw), (L, gw))])
        # x = const (goal lines, halfway line, box fronts)
        add("goal_line_left", X_CONST, -L, [((-L, -W), (-L, W))])
        add("goal_area_front_left", X_CONST, -L + gd, [((-L + gd, -gw), (-L + gd, gw))])
        add("penalty_front_left", X_CONST, -L + pd, [((-L + pd, -pw), (-L + pd, pw))])
        add("halfway", X_CONST, 0.0, [((0.0, -W), (0.0, W))])
        add("penalty_front_right", X_CONST, L - pd, [((L - pd, -pw), (L - pd, pw))])
        add("goal_area_front_right", X_CONST, L - gd, [((L - gd, -gw), (L - gd, gw))])
        add("goal_line_right", X_CONST, L, [((L, -W), (L, W))])
        self._lines = out
        return list(out)

    def lines_by_kind(self, kind: str) -> List[ModelLine]:
        """Members of one family sorted by their constant (ascending)."""
        return sorted((ln for ln in self.lines() if ln.kind == kind), key=lambda ln: ln.value)

    def line(self, name: str) -> ModelLine:
        for ln in self.lines():
            if ln.name == name:
                return ln
        raise KeyError(name)

    def arcs(self) -> List[ModelArc]:
        L, W = self.half_length, self.half_width
        r = self.circle_radius
        # The penalty arc is the part of the circle around the penalty mark that
        # lies outside the penalty area: |angle| <= acos((depth - mark) / r).
        a = math.degrees(math.acos((self.penalty_depth - self.penalty_mark) / r))
        c = self.corner_radius
        return [
            ModelArc("centre_circle", (0.0, 0.0), r, 0.0, 360.0),
            ModelArc("penalty_arc_left", (-L + self.penalty_mark, 0.0), r, -a, a),
            ModelArc("penalty_arc_right", (L - self.penalty_mark, 0.0), r, 180.0 - a, 180.0 + a),
            ModelArc("corner_far_left", (-L, -W), c, 0.0, 90.0),
            ModelArc("corner_near_left", (-L, W), c, 270.0, 360.0),
            ModelArc("corner_far_right", (L, -W), c, 90.0, 180.0),
            ModelArc("corner_near_right", (L, W), c, 180.0, 270.0),
        ]

    def arc_points(self, arc: ModelArc, step_deg: float = 5.0) -> np.ndarray:
        n = max(2, int(round(abs(arc.end_deg - arc.start_deg) / step_deg)) + 1)
        t = np.radians(np.linspace(arc.start_deg, arc.end_deg, n))
        return np.stack([arc.centre[0] + arc.radius * np.cos(t), arc.centre[1] + arc.radius * np.sin(t)], axis=1)

    def spots(self) -> Dict[str, Tuple[float, float]]:
        L = self.half_length
        return {
            "centre_mark": (0.0, 0.0),
            "penalty_spot_left": (-L + self.penalty_mark, 0.0),
            "penalty_spot_right": (L - self.penalty_mark, 0.0),
        }

    def keypoints(self) -> Dict[str, Tuple[float, float]]:
        """Named intersections of the markings (the usual calibration keypoints)."""
        L, W = self.half_length, self.half_width
        pd, pw = self.penalty_depth, self.penalty_width / 2.0
        gd, gw = self.goal_area_depth, self.goal_area_width / 2.0
        r = self.circle_radius
        pts: Dict[str, Tuple[float, float]] = {
            "corner_far_left": (-L, -W), "corner_near_left": (-L, W),
            "corner_far_right": (L, -W), "corner_near_right": (L, W),
            "halfway_far": (0.0, -W), "halfway_near": (0.0, W),
            "centre_circle_far": (0.0, -r), "centre_circle_near": (0.0, r),
        }
        for side, sx in (("left", -1.0), ("right", 1.0)):
            gl = sx * L
            pf = sx * (L - pd)
            gf = sx * (L - gd)
            pts[f"penalty_{side}_front_far"] = (pf, -pw)
            pts[f"penalty_{side}_front_near"] = (pf, pw)
            pts[f"penalty_{side}_goal_line_far"] = (gl, -pw)
            pts[f"penalty_{side}_goal_line_near"] = (gl, pw)
            pts[f"goal_area_{side}_front_far"] = (gf, -gw)
            pts[f"goal_area_{side}_front_near"] = (gf, gw)
            pts[f"goal_area_{side}_goal_line_far"] = (gl, -gw)
            pts[f"goal_area_{side}_goal_line_near"] = (gl, gw)
            pts[f"goal_{side}_post_far"] = (gl, -self.goal_width / 2.0)
            pts[f"goal_{side}_post_near"] = (gl, self.goal_width / 2.0)
        pts.update(self.spots())
        return pts

    def keypoint_array(self, names: Optional[Sequence[str]] = None) -> Tuple[List[str], np.ndarray]:
        kp = self.keypoints()
        names = list(names) if names is not None else sorted(kp)
        return names, np.array([kp[n] for n in names], dtype=float)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: points inside the pitch rectangle (plus ``margin`` metres)."""
        p = np.asarray(xy, float).reshape(-1, 2)
        return (np.abs(p[:, 0]) <= self.half_length + margin) & (np.abs(p[:, 1]) <= self.half_width + margin)

    def polylines(self, arc_step_deg: float = 5.0) -> List[Tuple[str, np.ndarray]]:
        """Every marking as an (N, 2) polyline in metres, for rendering or overlays."""
        out: List[Tuple[str, np.ndarray]] = []
        for ln in self.lines():
            for k, (p0, p1) in enumerate(ln.segments):
                out.append((f"{ln.name}#{k}" if len(ln.segments) > 1 else ln.name, np.array([p0, p1], float)))
        for arc in self.arcs():
            out.append((arc.name, self.arc_points(arc, arc_step_deg)))
        return out


DEFAULT_PITCH = PitchModel()


def to_corner_origin(xy: np.ndarray, pitch: PitchModel = DEFAULT_PITCH) -> np.ndarray:
    """Centre-origin metres -> corner-origin metres (what the R engine's
    ``to_attacking_frame(..., origin = "centre")`` does internally)."""
    p = np.asarray(xy, float).reshape(-1, 2).copy()
    p[:, 0] += pitch.half_length
    p[:, 1] += pitch.half_width
    return p
