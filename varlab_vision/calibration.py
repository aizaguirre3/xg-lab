"""Camera-to-pitch calibration from the pitch markings, written from scratch.

Classical and licence-free: no learned keypoint detector, no code from
PnLCalib (GPL-2) or TrackLab / sn-gamestate (GPL-3). The chain is

1. ``white_line_mask``: bright, low-saturation pixels next to grass.
2. ``extract_lines``: probabilistic Hough segments, merged into one line per
   marking and refitted to the mask pixels at sub-pixel precision (PCA with
   one outlier-rejection pass).
3. ``search_homography``: a RANSAC-style hypothesis search over assignments of
   four image lines to two ``x = const`` and two ``y = const`` lines of the
   :class:`~varlab_vision.pitch_model.PitchModel`. Every hypothesis is scored on
   all detected lines: an image line is an inlier of a model line when its
   endpoints lie within ``tau_px`` of the projected model line *and* its
   back-projection lies on a drawn segment of that line (not on the invisible
   part of the infinite line). The search is vectorised (batched DLT by SVD).
4. ``refine_homography``: Levenberg-Marquardt on the eight homography
   parameters, minimising the pixel distance between the inlier segments'
   endpoints and the projected model lines.
5. ``FrameCalibrator``: per-frame state. Frame t is first *tracked* from frame
   t-1 (model lines projected with the previous homography, matched to the new
   segments, refined), falling back to the global search when tracking has too
   few correspondences, and both are preceded by a shot-cut test (mean
   absolute difference of the downscaled grey frames above ``cut_threshold``
   resets all state). Consecutive solutions are blended (exponential smoothing)
   unless a cut was detected.

A frame is **calibrated** only when at least ``min_inliers`` lines match with at
least two from each family (otherwise the homography is unconstrained along one
pitch axis); anything else is reported as ``uncalibrated`` with NaN pitch
coordinates, never as a guess.

Errors reported per frame: ``fit_error_px`` (RMS pixel distance of inlier
endpoints to their projected model lines, the quantity minimised) and
``reprojection_error_m`` (RMS metre distance of the same endpoints,
back-projected, to the model lines: how the pixel error looks on the pitch).

Limits, stated once: single camera, planar pitch (no height, no lens
distortion), straight markings only (the circle and arcs are not used yet),
and a view that shows at least four markings. Accuracy on real broadcast
frames is unmeasured until SoccerNet clips are available; the synthetic clip in
``synthetic.py`` has perfect lines and no crowd, so numbers from it are an upper
bound on what to expect.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pitch_model import DEFAULT_PITCH, X_CONST, Y_CONST, ModelLine, PitchModel

STATUS_CALIBRATED = "calibrated"
STATUS_TRACKED = "tracked"
STATUS_UNCALIBRATED = "uncalibrated"


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------


@dataclass
class ImageLine:
    """A straight marking in the image: sub-pixel line plus its visible extent."""

    p0: np.ndarray
    p1: np.ndarray
    abc: np.ndarray  # a x + b y + c = 0, with a^2 + b^2 = 1
    n_pixels: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def mid(self) -> np.ndarray:
        return (self.p0 + self.p1) / 2.0

    @property
    def angle(self) -> float:
        """Direction in radians, in [0, pi)."""
        d = self.p1 - self.p0
        return float(math.atan2(d[1], d[0]) % math.pi)

    def endpoints(self) -> np.ndarray:
        return np.stack([self.p0, self.p1], axis=0)


@dataclass
class LineMatch:
    image_index: int
    model_name: str
    distance_px: float
    extent_fraction: float


@dataclass
class CalibrationResult:
    status: str
    h_m2px: Optional[np.ndarray] = None
    h_px2m: Optional[np.ndarray] = None
    matches: List[LineMatch] = field(default_factory=list)
    n_lines: int = 0
    fit_error_px: float = float("nan")
    reprojection_error_m: float = float("nan")
    shot_cut: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_CALIBRATED, STATUS_TRACKED) and self.h_px2m is not None

    @property
    def n_inliers(self) -> int:
        return len(self.matches)

    def project(self, points_px: np.ndarray) -> np.ndarray:
        """Pixels -> pitch metres; NaN everywhere when the frame is uncalibrated."""
        pts = np.asarray(points_px, float).reshape(-1, 2)
        if not self.ok:
            return np.full((len(pts), 2), np.nan)
        return apply_homography(self.h_px2m, pts)

    def project_boxes(self, xyxy: np.ndarray) -> Dict[str, np.ndarray]:
        """Bottom-left, bottom-right and bottom-middle of each box in metres."""
        b = np.asarray(xyxy, float).reshape(-1, 4)
        left = self.project(np.stack([b[:, 0], b[:, 3]], axis=1))
        right = self.project(np.stack([b[:, 2], b[:, 3]], axis=1))
        return {"left": left, "right": right, "middle": (left + right) / 2.0}

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "homography_px2m": None if self.h_px2m is None else np.round(self.h_px2m, 8).tolist(),
            "n_lines": int(self.n_lines),
            "n_inliers": int(self.n_inliers),
            "fit_error_px": None if math.isnan(self.fit_error_px) else round(self.fit_error_px, 3),
            "reprojection_error_m": (None if math.isnan(self.reprojection_error_m)
                                     else round(self.reprojection_error_m, 3)),
            "shot_cut": bool(self.shot_cut),
            "matches": [{"line": m.model_name, "image_index": m.image_index,
                         "distance_px": round(m.distance_px, 2)} for m in self.matches],
            "note": self.note,
        }


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.asarray(h, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


def _normalise_lines(abc: np.ndarray) -> np.ndarray:
    """Scale so that a^2 + b^2 = 1 (works on (..., 3) arrays)."""
    n = np.sqrt(abc[..., 0] ** 2 + abc[..., 1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return abc / n[..., None]


def project_model_lines(h_px2m: np.ndarray, model_abc: np.ndarray) -> np.ndarray:
    """Model lines (metres, (M, 3)) -> image lines (pixels, (M, 3), normalised).

    A line ``l`` transforms by the inverse transpose of the point homography:
    with ``x_m = H x_px``, ``l_px = H^T l_m``.
    """
    return _normalise_lines(model_abc @ np.asarray(h_px2m, float))


def _dlt_batch(src: np.ndarray, dst: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Batched 4-point DLT: ``src`` and ``dst`` are (B, 4, 2); returns (B, 3, 3)
    homographies with ``dst ~ H src`` and ``h33 = 1``.

    Solved through the 8x8 normal equations with a tiny ridge so that a degenerate
    hypothesis (three collinear points) yields a finite but useless matrix instead
    of raising; the scorer then discards it. Much faster than a batched SVD.
    """
    B = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z = np.zeros_like(x)
    o = np.ones_like(x)
    r1 = np.stack([x, y, o, z, z, z, -u * x, -u * y], axis=-1)
    r2 = np.stack([z, z, z, x, y, o, -v * x, -v * y], axis=-1)
    a = np.concatenate([r1, r2], axis=1)  # (B, 8, 8)
    b = np.concatenate([u, v], axis=1)[..., None]  # (B, 8, 1)
    ata = np.einsum("bij,bik->bjk", a, a) + ridge * np.eye(8)[None]
    atb = np.einsum("bij,bik->bjk", a, b)
    sol = np.linalg.solve(ata, atb)[..., 0]
    h = np.concatenate([sol, np.ones((B, 1))], axis=1).reshape(B, 3, 3)
    return h


def _quad_orientation_ok(pts: np.ndarray, min_area: float = 50.0) -> np.ndarray:
    """Shoelace test on (B, 4, 2) image quads ordered (xa,ya), (xa,yb), (xb,ya), (xb,yb).

    Re-ordered as a ring (xa,ya) -> (xb,ya) -> (xb,yb) -> (xa,yb), the model quad has
    positive signed area in a y-down frame; a non-mirrored camera keeps that sign.
    """
    ring = pts[:, [0, 2, 3, 1], :]
    x, y = ring[..., 0], ring[..., 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    return np.isfinite(area) & (area > min_area)


def _segment_intervals(model: PitchModel) -> Tuple[np.ndarray, np.ndarray, List[ModelLine]]:
    """Per model line: (M, 3) homogeneous form and (M, 2, 2) drawn intervals along
    the free coordinate (second interval NaN when the line is one segment)."""
    lines = model.lines()
    abc = np.stack([ln.homogeneous for ln in lines], axis=0)
    iv = np.full((len(lines), 2, 2), np.nan)
    for i, ln in enumerate(lines):
        for k, (p0, p1) in enumerate(ln.segments[:2]):
            t0, t1 = (p0[1], p1[1]) if ln.kind == X_CONST else (p0[0], p1[0])
            iv[i, k] = (min(t0, t1), max(t0, t1))
    return abc, iv, lines


def extent_fraction(t_lo: np.ndarray, t_hi: np.ndarray, intervals: np.ndarray, tol_m: float) -> np.ndarray:
    """Fraction of ``[t_lo, t_hi]`` covered by the (tolerance-widened) drawn intervals.

    Broadcasts: ``t_lo``/``t_hi`` of shape (...,) against ``intervals`` (..., 2, 2).
    """
    seg = np.maximum(t_hi - t_lo, 1e-9)
    cov = np.zeros_like(seg)
    for k in range(intervals.shape[-2]):
        a = intervals[..., k, 0] - tol_m
        b = intervals[..., k, 1] + tol_m
        ov = np.minimum(t_hi, b) - np.maximum(t_lo, a)
        cov = cov + np.where(np.isnan(a), 0.0, np.clip(ov, 0.0, None))
    return np.clip(cov / seg, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 1 + 2. Marking pixels and lines
# ----------------------------------------------------------------------------


def white_line_mask(frame_bgr: np.ndarray, white_max_sat: int = 70, white_min_val: int = 150,
                    green_hue: Tuple[int, int] = (35, 85), green_min_sat: int = 60,
                    grass_dilate_px: int = 15) -> np.ndarray:
    """Binary mask of marking pixels: bright, unsaturated, and within ``grass_dilate_px`` of grass."""
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (s <= white_max_sat) & (v >= white_min_val)
    green = ((h >= green_hue[0]) & (h <= green_hue[1]) & (s >= green_min_sat)).astype(np.uint8)
    if grass_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grass_dilate_px + 1, 2 * grass_dilate_px + 1))
        green = cv2.dilate(green, k)
    return (white & (green > 0)).astype(np.uint8)


def _fit_line_pca(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through (N, 2) points: (abc normalised, direction)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    abc = np.array([n[0], n[1], -(n @ c)])
    return abc, d


def extract_lines(mask: np.ndarray, min_length_px: float = 25.0, hough_threshold: int = 30,
                  max_gap_px: int = 10, merge_angle_deg: float = 2.0, merge_dist_px: float = 3.0,
                  band_px: float = 2.0, min_coverage: float = 0.6, max_lines: int = 24) -> List[ImageLine]:
    """Hough segments -> merged, sub-pixel lines with their visible extent.

    Segments are clustered by angle and by signed distance from the image centre
    (tight tolerances: in a wide shot the far-end markings are only a few pixels
    apart). Each cluster is refitted by PCA to the mask pixels within ``band_px``
    of it with two outlier-rejection passes, and its extent is trimmed to the part
    of the line that white pixels actually cover. A line whose extent is covered
    for less than ``min_coverage`` of its length is a chord through other
    markings, not a marking, and is dropped. Curved markings break into short
    segments at varying angles and fall below ``min_length_px``.
    """
    import cv2

    h, w = mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    segs = cv2.HoughLinesP(mask, rho=1.0, theta=np.pi / 360.0, threshold=int(hough_threshold),
                           minLineLength=int(max(8, min_length_px * 0.4)), maxLineGap=int(max_gap_px))
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    d = segs[:, 2:] - segs[:, :2]
    length = np.linalg.norm(d, axis=1)
    keep = length > 2
    segs, d, length = segs[keep], d[keep], length[keep]
    if len(segs) == 0:
        return []
    theta = np.arctan2(d[:, 1], d[:, 0]) % np.pi
    order = np.argsort(-length)
    tol_th = math.radians(merge_angle_deg)
    # Greedy clustering, longest first. A segment joins a cluster when its direction
    # is within ``merge_angle_deg`` of the cluster line and both its endpoints lie
    # within ``merge_dist_px`` of that line (an endpoint test, not a rho test: a
    # small angle error moves rho by several pixels over a long line). The cluster
    # line is refitted from all member endpoints after every merge.
    clusters: List[List[int]] = []
    c_abc: List[np.ndarray] = []
    c_dir: List[np.ndarray] = []
    c_theta: List[float] = []
    c_iv: List[List[Tuple[float, float]]] = []  # covered intervals along the cluster line

    def overlap(iv_list, lo, hi) -> float:
        return sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in iv_list)

    for i in order:
        e_i = np.array([segs[i, :2], segs[i, 2:]])
        placed = False
        for k in range(len(clusters)):
            dth = abs(theta[i] - c_theta[k])
            dth = min(dth, math.pi - dth)
            if dth >= tol_th:
                continue
            if np.abs(e_i @ c_abc[k][:2] + c_abc[k][2]).max() >= merge_dist_px:
                continue
            t = e_i @ c_dir[k]
            lo, hi = float(t.min()), float(t.max())
            # pieces of one marking lie end to end; a parallel neighbour lies alongside
            if overlap(c_iv[k], lo, hi) > max(4.0, 0.25 * (hi - lo)):
                continue
            clusters[k].append(int(i))
            c_iv[k].append((lo, hi))
            ends_k = np.concatenate([segs[clusters[k], :2], segs[clusters[k], 2:]], axis=0)
            abc_k, d_k = _fit_line_pca(ends_k)
            c_abc[k], c_dir[k] = abc_k, d_k
            c_theta[k] = float(math.atan2(d_k[1], d_k[0]) % math.pi)
            c_iv[k] = [(float(a), float(b)) for a, b in
                       ((min(p @ d_k for p in np.array([segs[j, :2], segs[j, 2:]])),
                         max(p @ d_k for p in np.array([segs[j, :2], segs[j, 2:]]))) for j in clusters[k])]
            placed = True
            break
        if not placed:
            clusters.append([int(i)])
            abc_i, d_i = _fit_line_pca(e_i)
            c_abc.append(abc_i)
            c_dir.append(d_i)
            c_theta.append(float(theta[i]))
            t = e_i @ d_i
            c_iv.append([(float(t.min()), float(t.max()))])

    ys, xs = np.nonzero(mask)
    pix = np.stack([xs, ys], axis=1).astype(float)
    thick = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out: List[ImageLine] = []
    for members in clusters:
        m = segs[members]
        ends = np.concatenate([m[:, :2], m[:, 2:]], axis=0)
        abc, dvec = _fit_line_pca(ends)
        c0 = ends.mean(axis=0)
        t_end = (ends - c0) @ dvec
        lo, hi = t_end.min() - band_px, t_end.max() + band_px
        pts = None
        for cutoff in (band_px, 1.2):
            dist = np.abs(pix @ abc[:2] + abc[2])
            t = (pix - c0) @ dvec
            sel = (dist <= cutoff) & (t >= lo) & (t <= hi)
            if sel.sum() < 6:
                pts = None
                break
            pts = pix[sel]
            abc, dvec = _fit_line_pca(pts)
        if pts is None:
            continue
        c = pts.mean(axis=0)
        tt = (pts - c) @ dvec
        t_lo, t_hi = tt.min(), tt.max()
        # coverage: walk the line in 1 px steps and look for white within +-1 px
        n = int(max(2, math.ceil(t_hi - t_lo)))
        samp = c + np.outer(np.linspace(t_lo, t_hi, n), dvec)
        sx = np.clip(np.round(samp[:, 0]).astype(int), 0, w - 1)
        sy = np.clip(np.round(samp[:, 1]).astype(int), 0, h - 1)
        hit = thick[sy, sx] > 0
        if hit.sum() < 2:
            continue
        idx = np.nonzero(hit)[0]
        first, last = idx[0], idx[-1]
        coverage = float(hit[first:last + 1].mean())
        p0, p1 = samp[first], samp[last]
        if np.linalg.norm(p1 - p0) < min_length_px or coverage < min_coverage:
            continue
        out.append(ImageLine(p0=p0, p1=p1, abc=abc, n_pixels=int(len(pts))))
    out.sort(key=lambda ln: -ln.length)
    kept: List[ImageLine] = []
    for ln in out:
        dup = False
        for k in kept:
            dth = abs(ln.angle - k.angle)
            dth = min(dth, math.pi - dth)
            if dth < tol_th and np.abs(ln.endpoints() @ k.abc[:2] + k.abc[2]).max() < merge_dist_px:
                dup = True
                break
        if not dup:
            kept.append(ln)
    return kept[:max_lines]


# ----------------------------------------------------------------------------
# 3. Hypothesis search
# ----------------------------------------------------------------------------


def _score_hypotheses(h_m2px: np.ndarray, lines: Sequence[ImageLine], model_abc: np.ndarray,
                      intervals: np.ndarray, model_kind: np.ndarray, tau_px: float, tol_m: float,
                      min_extent: float = 0.7, half_width: float = 34.0,
                      image_centre: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score (B,) hypotheses on all image lines.

    Returns (score, best_model_index (B, N), inlier mask (B, N)). An image line is
    an inlier of its nearest projected model line when both endpoints are within
    ``tau_px`` and the back-projected segment lies on a drawn part of the line.
    """
    B = h_m2px.shape[0]
    N = len(lines)
    ends = np.stack([ln.endpoints() for ln in lines], axis=0)  # (N, 2, 2)
    lengths = np.array([ln.length for ln in lines])
    with np.errstate(divide="ignore", invalid="ignore"):
        h_px2m = np.linalg.inv(h_m2px)
    # Camera-side prior: the markings are symmetric under a 180-degree turn of the
    # pitch, so lines alone cannot tell the near touchline from the far one. The
    # GSR frame puts the camera on the +y side, so (0, +W/2) must project lower in
    # the image (larger v) than (0, -W/2), and both must be in front of the camera.
    probe = np.array([[0.0, -half_width, 1.0], [0.0, half_width, 1.0]])  # (2, 3)
    pr = np.einsum("bij,kj->bki", h_m2px, probe)  # (B, 2, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = pr[..., 1] / pr[..., 2]
    orient_ok = (pr[..., 2] > 0).all(axis=1) & (v[:, 1] > v[:, 0])
    # Handedness: pixels (x right, y down) and the pitch frame (x right, y toward
    # the camera) have the same orientation, so the Jacobian of the pixel -> metre
    # map must have positive determinant at the image centre, which is always in
    # front of the camera. For a homography det J(p) = det(H) / w(p)^3.
    centre = np.array([image_centre[0], image_centre[1], 1.0])
    w_c = h_px2m @ centre  # (B, 3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        jac_sign = np.linalg.det(h_px2m) / (w_c[:, 2] ** 3)
    orient_ok &= np.isfinite(jac_sign) & (jac_sign > 0)
    # projected model lines in the image: (B, M, 3)
    l_px = _normalise_lines(np.einsum("mk,bkj->bmj", model_abc, h_px2m))
    ends_h = np.concatenate([ends, np.ones((N, 2, 1))], axis=-1)  # (N, 2, 3)
    dist = np.abs(np.einsum("nek,bmk->bnme", ends_h, l_px))  # (B, N, M, 2)
    dist = dist.max(axis=-1)  # worst endpoint
    best = np.argmin(dist, axis=2)  # (B, N)
    best_d = np.take_along_axis(dist, best[..., None], axis=2)[..., 0]
    # back-project endpoints to metres: (B, N, 2, 3)
    bp = np.einsum("nek,bjk->bnej", ends_h, h_px2m)
    w = bp[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = bp[..., :2] / w[..., None]
    kind = model_kind[best]  # (B, N): True where x_const (free coordinate is y)
    t = np.where(kind[..., None], xy[..., 1], xy[..., 0])  # (B, N, 2)
    t_lo, t_hi = t.min(axis=-1), t.max(axis=-1)
    iv = intervals[best]  # (B, N, 2, 2)
    frac = extent_fraction(t_lo, t_hi, iv, tol_m)
    same_sign = (np.sign(w[..., 0]) == np.sign(w[..., 1])) & np.isfinite(xy).all(axis=(-1, -2))
    inlier = (best_d <= tau_px) & (frac >= min_extent) & same_sign
    score = (inlier * lengths[None, :] * frac).sum(axis=1)
    score = np.where(np.isfinite(h_m2px).all(axis=(1, 2)) & orient_ok, score, -1.0)
    return score, best, inlier


def search_homography(lines: Sequence[ImageLine], width: int, height: int, model: PitchModel = DEFAULT_PITCH,
                      tau_px: Optional[float] = None, tol_m: float = 2.0, top_k: int = 6, n_score: int = 10,
                      n_rescore: int = 64, chunk: int = 40000) -> Tuple[Optional[np.ndarray], List[LineMatch], float]:
    """Global search: assign four image lines to two x = const and two y = const
    model lines, solve the homography from their four intersections, score on all
    lines; returns ``(H_m2px, matches, score)`` or ``(None, [], 0)``.
    """
    if len(lines) < 4:
        return None, [], 0.0
    tau = tau_px if tau_px is not None else max(2.5, 0.006 * width)
    model_abc, intervals, mlines = _segment_intervals(model)
    model_kind = np.array([ln.kind == X_CONST for ln in mlines])
    xs = [(i, ln.value) for i, ln in enumerate(mlines) if ln.kind == X_CONST]
    ys = [(i, ln.value) for i, ln in enumerate(mlines) if ln.kind == Y_CONST]
    k = min(top_k, len(lines))
    cand = list(range(k))  # lines are sorted by length
    # image assignments: ordered pair A -> x_const, ordered pair B -> y_const (disjoint)
    img_assign = []
    for a in itertools.combinations(cand, 2):
        rest = [c for c in cand if c not in a]
        for b in itertools.combinations(rest, 2):
            for a_ord in (a, a[::-1]):
                for b_ord in (b, b[::-1]):
                    img_assign.append((a_ord, b_ord))
    model_assign = [(xa, xb, ya, yb) for xa, xb in itertools.combinations(xs, 2)
                    for ya, yb in itertools.combinations(ys, 2)]
    # intersections of image lines (A_i with B_j) give the four image points; the model
    # points are (x_a, y_a), (x_a, y_b), (x_b, y_a), (x_b, y_b).
    abcs = np.stack([ln.abc for ln in lines], axis=0)

    def intersect(i, j):
        p = np.cross(abcs[i], abcs[j])
        return p[:2] / p[2] if abs(p[2]) > 1e-12 else np.array([np.nan, np.nan])

    img_pts = []
    for (a1, a2), (b1, b2) in img_assign:
        img_pts.append([intersect(a1, b1), intersect(a1, b2), intersect(a2, b1), intersect(a2, b2)])
    img_pts = np.array(img_pts)  # (IA, 4, 2)
    mod_pts = np.array([[(xa[1], ya[1]), (xa[1], yb[1]), (xb[1], ya[1]), (xb[1], yb[1])]
                        for xa, xb, ya, yb in model_assign])  # (MA, 4, 2)
    IA, MA = len(img_assign), len(model_assign)
    ok_img = np.isfinite(img_pts).all(axis=(1, 2)) & _quad_orientation_ok(img_pts)
    # normalisation for conditioning
    s_px = np.array([width, height], float)
    s_m = np.array([model.half_length, model.half_width], float)
    pairs = [(i, j) for i in range(IA) if ok_img[i] for j in range(MA)]
    score_lines = list(lines[:n_score])
    leaders: List[Tuple[float, np.ndarray]] = []
    for start in range(0, len(pairs), chunk):
        sub = pairs[start:start + chunk]
        ii = np.array([p[0] for p in sub]); jj = np.array([p[1] for p in sub])
        src = mod_pts[jj] / s_m
        dst = img_pts[ii] / s_px
        hn = _dlt_batch(src, dst)
        # unnormalise: px = S_px ... H_n ... S_m^-1
        h = np.einsum("ij,bjk,kl->bil", np.diag([s_px[0], s_px[1], 1.0]), hn, np.diag([1 / s_m[0], 1 / s_m[1], 1.0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = h / h[:, 2:3, 2:3]
        score, _best, _inl = _score_hypotheses(h, score_lines, model_abc, intervals, model_kind, tau, tol_m,
                                               half_width=model.half_width, image_centre=(width / 2.0, height / 2.0))
        top = np.argsort(-score)[:n_rescore]
        leaders.extend((float(score[b]), h[b].copy()) for b in top if score[b] > 0)
    if not leaders:
        return None, [], 0.0
    leaders.sort(key=lambda t: -t[0])
    hs = np.stack([h for _, h in leaders[:n_rescore]], axis=0)
    full, _b, _i = _score_hypotheses(hs, lines, model_abc, intervals, model_kind, tau, tol_m,
                                     half_width=model.half_width, image_centre=(width / 2.0, height / 2.0))
    b = int(np.argmax(full))
    if full[b] <= 0:
        return None, [], 0.0
    best_h = hs[b]
    matches = match_lines(best_h, lines, model, tau_px=tau, tol_m=tol_m)
    return best_h, matches, float(full[b])


def match_lines(h_m2px: np.ndarray, lines: Sequence[ImageLine], model: PitchModel = DEFAULT_PITCH,
                tau_px: float = 4.0, tol_m: float = 2.0, min_extent: float = 0.7,
                image_centre: Optional[Tuple[float, float]] = None) -> List[LineMatch]:
    """Image lines that lie on a drawn model segment under ``h_m2px`` (one model line
    may take several image segments; each image line takes at most one model line)."""
    if len(lines) == 0:
        return []
    model_abc, intervals, mlines = _segment_intervals(model)
    model_kind = np.array([ln.kind == X_CONST for ln in mlines])
    if image_centre is None:  # the mean of the line midpoints is always in front of the camera
        image_centre = tuple(np.mean([ln.mid for ln in lines], axis=0))
    score, best, inl = _score_hypotheses(h_m2px[None], lines, model_abc, intervals, model_kind, tau_px, tol_m,
                                         min_extent=min_extent, half_width=model.half_width,
                                         image_centre=image_centre)
    if score[0] <= 0:
        return []
    out: List[LineMatch] = []
    h_px2m = np.linalg.inv(h_m2px)
    l_px = project_model_lines(h_px2m, model_abc)
    for n, ln in enumerate(lines):
        if not inl[0, n]:
            continue
        m = int(best[0, n])
        e = np.concatenate([ln.endpoints(), np.ones((2, 1))], axis=1)
        d = float(np.abs(e @ l_px[m]).max())
        xy = apply_homography(h_px2m, ln.endpoints())
        t = xy[:, 1] if model_kind[m] else xy[:, 0]
        fr = float(extent_fraction(np.array(t.min()), np.array(t.max()), intervals[m][None], tol_m)[0])
        out.append(LineMatch(image_index=n, model_name=mlines[m].name, distance_px=d, extent_fraction=fr))
    return out


# ----------------------------------------------------------------------------
# 4. Refinement
# ----------------------------------------------------------------------------


def refine_homography(h_m2px: np.ndarray, lines: Sequence[ImageLine], matches: Sequence[LineMatch],
                      model: PitchModel = DEFAULT_PITCH, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Levenberg-Marquardt on the 8 free parameters: minimise the pixel distances of
    the matched segments' endpoints and midpoints to the projected model lines."""
    from scipy.optimize import least_squares

    if len(matches) == 0:
        return h_m2px
    model_lines = {ln.name: ln for ln in model.lines()}
    pts, mabc, wts = [], [], []
    for m in matches:
        ln = lines[m.image_index]
        ml = model_lines[m.model_name].homogeneous
        for p in (ln.p0, ln.mid, ln.p1):
            pts.append(p); mabc.append(ml); wts.append(math.sqrt(max(ln.length, 1.0)))
    pts = np.concatenate([np.array(pts), np.ones((len(pts), 1))], axis=1)
    mabc = np.array(mabc)
    wts = np.array(wts) / np.sqrt(np.mean(np.array(wts) ** 2))
    s_px = np.diag([width, height, 1.0])
    s_px_inv = np.linalg.inv(s_px)
    s_m = np.diag([model.half_length, model.half_width, 1.0])
    hn0 = s_px_inv @ h_m2px @ s_m  # normalised parametrisation
    hn0 = hn0 / hn0[2, 2]

    def unpack(p):
        hn = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])
        return s_px @ hn @ np.linalg.inv(s_m)

    def residuals(p):
        h = unpack(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_inv = np.linalg.inv(h)
        l_px = _normalise_lines(mabc @ h_inv)
        return wts * (pts * l_px).sum(axis=1)

    p0 = hn0.reshape(-1)[:8]
    try:
        sol = least_squares(residuals, p0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=400)
        h = unpack(sol.x)
        if np.isfinite(h).all():
            return h / h[2, 2]
    except (ValueError, np.linalg.LinAlgError):
        pass
    return h_m2px


def _errors(h_m2px: np.ndarray, lines: Sequence[ImageLine], matches: Sequence[LineMatch],
            model: PitchModel) -> Tuple[float, float]:
    if len(matches) == 0:
        return float("nan"), float("nan")
    model_lines = {ln.name: ln for ln in model.lines()}
    h_px2m = np.linalg.inv(h_m2px)
    d_px, d_m = [], []
    for m in matches:
        ln = lines[m.image_index]
        ml = model_lines[m.model_name].homogeneous
        l_px = _normalise_lines(ml @ h_px2m)
        e = np.concatenate([ln.endpoints(), np.ones((2, 1))], axis=1)
        d_px.extend(np.abs(e @ l_px).tolist())
        xy = apply_homography(h_px2m, ln.endpoints())
        d_m.extend(np.abs(np.concatenate([xy, np.ones((2, 1))], axis=1) @ ml).tolist())
    return float(np.sqrt(np.mean(np.square(d_px)))), float(np.sqrt(np.mean(np.square(d_m))))


def _families_ok(matches: Sequence[LineMatch], model: PitchModel, min_inliers: int) -> bool:
    kinds = {ln.name: ln.kind for ln in model.lines()}
    names = {m.model_name for m in matches}
    nx = sum(1 for n in names if kinds[n] == X_CONST)
    ny = sum(1 for n in names if kinds[n] == Y_CONST)
    return len(matches) >= min_inliers and nx >= 2 and ny >= 2


def calibrate_frame(frame_bgr: np.ndarray, model: PitchModel = DEFAULT_PITCH, prior_m2px: Optional[np.ndarray] = None,
                    min_inliers: int = 4, tau_px: Optional[float] = None, track_tau_px: Optional[float] = None,
                    mask_kw: Optional[dict] = None, line_kw: Optional[dict] = None) -> CalibrationResult:
    """Calibrate one frame from scratch, or from ``prior_m2px`` (tracking) with a
    global search as the fallback."""
    h, w = frame_bgr.shape[:2]
    tau = tau_px if tau_px is not None else max(2.5, 0.006 * w)
    tau_track = track_tau_px if track_tau_px is not None else 4.0 * tau
    mask = white_line_mask(frame_bgr, **(mask_kw or {}))
    lines = extract_lines(mask, **(line_kw or {}))
    if len(lines) < 4:
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines),
                                 note=f"{len(lines)} straight markings found, need 4")

    def finish(h0: np.ndarray, status: str, note: str) -> Optional[CalibrationResult]:
        hh = h0
        for round_tau in (tau_track if status == STATUS_TRACKED else tau * 2.0, tau):
            matches = match_lines(hh, lines, model, tau_px=round_tau)
            if not _families_ok(matches, model, min_inliers):
                return None
            hh = refine_homography(hh, lines, matches, model, width=w, height=h)
        matches = match_lines(hh, lines, model, tau_px=tau)
        if not _families_ok(matches, model, min_inliers):
            return None
        fit_px, rep_m = _errors(hh, lines, matches, model)
        return CalibrationResult(status=status, h_m2px=hh, h_px2m=np.linalg.inv(hh) / np.linalg.inv(hh)[2, 2],
                                 matches=matches, n_lines=len(lines), fit_error_px=fit_px,
                                 reprojection_error_m=rep_m, note=note)

    if prior_m2px is not None:
        res = finish(prior_m2px, STATUS_TRACKED, "tracked from the previous frame")
        if res is not None:
            return res
    h_best, matches, score = search_homography(lines, w, h, model, tau_px=tau)
    if h_best is None or not _families_ok(matches, model, min_inliers):
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines),
                                 note=f"global search: {len(matches)} consistent lines, need {min_inliers} across both families")
    res = finish(h_best, STATUS_CALIBRATED, f"global search over {len(lines)} lines")
    if res is None:
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines),
                                 note="global search found a hypothesis that did not survive refinement")
    return res


# ----------------------------------------------------------------------------
# 5. Per-frame state
# ----------------------------------------------------------------------------


def frame_difference(prev_small: Optional[np.ndarray], frame_bgr: np.ndarray, size: Tuple[int, int] = (64, 36)) -> Tuple[float, np.ndarray]:
    """Mean absolute grey difference against the previous downscaled frame (0-255)."""
    import cv2

    small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(float)
    if prev_small is None:
        return 0.0, small
    return float(np.abs(small - prev_small).mean()), small


class FrameCalibrator:
    """Stateful per-frame calibration with tracking, smoothing and shot-cut resets.

    ``update(frame)`` returns a :class:`CalibrationResult` whose ``project`` maps
    pixels to metres (NaN when uncalibrated). ``name`` lets the pipeline report it
    as its projector backend.
    """

    name = "line-calibration"

    def __init__(self, model: PitchModel = DEFAULT_PITCH, cut_threshold: float = 30.0, smoothing: float = 0.6,
                 hold_frames: int = 5, min_inliers: int = 4, **calibrate_kw):
        self.model = model
        self.cut_threshold = cut_threshold
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.min_inliers = min_inliers
        self.calibrate_kw = calibrate_kw
        self.reset()

    def reset(self) -> None:
        self._prior: Optional[np.ndarray] = None
        self._prev_small: Optional[np.ndarray] = None
        self._missed = 0
        self.history: List[str] = []

    def update(self, frame_bgr: np.ndarray) -> CalibrationResult:
        diff, small = frame_difference(self._prev_small, frame_bgr)
        cut = self._prev_small is not None and diff >= self.cut_threshold
        self._prev_small = small
        if cut:
            self._prior = None
            self._missed = 0
        res = calibrate_frame(frame_bgr, self.model, prior_m2px=self._prior, min_inliers=self.min_inliers,
                              **self.calibrate_kw)
        res.shot_cut = bool(cut)
        if res.ok:
            h_new = res.h_m2px
            if self._prior is not None and res.status == STATUS_TRACKED and 0.0 < self.smoothing < 1.0:
                h_s = self.smoothing * h_new + (1.0 - self.smoothing) * self._prior
                h_s = h_s / h_s[2, 2]
                # keep the smoothed solution only if it still explains the lines
                res.h_m2px = h_s
                res.h_px2m = np.linalg.inv(h_s) / np.linalg.inv(h_s)[2, 2]
                mask = white_line_mask(frame_bgr)
                lines = extract_lines(mask)
                res.matches = match_lines(h_s, lines, self.model, tau_px=self.calibrate_kw.get("tau_px") or max(2.5, 0.006 * frame_bgr.shape[1]))
                res.fit_error_px, res.reprojection_error_m = _errors(h_s, lines, res.matches, self.model)
            self._prior = res.h_m2px
            self._missed = 0
        else:
            self._missed += 1
            if self._missed > self.hold_frames:
                self._prior = None
        self.history.append(res.status)
        return res

    # PitchProjector-compatible static use: project with the last good homography.
    def project(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, float).reshape(-1, 2)
        if self._prior is None:
            return np.full((len(pts), 2), np.nan)
        return apply_homography(np.linalg.inv(self._prior), pts)
