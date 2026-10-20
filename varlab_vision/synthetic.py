"""A synthetic clip: a rendered pitch seen by a broadcast-style camera, with
coloured rectangles standing on it.

This exists to exercise the pipeline and the calibration code path on a CPU
with no data. The rectangles are not people, so the real detectors (RF-DETR)
will not find them; the smoke backends in ``pipeline.py`` (colour blobs,
mean-colour team split) do. The pitch markings, however, are the real ones from
``pitch_model.py`` drawn under a known pinhole-camera homography, so the
calibration stage can be scored against ground truth. The clip comes with a
``Labels-GameState.json`` in the GSR layout so the loader and the GS-HOTA
adapter run end to end on it.

Camera model. ``BroadcastCamera`` is a pinhole camera at ``position`` (metres in
the pitch frame, ``z`` up) looking at ``look_at`` on the pitch plane, with a
horizontal field of view ``hfov_deg`` and no distortion. Image x grows toward
the right goal and image y grows toward the near touchline, which is the GSR
convention. For the plane ``z = 0`` the projection is the homography
``H_m2px = K [r1 r2 t]``; ``homography_px2m`` is its inverse. A per-frame
*pan* shifts ``look_at`` along x; a *shot cut* switches to a second camera
(and optionally a different green) at ``cut_at_frame``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .gsr_format import (
    LABELS_FILE,
    BBoxImage,
    BBoxPitch,
    ClipInfo,
    GSRClip,
    ImageRecord,
    ObjectAnnotation,
    category_id_for_role,
    default_categories_list,
    frame_file_name,
    new_annotation_id,
    new_image_id,
    parse_categories,
    write_labels_json,
)
from .pitch_model import DEFAULT_PITCH, PitchModel

GREEN_BGR = (40, 150, 40)
LINE_BGR = (255, 255, 255)
TEAM_COLOURS_BGR = {"left": (30, 30, 220), "right": (220, 60, 30)}  # red-ish, blue-ish (BGR)
REFEREE_BGR = (20, 20, 20)


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BroadcastCamera:
    width: int
    height: int
    hfov_deg: float
    position: Tuple[float, float, float]  # metres, pitch frame, z up
    look_at: Tuple[float, float]  # a point on the pitch plane

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def intrinsics(self) -> np.ndarray:
        f = self.focal_px
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    def rotation(self) -> np.ndarray:
        """World -> camera rotation with rows (right, down, forward)."""
        c = np.array(self.position, float)
        t = np.array([self.look_at[0], self.look_at[1], 0.0])
        fwd = t - c
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, fwd)
        right /= np.linalg.norm(right)
        down = np.cross(right, fwd)
        return np.stack([right, down, fwd], axis=0)

    def homography_m2px(self) -> np.ndarray:
        r = self.rotation()
        c = np.array(self.position, float)
        t = -r @ c
        h = self.intrinsics() @ np.stack([r[:, 0], r[:, 1], t], axis=1)
        return h / h[2, 2]

    def homography_px2m(self) -> np.ndarray:
        h = np.linalg.inv(self.homography_m2px())
        return h / h[2, 2]

    def project(self, xy_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pitch metres -> pixels; also returns the homogeneous depth sign (``w > 0`` = in front)."""
        pts = np.asarray(xy_m, float).reshape(-1, 2)
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ self.homography_m2px().T
        with np.errstate(divide="ignore", invalid="ignore"):
            px = hom[:, :2] / hom[:, 2:3]
        return px, hom[:, 2]

    def panned(self, dx_m: float) -> "BroadcastCamera":
        return replace(self, look_at=(self.look_at[0] + dx_m, self.look_at[1]))


def wide_camera(width: int, height: int, pitch: PitchModel = DEFAULT_PITCH) -> BroadcastCamera:
    """A high main-stand camera that sees the whole pitch with a margin (the
    default clip: every actor stays in frame, every marking is visible)."""
    return BroadcastCamera(width=width, height=height, hfov_deg=96.0,
                           position=(0.0, pitch.half_width + 40.0, 30.0), look_at=(0.0, -2.0))


def medium_camera(width: int, height: int, look_x: float = -25.0, pitch: PitchModel = DEFAULT_PITCH) -> BroadcastCamera:
    """A tighter broadcast view of one half: penalty area, halfway line, both touchlines."""
    return BroadcastCamera(width=width, height=height, hfov_deg=58.0,
                           position=(0.0, pitch.half_width + 30.0, 18.0), look_at=(look_x, 0.0))


def project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h.T
    return hom[:, :2] / hom[:, 2:3]


def synthetic_homography(width: int, height: int) -> np.ndarray:
    """Kept for reference: the plain pixel -> metre scaling the first smoke clip used.

    ``x_m = (px - width/2) * 105/width``, ``y_m = (py - height/2) * 68/height``.
    The clips rendered by :func:`render_synthetic_clip` now use a camera
    homography instead (``SyntheticClip.homography``).
    """
    return np.array(
        [[105.0 / width, 0.0, -52.5], [0.0, 68.0 / height, -34.0], [0.0, 0.0, 1.0]], dtype=float
    )


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------


def draw_markings(frame: np.ndarray, camera: BroadcastCamera, pitch: PitchModel = DEFAULT_PITCH,
                  thickness: int = 3, colour: Tuple[int, int, int] = LINE_BGR, step_m: float = 1.0) -> None:
    """Draw every marking in white under the camera's homography (in place).

    Long segments are subdivided every ``step_m`` metres and only pieces whose
    endpoints are in front of the camera and near the image are drawn, so a
    marking that crosses the horizon never produces a spurious stroke.
    """
    import cv2

    h, w = frame.shape[:2]
    for _name, poly in pitch.polylines(arc_step_deg=4.0):
        dense: List[np.ndarray] = []
        for p0, p1 in zip(poly[:-1], poly[1:]):
            n = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / step_m)))
            dense.append(np.linspace(p0, p1, n + 1)[:-1])
        dense.append(poly[-1:])
        pts_m = np.concatenate(dense, axis=0)
        px, depth = camera.project(pts_m)
        ok = (depth > 1e-6) & np.isfinite(px).all(axis=1) & (np.abs(px[:, 0]) < 8 * w) & (np.abs(px[:, 1]) < 8 * h)
        # Sub-pixel vertices (4 fractional bits): rounding them to whole pixels shifts a
        # near-horizontal far touchline by a constant fraction of a pixel along its whole
        # length (up to 0.5 px), a bias a camera does not have and that the calibration
        # cannot average out.
        fp = np.round(px * 16.0).astype(np.int64)
        for i in range(len(px) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(frame, (int(fp[i, 0]), int(fp[i, 1])), (int(fp[i + 1, 0]), int(fp[i + 1, 1])), colour,
                         thickness=thickness, lineType=cv2.LINE_AA, shift=4)


@dataclass
class SyntheticClip:
    clip_dir: Path
    labels_path: Path
    width: int
    height: int
    n_frames: int
    homography: np.ndarray  # pixels -> metres for frame 1 (every frame when there is no pan or cut)
    homographies: List[np.ndarray]  # pixels -> metres per frame
    cameras: List[BroadcastCamera]
    cut_at_frame: Optional[int] = None

    @property
    def image_dir(self) -> Path:
        return self.clip_dir / "img1"

    @property
    def is_static(self) -> bool:
        return all(np.allclose(h, self.homographies[0]) for h in self.homographies)


def _box_track(cams: Sequence[BroadcastCamera], x0: float, y0: float, vx: float, vy: float,
               bw: int, bh: int, pitch: PitchModel) -> np.ndarray:
    """Per-frame pixel box (x1, y1, x2, y2) of an actor, NaN when out of frame."""
    out = np.full((len(cams), 4), np.nan)
    for i, cam in enumerate(cams):
        xm = float(np.clip(x0 + vx * i, -pitch.half_length + 1.0, pitch.half_length - 1.0))
        ym = float(np.clip(y0 + vy * i, -pitch.half_width + 1.0, pitch.half_width - 1.0))
        foot, depth = cam.project(np.array([[xm, ym]]))
        if depth[0] <= 0 or not np.isfinite(foot).all():
            continue
        xi = int(round(float(foot[0, 0]) - bw / 2.0))
        yi = int(round(float(foot[0, 1]) - bh))
        if xi < 0 or yi < 0 or xi + bw > cam.width or yi + bh > cam.height:
            continue
        out[i] = (xi, yi, xi + bw, yi + bh)
    return out


def _overlaps(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    """True if the two per-frame box tracks overlap (with ``margin`` px) in any frame."""
    both = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not both.any():
        return False
    a, b = a[both], b[both]
    sep = (a[:, 2] + margin <= b[:, 0]) | (b[:, 2] + margin <= a[:, 0]) | \
          (a[:, 3] + margin <= b[:, 1]) | (b[:, 3] + margin <= a[:, 1])
    return bool((~sep).any())


def _actor_positions(rng: np.random.Generator, n_per_team: int, with_referee: bool, pitch: PitchModel,
                     cams: Sequence[BroadcastCamera], bw: int, bh: int,
                     speed_m_per_frame: Tuple[float, float] = (0.05, 0.25), margin_px: float = 6.0,
                     max_tries: int = 400) -> List[tuple]:
    """Actors in pitch metres: (role, team, colour, start xy, velocity).

    Team ``left`` lives in ``x < 0`` and ``right`` in ``x > 0`` and nobody crosses
    the halfway line within the clip, so the pipeline's placeholder left/right
    naming by mean image x is well defined. Placements are rejection-sampled so
    that no two rectangles overlap in any frame of the clip (the smoke detector
    is a connected-components stand-in and would merge them).
    """
    out: List[tuple] = []
    tracks: List[np.ndarray] = []
    L, W = pitch.half_length, pitch.half_width

    def place(role, team, colour, lo_hi, fixed=None) -> None:
        for _ in range(max_tries):
            if fixed is not None:
                (x0, y0), (vx, vy) = fixed
            else:
                x0 = rng.uniform(*lo_hi)
                y0 = rng.uniform(-W + 6.0, W - 6.0)
                s = rng.uniform(*speed_m_per_frame)
                ang = rng.uniform(0, 2 * math.pi)
                vx, vy = s * math.cos(ang), s * math.sin(ang)
            tr = _box_track(cams, x0, y0, vx, vy, bw, bh, pitch)
            if fixed is None and not np.isfinite(tr).all():
                continue  # keep randomly placed actors in frame for the whole clip when possible
            if all(not _overlaps(tr, t, margin_px) for t in tracks):
                out.append((role, team, colour, (x0, y0), (vx, vy)))
                tracks.append(tr)
                return
            if fixed is not None:
                fixed = ((fixed[0][0] + rng.uniform(-3, 3), fixed[0][1] + rng.uniform(-3, 3)), fixed[1])
        # last resort: accept the final candidate (only reachable with a tight camera)
        out.append((role, team, colour, (x0, y0), (vx, vy)))
        tracks.append(tr)

    if with_referee:
        place("referee", None, REFEREE_BGR, None, fixed=((0.0, 4.0), (-0.03, 0.02)))
    for team, lo_hi in (("left", (-L + 8.0, -12.0)), ("right", (12.0, L - 8.0))):
        for _ in range(n_per_team):
            place("player", team, TEAM_COLOURS_BGR[team], lo_hi)
    return out


def render_synthetic_clip(out_dir: Path | str, n_frames: int = 30, width: int = 640, height: int = 360,
                          n_per_team: int = 3, with_referee: bool = True, seed: int = 0,
                          name: str = "SYN-001", game_id: str = "syn-game-1",
                          box_wh: Tuple[int, int] = (24, 56), with_jerseys: bool = False,
                          camera: Optional[BroadcastCamera] = None, pan_m_per_frame: float = 0.0,
                          with_markings: bool = True, line_thickness: int = 3,
                          cut_at_frame: Optional[int] = None, camera_after_cut: Optional[BroadcastCamera] = None,
                          green_after_cut: Optional[Tuple[int, int, int]] = None,
                          pitch: PitchModel = DEFAULT_PITCH, n_frames_total: Optional[int] = None) -> SyntheticClip:
    """Write ``<out_dir>/<name>/img1/000001.jpg ...`` and the ground-truth labels.

    Actors live in pitch metres and are drawn as fixed-size rectangles whose
    bottom-centre is the actor's projected foot point; ``bbox_pitch`` in the
    labels is the projection of the rectangle's bottom edge through the frame's
    true pixel -> metre homography, exactly what the pipeline computes from a
    calibrated frame. Jersey numbers are null unless ``with_jerseys`` (GS-HOTA
    needs them to match, and the pipeline has no jersey recognition).

    ``camera`` defaults to :func:`wide_camera`; ``pan_m_per_frame`` moves its
    look-at point along x every frame; ``cut_at_frame`` switches to
    ``camera_after_cut`` (default: a :func:`medium_camera`) from that frame on,
    with ``green_after_cut`` as the new grass colour when given, which is how a
    hard cut looks to a frame-difference detector. ``with_markings=False``
    renders plain grass, the "uncalibrated" case.
    """
    import cv2

    rng = np.random.default_rng(seed)
    clip_dir = Path(out_dir) / name
    img_dir = clip_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    bw, bh = box_wh
    cam0 = camera if camera is not None else wide_camera(width, height, pitch)
    cam_cut = camera_after_cut if camera_after_cut is not None else medium_camera(width, height, pitch=pitch)
    cams_all: List[BroadcastCamera] = []
    for f in range(1, n_frames + 1):
        if cut_at_frame is not None and f >= cut_at_frame:
            cams_all.append(cam_cut.panned(pan_m_per_frame * (f - cut_at_frame)))
        else:
            cams_all.append(cam0.panned(pan_m_per_frame * (f - 1)))
    actors = _actor_positions(rng, n_per_team, with_referee, pitch, cams_all, bw, bh)

    cats = parse_categories(default_categories_list())
    images: List[ImageRecord] = []
    objects: List[ObjectAnnotation] = []
    homographies: List[np.ndarray] = []
    cameras: List[BroadcastCamera] = []
    clip_id = "9001"
    for f in range(1, n_frames + 1):
        after_cut = cut_at_frame is not None and f >= cut_at_frame
        cam = cams_all[f - 1]
        green = (green_after_cut if (after_cut and green_after_cut is not None) else GREEN_BGR)
        h_px2m = cam.homography_px2m()
        homographies.append(h_px2m)
        cameras.append(cam)

        frame = np.full((height, width, 3), green, dtype=np.uint8)
        if with_markings:
            draw_markings(frame, cam, pitch, thickness=line_thickness)
        image_id = new_image_id(clip_id, f)
        images.append(ImageRecord(image_id=image_id, file_name=frame_file_name(f), width=width, height=height,
                                  is_labeled=True, has_labeled_person=True, has_labeled_pitch=with_markings,
                                  has_labeled_camera=True))
        for t, (role, team, colour, (x0, y0), (vx, vy)) in enumerate(actors, start=1):
            xm = float(np.clip(x0 + vx * (f - 1), -pitch.half_length + 1.0, pitch.half_length - 1.0))
            ym = float(np.clip(y0 + vy * (f - 1), -pitch.half_width + 1.0, pitch.half_width - 1.0))
            foot, depth = cam.project(np.array([[xm, ym]]))
            if depth[0] <= 0 or not np.isfinite(foot).all():
                continue
            fx, fy = float(foot[0, 0]), float(foot[0, 1])
            xi = int(round(fx - bw / 2.0))
            yi = int(round(fy - bh))
            if xi < 0 or yi < 0 or xi + bw > width or yi + bh > height:
                continue  # out of frame: no box, no label (a wide camera keeps everyone in frame)
            cv2.rectangle(frame, (xi, yi), (xi + bw - 1, yi + bh - 1), colour, thickness=-1)
            box = BBoxImage(x=float(xi), y=float(yi), w=float(bw), h=float(bh))
            bl, br = project(h_px2m, np.array([[box.x, box.y + box.h], [box.x + box.w, box.y + box.h]]))
            jersey = str(t) if (with_jerseys and role == "player") else None
            objects.append(
                ObjectAnnotation(
                    id=new_annotation_id(image_id, t),
                    image_id=image_id,
                    track_id=t,
                    category_id=category_id_for_role(role, cats),
                    role=role,
                    team=team,
                    jersey=jersey,
                    bbox_image=box,
                    bbox_pitch=BBoxPitch.from_points((bl[0], bl[1]), (br[0], br[1])),
                )
            )
        ok = cv2.imwrite(str(img_dir / frame_file_name(f)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise IOError(f"could not write frame {f}")

    info = ClipInfo(name=name, game_id=game_id, clip_id=clip_id, version="1.3", frame_rate=25,
                    seq_length=n_frames, im_dir="img1", im_ext=".jpg",
                    extra={"synthetic": True,
                           "note": "coloured rectangles on a rendered pitch; code-path test only",
                           "camera": {"position": list(cam0.position), "look_at": list(cam0.look_at),
                                      "hfov_deg": cam0.hfov_deg, "pan_m_per_frame": pan_m_per_frame,
                                      "cut_at_frame": cut_at_frame, "markings": with_markings}})
    clip = GSRClip(path=clip_dir / LABELS_FILE, info=info, images=images, objects=objects, categories=cats)
    write_labels_json(clip)
    return SyntheticClip(clip_dir=clip_dir, labels_path=clip.path, width=width, height=height,
                         n_frames=n_frames, homography=homographies[0], homographies=homographies,
                         cameras=cameras, cut_at_frame=cut_at_frame)
