"""Detection -> tracking -> team assignment -> optional pose, frames in, GSR JSON out.

Backends are swappable so the same code path runs in two configurations:

* Production (Colab L4): RF-DETR (``rfdetr``, Apache-2.0) for people and the
  ball, ByteTrack from Roboflow's ``trackers`` (Apache-2.0; ``supervision`` 0.31
  no longer ships ByteTrack, it only supplies the ``Detections`` container),
  SigLIP image embeddings (``transformers``) clustered with a small k-means for
  the two teams, and RTMPose via ``rtmlib`` (ONNX Runtime) for 2D keypoints.
* Smoke (any CPU, no weights): colour-blob detector, the same ByteTrack or a
  greedy IoU tracker when ``trackers`` is absent, mean-colour embeddings and the
  same k-means. Only meant to exercise the code path on the synthetic clip in
  ``synthetic.py``; it says nothing about accuracy.

Heavy imports happen inside the backend constructors, so importing this module
needs only numpy and OpenCV.

Output: one ``Labels-GameState.json`` per clip in the GSR layout documented in
``gsr_format.py`` (per-frame image records and per-object annotations with
``track_id``, ``attributes`` and ``bbox_image``; ``bbox_pitch`` is filled when a
pitch projector is given, else null). Keypoints, when computed, go under the
extra key ``keypoints`` (COCO-17 ``[x, y, score]`` per joint); the loader keeps
unknown keys in ``ObjectAnnotation.extra``.

Pitch coordinates come from one of two sources: a static ``PitchProjector``
(a known homography, used by the synthetic smoke test) or a per-frame
``FrameCalibrator`` from ``calibration.py`` that finds the markings (fed through
``push`` / ``flush``, so its frame stride applies: during steady tracking most
frames are interpolated between fitted ones). With a calibrator each object also
carries ``pitch_x`` / ``pitch_y`` (metres, bottom-middle; null when the frame is
uncalibrated), ``calib_error`` (the frame's RMS line-fit error in pixels; null
when unknown, which includes every interpolated frame) and ``calib_status``
(``calibrated`` / ``tracked`` / ``interpolated`` / ``uncalibrated``) in its extra keys, and every frame gets a ``camera`` annotation record
(``calibration``: homography and diagnostics). ``bbox_pitch`` is null on every
frame without a calibration and on any box whose projection is not finite, so
the JSON never contains NaN or Infinity. ``has_labeled_camera`` on the image
record is true only when that frame's calibration is usable.

Known placeholders, to be replaced in later M4 work: every tracked person is
written with role ``player`` (no referee / goalkeeper classifier yet); the two
colour clusters are named ``left`` / ``right`` by their mean image x, not by
pitch side; jersey numbers are always null.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .gsr_format import (
    LABELS_FILE,
    ROLE_BALL,
    ROLE_PLAYER,
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
from .loaders import FrameSequence, list_frames

if TYPE_CHECKING:  # typing only; calibration imports scipy, which the smoke path does not need
    from .calibration import CalibrationResult, FrameCalibrator  # noqa: F401

CLASS_PERSON = "person"
CLASS_BALL = "ball"
BALL_TRACK_ID = 0  # single ball track; people get ids >= 1
UNASSIGNED = -1
CAMERA_ANNOTATION_INDEX = 9000  # per-frame camera record id = image_id + this index, past any object index

# COCO ids as used by rfdetr.assets.coco_classes (91-id scheme).
COCO_PERSON_ID = 1
COCO_SPORTS_BALL_ID = 37


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------


@dataclass
class FrameDetections:
    xyxy: np.ndarray                       # (N, 4) float, pixels
    confidence: np.ndarray                 # (N,) float
    class_name: np.ndarray                 # (N,) str: "person" | "ball"
    tracker_id: Optional[np.ndarray] = None      # (N,) int, UNASSIGNED when none
    team: Optional[np.ndarray] = None            # (N,) object: "left" | "right" | None
    keypoints: Optional[np.ndarray] = None       # (N, K, 2)
    keypoint_scores: Optional[np.ndarray] = None  # (N, K)

    @classmethod
    def empty(cls) -> "FrameDetections":
        return cls(xyxy=np.zeros((0, 4), float), confidence=np.zeros(0, float),
                   class_name=np.zeros(0, dtype=object))

    def __len__(self) -> int:
        return int(len(self.xyxy))

    def select(self, mask: np.ndarray) -> "FrameDetections":
        def pick(a):
            return None if a is None else a[mask]
        return FrameDetections(xyxy=self.xyxy[mask], confidence=self.confidence[mask],
                               class_name=self.class_name[mask], tracker_id=pick(self.tracker_id),
                               team=pick(self.team), keypoints=pick(self.keypoints),
                               keypoint_scores=pick(self.keypoint_scores))


def crop(frame: np.ndarray, xyxy: Sequence[float], pad: int = 0) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(math.floor(x1)) - pad); y1 = max(0, int(math.floor(y1)) - pad)
    x2 = min(w, int(math.ceil(x2)) + pad); y2 = min(h, int(math.ceil(y2)) + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


# ----------------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------------


class Detector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections: ...


class Tracker(Protocol):
    name: str

    def update(self, dets: FrameDetections, frame_bgr: Optional[np.ndarray] = None) -> FrameDetections: ...

    def reset(self) -> None: ...


class CropEmbedder(Protocol):
    name: str

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray: ...


class PoseEstimator(Protocol):
    name: str

    def estimate(self, frame_bgr: np.ndarray, xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...


class PitchProjector(Protocol):
    name: str

    def project(self, points_px: np.ndarray) -> np.ndarray: ...


# ----------------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------------


class ColorBlobDetector:
    """Saturated or dark connected components on grass; the smoke-test stand-in for a
    person detector. Green (the pitch) and bright low-saturation pixels (the white
    markings) are background, so the rendered lines do not merge the rectangles."""

    name = "color-blob"

    def __init__(self, min_area: int = 80, max_area: Optional[int] = None,
                 green_hue: Tuple[int, int] = (35, 85), min_sat: int = 60,
                 white_max_sat: int = 70, white_min_val: int = 150, open_px: int = 3):
        self.min_area = min_area
        self.max_area = max_area
        self.green_hue = green_hue
        self.min_sat = min_sat
        self.white_max_sat = white_max_sat
        self.white_min_val = white_min_val
        self.open_px = open_px

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections:
        import cv2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        is_green = (h >= self.green_hue[0]) & (h <= self.green_hue[1]) & (s >= self.min_sat)
        is_white = (s <= self.white_max_sat) & (v >= self.white_min_val)
        fg = (~(is_green | is_white)).astype(np.uint8)
        if self.open_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.open_px, self.open_px))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        boxes, conf = [], []
        for k in range(1, n):
            x, y, w, hh, area = stats[k]
            if area < self.min_area or (self.max_area is not None and area > self.max_area):
                continue
            boxes.append([x, y, x + w, y + hh])
            conf.append(min(1.0, area / (w * hh + 1e-9)))
        if not boxes:
            return FrameDetections.empty()
        return FrameDetections(xyxy=np.array(boxes, float), confidence=np.array(conf, float),
                               class_name=np.array([CLASS_PERSON] * len(boxes), dtype=object))


class RFDETRDetector:
    """RF-DETR (Apache-2.0 sizes only: nano / small / medium / large), COCO person + sports ball.

    Weights are fetched by the package itself from Roboflow's official bucket
    (``https://storage.googleapis.com/rfdetr/``) into ``~/.roboflow/models`` and
    checked against the md5 shipped in the pinned ``rfdetr`` version.
    """

    name = "rf-detr"
    SIZES = ("nano", "small", "medium", "large")

    def __init__(self, size: str = "medium", threshold: float = 0.5, device: Optional[str] = None,
                 keep_ball: bool = True):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of {self.SIZES} (the Apache-2.0 models), got {size!r}")
        import rfdetr

        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
               "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[size]
        kw: Dict[str, Any] = {}
        if device is not None:
            kw["device"] = device
        self.model = cls(**kw)
        self.threshold = threshold
        self.keep_ball = keep_ball
        self.size = size
        self.name = f"rf-detr-{size}"

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=self.threshold, include_source_image=False)
        if det is None or len(det) == 0:
            return FrameDetections.empty()
        cls_id = np.asarray(det.class_id)
        is_person = cls_id == COCO_PERSON_ID
        keep = is_person | (cls_id == COCO_SPORTS_BALL_ID) if self.keep_ball else is_person
        if not keep.any():
            return FrameDetections.empty()
        names = np.where(is_person, CLASS_PERSON, CLASS_BALL).astype(object)
        return FrameDetections(xyxy=np.asarray(det.xyxy, float)[keep],
                               confidence=np.asarray(det.confidence, float)[keep],
                               class_name=names[keep])


# ----------------------------------------------------------------------------
# Trackers
# ----------------------------------------------------------------------------


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _with_ids(dets: FrameDetections, ids: np.ndarray) -> FrameDetections:
    return FrameDetections(xyxy=dets.xyxy, confidence=dets.confidence, class_name=dets.class_name,
                           tracker_id=ids, team=dets.team, keypoints=dets.keypoints,
                           keypoint_scores=dets.keypoint_scores)


class GreedyIoUTracker:
    """Hungarian IoU association with a lost-track buffer; the no-dependency fallback."""

    name = "greedy-iou"

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.reset()

    def reset(self) -> None:
        self._next_id = 1
        self._tracks: Dict[int, Tuple[np.ndarray, int]] = {}  # id -> (xyxy, frames lost)

    def update(self, dets: FrameDetections, frame_bgr: Optional[np.ndarray] = None) -> FrameDetections:
        from scipy.optimize import linear_sum_assignment

        ids = np.full(len(dets), UNASSIGNED, dtype=int)
        person = np.asarray(dets.class_name) == CLASS_PERSON
        p_idx = np.where(person)[0]
        boxes = dets.xyxy[p_idx].reshape(-1, 4)
        track_ids = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[t][0] for t in track_ids]).reshape(-1, 4)
        matched_t, matched_d = set(), set()
        if len(track_ids) and len(boxes):
            iou = _iou_matrix(track_boxes, boxes)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= self.iou_threshold:
                    ids[p_idx[j]] = track_ids[i]
                    self._tracks[track_ids[i]] = (boxes[j], 0)
                    matched_t.add(track_ids[i]); matched_d.add(j)
        for t in track_ids:
            if t not in matched_t:
                box, lost = self._tracks[t]
                if lost + 1 > self.max_lost:
                    del self._tracks[t]
                else:
                    self._tracks[t] = (box, lost + 1)
        for j in range(len(boxes)):
            if j not in matched_d:
                ids[p_idx[j]] = self._next_id
                self._tracks[self._next_id] = (boxes[j], 0)
                self._next_id += 1
        ids[np.asarray(dets.class_name) == CLASS_BALL] = BALL_TRACK_ID
        return _with_ids(dets, ids)


class ByteTrackTracker:
    """ByteTrack from the ``trackers`` package (Apache-2.0), on person boxes only.

    Ids are remapped to start at 1; the ball keeps ``BALL_TRACK_ID``. Boxes the
    tracker has not confirmed yet come back with ``UNASSIGNED``.
    """

    name = "bytetrack"

    def __init__(self, frame_rate: float = 25.0, track_activation_threshold: float = 0.5,
                 lost_track_buffer: int = 30, minimum_consecutive_frames: int = 1,
                 minimum_iou_threshold: float = 0.1, high_conf_det_threshold: float = 0.6):
        import supervision as sv  # noqa: F401  (Detections container)
        import trackers

        self._kw = dict(frame_rate=frame_rate, track_activation_threshold=track_activation_threshold,
                        lost_track_buffer=lost_track_buffer, minimum_consecutive_frames=minimum_consecutive_frames,
                        minimum_iou_threshold=minimum_iou_threshold, high_conf_det_threshold=high_conf_det_threshold)
        self._cls = trackers.ByteTrackTracker
        self.reset()

    def reset(self) -> None:
        self._tracker = self._cls(**self._kw)

    def update(self, dets: FrameDetections, frame_bgr: Optional[np.ndarray] = None) -> FrameDetections:
        import supervision as sv

        ids = np.full(len(dets), UNASSIGNED, dtype=int)
        person = np.asarray(dets.class_name) == CLASS_PERSON
        p_idx = np.where(person)[0]
        sv_in = sv.Detections(xyxy=np.asarray(dets.xyxy[p_idx], float).reshape(-1, 4),
                              confidence=np.asarray(dets.confidence[p_idx], float),
                              class_id=np.zeros(len(p_idx), dtype=int))
        out = self._tracker.update(sv_in)  # ByteTrack ignores the frame; passing it only warns
        # trackers returns the boxes it was given (possibly a subset) with tracker_id; map back by box value.
        if len(out) and out.tracker_id is not None:
            key = {tuple(np.round(b, 3)): i for i, b in enumerate(np.asarray(sv_in.xyxy))}
            for b, tid in zip(np.asarray(out.xyxy), np.asarray(out.tracker_id)):
                i = key.get(tuple(np.round(b, 3)))
                if i is not None and tid is not None and int(tid) >= 0:
                    ids[p_idx[i]] = int(tid) + 1
        ids[np.asarray(dets.class_name) == CLASS_BALL] = BALL_TRACK_ID
        return _with_ids(dets, ids)


def default_tracker(frame_rate: float = 25.0) -> Tracker:
    """ByteTrack when ``trackers`` is installed, else the greedy IoU fallback."""
    try:
        return ByteTrackTracker(frame_rate=frame_rate)
    except ImportError:
        return GreedyIoUTracker()


# ----------------------------------------------------------------------------
# Team assignment: crop embeddings + k-means, majority vote per track
# ----------------------------------------------------------------------------


class MeanColorEmbedder:
    """Mean Lab colour of the torso region (upper-middle of the crop); 3-D feature."""

    name = "mean-color"

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        feats = []
        for c in crops_bgr:
            h, w = c.shape[:2]
            y0, y1 = int(0.15 * h), max(int(0.15 * h) + 1, int(0.55 * h))
            x0, x1 = int(0.2 * w), max(int(0.2 * w) + 1, int(0.8 * w))
            torso = c[y0:y1, x0:x1]
            lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
            feats.append(lab.mean(axis=0))
        return np.array(feats, float).reshape(-1, 3)


class SigLIPEmbedder:
    """SigLIP vision tower (``google/siglip-base-patch16-224``, Apache-2.0) pooled embeddings.

    ``revision`` pins the Hugging Face commit; weights come from huggingface.co
    into the standard HF cache (token, if any, from ``HF_TOKEN`` in the env).
    """

    name = "siglip"
    DEFAULT_MODEL = "google/siglip-base-patch16-224"
    DEFAULT_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"  # main on 2026-10-17 (last modified 2024-09-26)

    def __init__(self, model_id: str = DEFAULT_MODEL, revision: Optional[str] = DEFAULT_REVISION,
                 device: Optional[str] = None, batch_size: int = 64):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = SiglipVisionModel.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.batch_size = batch_size

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image

        out = []
        with torch.inference_mode():
            for i in range(0, len(crops_bgr), self.batch_size):
                batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr[i:i + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                emb = self.model(**inputs).pooler_output
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), float)


class KMeansTeams:
    """Tiny deterministic k-means (numpy) with a few restarts; k=2 for the two teams."""

    def __init__(self, k: int = 2, n_init: int = 5, max_iter: int = 100, seed: int = 0):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed
        self.centers: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "KMeansTeams":
        x = np.asarray(x, float)
        if len(x) < self.k:
            raise ValueError(f"need at least {self.k} samples, got {len(x)}")
        rng = np.random.default_rng(self.seed)
        best, best_inertia = None, np.inf
        for _ in range(self.n_init):
            centers = x[rng.choice(len(x), self.k, replace=False)]
            for _it in range(self.max_iter):
                d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                lab = d.argmin(axis=1)
                new = np.array([x[lab == j].mean(axis=0) if np.any(lab == j) else centers[j] for j in range(self.k)])
                if np.allclose(new, centers):
                    break
                centers = new
            d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            inertia = float(((x - centers[lab]) ** 2).sum())
            if inertia < best_inertia:
                best, best_inertia = centers.copy(), inertia
        self.centers = best
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("fit first")
        d = ((np.asarray(x, float)[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


def assign_teams_by_track(track_ids: np.ndarray, cluster: np.ndarray, mean_x: np.ndarray,
                          names: Tuple[str, str] = ("left", "right")) -> Dict[int, str]:
    """Majority cluster per track, then name the cluster with the smaller mean image x ``left``.

    The left/right naming is a placeholder until pitch calibration gives the real side.
    """
    track_ids = np.asarray(track_ids); cluster = np.asarray(cluster); mean_x = np.asarray(mean_x, float)
    per_track: Dict[int, int] = {}
    for t in np.unique(track_ids):
        votes = cluster[track_ids == t]
        per_track[int(t)] = int(np.bincount(votes, minlength=2).argmax())
    xs = {j: (mean_x[cluster == j].mean() if np.any(cluster == j) else np.inf) for j in (0, 1)}
    order = sorted((0, 1), key=lambda j: xs[j])
    name_of = {order[0]: names[0], order[1]: names[1]}
    return {t: name_of[c] for t, c in per_track.items()}


# ----------------------------------------------------------------------------
# Pose
# ----------------------------------------------------------------------------


class RTMPoseEstimator:
    """RTMPose top-down (COCO-17) through ``rtmlib`` (Apache-2.0), ONNX Runtime backend.

    Model zips are fetched by rtmlib from ``download.openmmlab.com`` (official
    OpenMMLab host, Hugging Face mirror on failure) into ``~/.cache/rtmlib``.
    """

    name = "rtmpose"
    FALLBACK_URLS = {
        "balanced": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                     "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip", (192, 256)),
        "lightweight": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip", (192, 256)),
    }

    def __init__(self, mode: str = "balanced", device: str = "cpu", backend: str = "onnxruntime"):
        from rtmlib import RTMPose

        try:
            from rtmlib.tools.solution.body import Body

            url = Body.MODE[mode]["pose"]
            size = Body.MODE[mode]["pose_input_size"]
        except Exception:  # noqa: BLE001  (older/newer rtmlib layouts)
            url, size = self.FALLBACK_URLS[mode]
        self.model = RTMPose(onnx_model=url, model_input_size=tuple(size), backend=backend, device=device)
        self.mode = mode
        self.url = url

    def estimate(self, frame_bgr: np.ndarray, xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(xyxy) == 0:
            return np.zeros((0, 17, 2), float), np.zeros((0, 17), float)
        kps, scores = self.model(frame_bgr, bboxes=[list(map(float, b)) for b in xyxy])
        return np.asarray(kps, float), np.asarray(scores, float)


# ----------------------------------------------------------------------------
# Pitch projection
# ----------------------------------------------------------------------------


class HomographyProjector:
    name = "homography"

    def __init__(self, h: np.ndarray):
        self.h = np.asarray(h, float).reshape(3, 3)

    def project(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, float).reshape(-1, 2)
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ self.h.T
        return hom[:, :2] / hom[:, 2:3]


# ----------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------


@dataclass
class StageTimer:
    seconds: float = 0.0
    items: int = 0

    def add(self, dt: float, n: int = 1) -> None:
        self.seconds += dt
        self.items += n

    @property
    def per_second(self) -> float:
        return self.items / self.seconds if self.seconds > 0 else float("nan")


@dataclass
class RunReport:
    clip_name: str
    n_frames: int
    n_detections: int
    n_tracks: int
    out_path: Path
    stages: Dict[str, StageTimer] = field(default_factory=dict)
    backends: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    calibration: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clip": self.clip_name,
            "frames": self.n_frames,
            "detections": self.n_detections,
            "tracks": self.n_tracks,
            "out": str(self.out_path),
            "backends": dict(self.backends),
            "stages": {k: {"seconds": round(v.seconds, 4), "items": v.items,
                           "per_second": (round(v.per_second, 2) if v.seconds > 0 else None)}
                       for k, v in self.stages.items()},
            "notes": list(self.notes),
            "calibration": dict(self.calibration),
        }

    def summary(self) -> str:
        parts = [f"{self.clip_name}: {self.n_frames} frames, {self.n_detections} detections, {self.n_tracks} tracks"]
        for k, v in self.stages.items():
            unit = "fps" if k in ("read", "detect", "track", "pose", "calib", "total") else "items/s"
            parts.append(f"  {k:<8} {v.seconds:8.3f} s  {v.per_second:8.2f} {unit}")
        return "\n".join(parts)


class VisionPipeline:
    def __init__(self, detector: Detector, tracker: Optional[Tracker] = None,
                 embedder: Optional[CropEmbedder] = None, pose: Optional[PoseEstimator] = None,
                 projector: Optional[PitchProjector] = None, frame_rate: float = 25.0,
                 max_crops_per_track: int = 20, team_k: int = 2, seed: int = 0,
                 calibrator: Optional["FrameCalibrator"] = None):
        self.detector = detector
        self.tracker = tracker if tracker is not None else default_tracker(frame_rate)
        self.embedder = embedder
        self.pose = pose
        self.projector = projector
        self.calibrator = calibrator  # per-frame line calibration; overrides ``projector`` when set
        self.frame_rate = frame_rate
        self.max_crops_per_track = max_crops_per_track
        self.team_k = team_k
        self.seed = seed

    def backends(self) -> Dict[str, str]:
        return {
            "detector": getattr(self.detector, "name", type(self.detector).__name__),
            "tracker": getattr(self.tracker, "name", type(self.tracker).__name__),
            "embedder": getattr(self.embedder, "name", "none") if self.embedder else "none",
            "pose": getattr(self.pose, "name", "none") if self.pose else "none",
            "projector": (getattr(self.calibrator, "name", "line-calibration") if self.calibrator is not None
                          else (getattr(self.projector, "name", "none") if self.projector is not None else "none")),
        }

    # -- per-frame work -----------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, timers: Dict[str, StageTimer]) -> FrameDetections:
        t0 = time.perf_counter()
        dets = self.detector.detect(frame_bgr)
        timers["detect"].add(time.perf_counter() - t0, 1)
        t0 = time.perf_counter()
        dets = self.tracker.update(dets, frame_bgr)
        timers["track"].add(time.perf_counter() - t0, 1)
        if self.pose is not None:
            t0 = time.perf_counter()
            person = np.asarray(dets.class_name) == CLASS_PERSON
            kps = np.full((len(dets), 17, 2), np.nan); sc = np.zeros((len(dets), 17))
            if person.any():
                k, s = self.pose.estimate(frame_bgr, dets.xyxy[person])
                if k.ndim == 3 and k.shape[1] == 17:
                    kps[person] = k; sc[person] = s
            dets.keypoints, dets.keypoint_scores = kps, sc
            timers["pose"].add(time.perf_counter() - t0, 1)
        return dets

    def _calibrate(self, frame_bgr: Optional[np.ndarray]) -> List["CalibrationResult"]:
        """One calibrator step: the results that became final with this frame, in frame order (none
        while a ``FrameCalibrator`` holds the frame back, several when its held stretch closes; a
        calibrator with only ``update`` gives one per frame). ``None`` flushes what is still held at
        the end of the clip. When the calibrator chokes (a singular homography on a degenerate view,
        say), this frame and every frame it was still holding are reported uncalibrated with the
        reason in ``note`` and the calibrator is reset, rather than aborting a clip that has already
        cost minutes of detector time."""
        push = getattr(self.calibrator, "push", None)
        held = int(getattr(self.calibrator, "n_pending", 0)) if push is not None else 0
        try:
            if frame_bgr is None:
                return self.calibrator.flush() if push is not None else []
            return push(frame_bgr) if push is not None else [self.calibrator.update(frame_bgr)]
        except Exception as e:  # noqa: BLE001
            from .calibration import STATUS_UNCALIBRATED, CalibrationResult

            self.calibrator.reset()
            n = held + (frame_bgr is not None)
            return [CalibrationResult(status=STATUS_UNCALIBRATED, note=f"calibrator raised {type(e).__name__}: {e}")
                    for _ in range(n)]

    # -- whole clip ---------------------------------------------------------
    def run_clip(self, clip_dir: Path | str, out_path: Path | str, clip_name: Optional[str] = None,
                 game_id: str = "", frames: Optional[FrameSequence] = None,
                 progress: Optional[Callable[[int, int], None]] = None) -> RunReport:
        import cv2

        clip_dir = Path(clip_dir)
        seq = frames if frames is not None else list_frames(clip_dir, None, strict=False)
        name = clip_name or clip_dir.name
        timers = {k: StageTimer() for k in ("read", "detect", "track", "pose", "calib", "team", "write", "total")}
        self.tracker.reset()
        if self.calibrator is not None:
            self.calibrator.reset()
        t_all = time.perf_counter()

        per_frame: List[Tuple[int, int, int, FrameDetections]] = []  # (frame_idx, w, h, dets)
        calib: List["CalibrationResult"] = []
        crops: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}  # track -> [(frame, crop, cx)]
        n = len(seq)
        stride = max(1, n // self.max_crops_per_track)
        for k, (fidx, path) in enumerate(seq):
            t0 = time.perf_counter()
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise IOError(f"could not read {path}")
            timers["read"].add(time.perf_counter() - t0, 1)
            dets = self.process_frame(frame, timers)
            h, w = frame.shape[:2]
            per_frame.append((fidx, w, h, dets))
            if self.calibrator is not None:
                t0 = time.perf_counter()
                calib.extend(self._calibrate(frame))
                timers["calib"].add(time.perf_counter() - t0, 1)
            if self.embedder is not None and dets.tracker_id is not None and k % stride == 0:
                for i in range(len(dets)):
                    tid = int(dets.tracker_id[i])
                    if tid <= 0 or dets.class_name[i] != CLASS_PERSON:
                        continue
                    lst = crops.setdefault(tid, [])
                    if len(lst) < self.max_crops_per_track:
                        cx = float((dets.xyxy[i, 0] + dets.xyxy[i, 2]) / 2)
                        lst.append((fidx, crop(frame, dets.xyxy[i]).copy(), cx))
            if progress is not None:
                progress(k + 1, n)

        if self.calibrator is not None:  # the frames the calibrator still holds back
            t0 = time.perf_counter()
            calib.extend(self._calibrate(None))
            timers["calib"].add(time.perf_counter() - t0, 0)

        # Team assignment over the whole clip.
        team_of: Dict[int, str] = {}
        notes: List[str] = []
        if self.embedder is not None:
            t0 = time.perf_counter()
            tids, imgs, cxs = [], [], []
            for tid, lst in crops.items():
                for _fidx, c, cx in lst:
                    tids.append(tid); imgs.append(c); cxs.append(cx)
            if len(imgs) >= self.team_k and len(set(tids)) >= self.team_k:
                feats = self.embedder.embed(imgs)
                km = KMeansTeams(k=self.team_k, seed=self.seed).fit(feats)
                team_of = assign_teams_by_track(np.array(tids), km.predict(feats), np.array(cxs))
            else:
                notes.append(f"team assignment skipped: {len(set(tids))} tracks, {len(imgs)} crops")
            timers["team"].add(time.perf_counter() - t0, len(imgs))

        # Write the GSR-layout JSON.
        t0 = time.perf_counter()
        clip = self._to_gsr(name, game_id, per_frame, team_of, calib if self.calibrator is not None else None)
        out_path = Path(out_path)
        write_labels_json(clip, out_path)
        timers["write"].add(time.perf_counter() - t0, 1)
        timers["total"].add(time.perf_counter() - t_all, n)
        n_det = sum(len(d) for _, _, _, d in per_frame)
        tracks = {int(t) for _, _, _, d in per_frame if d.tracker_id is not None for t in d.tracker_id if t > 0}
        summary = _calibration_summary(calib) if self.calibrator is not None else {}
        n_unc = int(summary.get("uncalibrated", 0))
        if n_unc and n_unc == n:
            reason = next((r.note for r in calib if r is not None and r.note), "")
            notes.append(f"calibration: all {n} frames uncalibrated, no pitch coordinates written"
                         + (f" ({reason})" if reason else ""))
        elif n_unc:
            notes.append(f"calibration: {n_unc} of {n} frames uncalibrated")
        return RunReport(clip_name=name, n_frames=n, n_detections=n_det, n_tracks=len(tracks), out_path=out_path,
                         stages=timers, backends=self.backends(), notes=notes, calibration=summary)

    def _to_gsr(self, name: str, game_id: str, per_frame, team_of: Dict[int, str],
                calib: Optional[List["CalibrationResult"]] = None) -> GSRClip:
        """Build the clip. ``calib`` is None without a calibrator, else one result per frame, in order."""
        if calib is not None and len(calib) != len(per_frame):
            raise ValueError(f"{len(calib)} calibration results for {len(per_frame)} frames")
        cats = parse_categories(default_categories_list())
        clip_id = "".join(ch for ch in name if ch.isdigit()) or "0"
        images: List[ImageRecord] = []
        objects: List[ObjectAnnotation] = []
        cameras: List[Dict[str, Any]] = []
        camera_cid = category_id_for_role("camera", cats)
        for k, (fidx, w, h, dets) in enumerate(per_frame):
            image_id = new_image_id(clip_id, fidx)
            res = calib[k] if calib is not None else None
            images.append(ImageRecord(image_id=image_id, file_name=frame_file_name(fidx), width=w, height=h,
                                      is_labeled=True, has_labeled_person=True,
                                      has_labeled_camera=bool(res is not None and res.ok)))
            if res is not None:
                cameras.append({"id": new_annotation_id(image_id, CAMERA_ANNOTATION_INDEX), "image_id": image_id,
                                "supercategory": "camera", "category_id": camera_cid, "calibration": res.as_dict()})
            if len(dets) == 0:
                continue
            lefts = np.stack([dets.xyxy[:, 0], dets.xyxy[:, 3]], axis=1)
            rights = np.stack([dets.xyxy[:, 2], dets.xyxy[:, 3]], axis=1)
            # Bottom corners in metres, (N, 2) each; None when there is no source for them (an
            # uncalibrated frame, or no calibrator and no static projector). With a calibrator the
            # static projector is never consulted: one clip must not mix two coordinate sources.
            pl = pr = None
            if res is not None:
                if res.ok:
                    pl, pr = res.project(lefts), res.project(rights)
            elif self.projector is not None:
                pl, pr = self.projector.project(lefts), self.projector.project(rights)
            for i in range(len(dets)):
                x1, y1, x2, y2 = map(float, dets.xyxy[i])
                tid = int(dets.tracker_id[i]) if dets.tracker_id is not None else UNASSIGNED
                is_ball = dets.class_name[i] == CLASS_BALL
                role = ROLE_BALL if is_ball else ROLE_PLAYER
                # One finite check decides bbox_pitch and pitch_x / pitch_y together, so a box whose
                # projection is NaN or infinite (a corner on the horizon) is null in both and the
                # JSON never holds a NaN.
                bbox_pitch = None
                if pl is not None and np.isfinite(pl[i]).all() and np.isfinite(pr[i]).all():
                    bbox_pitch = BBoxPitch.from_points((float(pl[i, 0]), float(pl[i, 1])),
                                                       (float(pr[i, 0]), float(pr[i, 1])))
                extra: Dict[str, Any] = {"confidence": round(float(dets.confidence[i]), 4)}
                if res is not None:
                    mid = bbox_pitch.xy if bbox_pitch is not None else None
                    extra["pitch_x"] = None if mid is None else round(mid[0], 3)
                    extra["pitch_y"] = None if mid is None else round(mid[1], 3)
                    extra["calib_error"] = None if math.isnan(res.fit_error_px) else round(res.fit_error_px, 3)
                    extra["calib_status"] = res.status
                if dets.keypoints is not None and not is_ball:
                    kp = dets.keypoints[i]; sc = dets.keypoint_scores[i]
                    if not np.isnan(kp).all():
                        extra["keypoints"] = [[round(float(x), 2), round(float(y), 2), round(float(s), 4)]
                                              for (x, y), s in zip(kp, sc)]
                objects.append(ObjectAnnotation(
                    id=new_annotation_id(image_id, i + 1), image_id=image_id,
                    track_id=None if tid == UNASSIGNED else tid,
                    category_id=category_id_for_role(role, cats), role=role,
                    team=None if is_ball else team_of.get(tid), jersey=None,
                    bbox_image=BBoxImage(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    bbox_pitch=bbox_pitch, extra=extra))
        info = ClipInfo(name=name, game_id=game_id, clip_id=clip_id, version="1.3", frame_rate=self.frame_rate,
                        seq_length=len(per_frame), extra={"generator": "varlab_vision.pipeline", "backends": self.backends()})
        return GSRClip(path=Path(name) / LABELS_FILE, info=info, images=images, objects=objects, categories=cats,
                       other_annotations=cameras)


def _calibration_summary(calib: Sequence[Optional["CalibrationResult"]]) -> Dict[str, Any]:
    """Per-clip calibration diagnostics for the run report."""
    res = [r for r in calib if r is not None]
    if not res:
        return {}
    ok = [r for r in res if r.ok and r.status != "interpolated"]  # the fitted frames: errors and inliers are theirs
    fit = [r.fit_error_px for r in ok if not math.isnan(r.fit_error_px)]
    rep = [r.reprojection_error_m for r in ok if not math.isnan(r.reprojection_error_m)]
    return {
        "frames": len(res),
        "calibrated": sum(1 for r in res if r.status == "calibrated"),
        "tracked": sum(1 for r in res if r.status == "tracked"),
        "interpolated": sum(1 for r in res if r.status == "interpolated"),
        "uncalibrated": sum(1 for r in res if not r.ok),
        "shot_cuts": sum(1 for r in res if r.shot_cut),
        "fit_error_px_mean": (round(float(np.mean(fit)), 3) if fit else None),
        "fit_error_px_max": (round(float(np.max(fit)), 3) if fit else None),
        "reprojection_error_m_mean": (round(float(np.mean(rep)), 3) if rep else None),
        "inliers_mean": (round(float(np.mean([r.n_inliers for r in ok])), 2) if ok else None),
    }


def smoke_pipeline(projector: Optional[PitchProjector] = None, frame_rate: float = 25.0,
                   tracker: Optional[Tracker] = None, calibrator: Optional["FrameCalibrator"] = None) -> VisionPipeline:
    """CPU-only backends; exercises the code path on the synthetic clip, nothing more."""
    return VisionPipeline(detector=ColorBlobDetector(), tracker=tracker or default_tracker(frame_rate),
                          embedder=MeanColorEmbedder(), pose=None, projector=projector, frame_rate=frame_rate,
                          calibrator=calibrator)


def production_pipeline(detector_size: str = "medium", threshold: float = 0.5, device: Optional[str] = None,
                        with_pose: bool = True, pose_device: Optional[str] = None,
                        siglip_revision: Optional[str] = SigLIPEmbedder.DEFAULT_REVISION,
                        projector: Optional[PitchProjector] = None, frame_rate: float = 25.0,
                        calibrator: Optional["FrameCalibrator"] = None, with_calibration: bool = True) -> VisionPipeline:
    """RF-DETR + ByteTrack + SigLIP/k-means (+ RTMPose) with per-frame line calibration.
    Needs requirements-vision.txt and the weights.

    Calibration: an explicit ``calibrator`` is used as given; otherwise a default ``FrameCalibrator`` is
    built unless ``with_calibration`` is False or a static ``projector`` is passed (a known homography
    replaces the calibrator, it is not combined with it)."""
    import torch

    if calibrator is None and with_calibration and projector is None:
        from .calibration import FrameCalibrator

        calibrator = FrameCalibrator()

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pose = RTMPoseEstimator(device=(pose_device or ("cuda" if dev.startswith("cuda") else "cpu"))) if with_pose else None
    return VisionPipeline(detector=RFDETRDetector(size=detector_size, threshold=threshold, device=dev),
                          tracker=ByteTrackTracker(frame_rate=frame_rate),
                          embedder=SigLIPEmbedder(revision=siglip_revision, device=dev),
                          pose=pose, projector=projector, frame_rate=frame_rate, calibrator=calibrator)
