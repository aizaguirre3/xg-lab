"""A synthetic clip: coloured rectangles drifting over a green background.

This exists only to exercise the pipeline code path on a CPU with no data. The
rectangles are not people, so the real detectors (RF-DETR) will not find them;
the smoke backends in ``pipeline.py`` (colour blobs, mean-colour team split) do.
The clip comes with a ground-truth ``Labels-GameState.json`` in the GSR layout
so the loader and the GS-HOTA adapter can be run end to end on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

GREEN_BGR = (40, 150, 40)
TEAM_COLOURS_BGR = {"left": (30, 30, 220), "right": (220, 60, 30)}  # red-ish, blue-ish (BGR)
REFEREE_BGR = (20, 20, 20)


def synthetic_homography(width: int, height: int) -> np.ndarray:
    """Pixels -> metres: a plain scaling that puts the frame over the whole pitch.

    ``x_m = (px - width/2) * 105/width``, ``y_m = (py - height/2) * 68/height``.
    Not a camera model; it only gives the smoke test a pitch frame to project into.
    """
    return np.array(
        [[105.0 / width, 0.0, -52.5], [0.0, 68.0 / height, -34.0], [0.0, 0.0, 1.0]], dtype=float
    )


def project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) pixel points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h.T
    return hom[:, :2] / hom[:, 2:3]


@dataclass
class SyntheticClip:
    clip_dir: Path
    labels_path: Path
    width: int
    height: int
    n_frames: int
    homography: np.ndarray

    @property
    def image_dir(self) -> Path:
        return self.clip_dir / "img1"


def render_synthetic_clip(out_dir: Path | str, n_frames: int = 30, width: int = 640, height: int = 360,
                          n_per_team: int = 3, with_referee: bool = True, seed: int = 0,
                          name: str = "SYN-001", game_id: str = "syn-game-1",
                          box_wh: Tuple[int, int] = (24, 56), with_jerseys: bool = False) -> SyntheticClip:
    """Write ``<out_dir>/<name>/img1/000001.jpg ...`` and the ground-truth labels.

    Jersey numbers are null unless ``with_jerseys``: GS-HOTA requires the predicted
    jersey to equal the ground truth, and the pipeline has no jersey recognition
    yet, so a ground truth with numbers would score every player as a miss.
    """
    import cv2

    rng = np.random.default_rng(seed)
    clip_dir = Path(out_dir) / name
    img_dir = clip_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    bw, bh = box_wh
    h_px2m = synthetic_homography(width, height)

    # Actors: (role, team, jersey, colour, start xy (top-left), velocity)
    actors = []
    margin = 10
    k = 0
    # Team "left" starts in the left half and "right" in the right half, so the
    # pipeline's placeholder left/right naming (by mean image x) is well defined.
    half = width / 2.0
    drift = 2.0 * n_frames + 10  # room so a box never crosses the halfway line
    for team in ("left", "right"):
        lo, hi = (margin, half - bw - drift) if team == "left" else (half + 10, width - bw - margin - drift)
        for j in range(n_per_team):
            k += 1
            x0 = rng.uniform(lo, max(lo + 1, hi))
            y0 = rng.uniform(margin, height - bh - margin - 40)
            vx = rng.uniform(0.5, 2.0)
            vy = rng.uniform(-1.0, 1.0)
            jersey = str(k) if with_jerseys else None
            actors.append(("player", team, jersey, TEAM_COLOURS_BGR[team], (x0, y0), (vx, vy)))
    if with_referee:
        actors.append(("referee", None, None, REFEREE_BGR,
                       (width * 0.5, height * 0.5), (-0.5, 0.3)))

    cats = parse_categories(default_categories_list())
    images: List[ImageRecord] = []
    objects: List[ObjectAnnotation] = []
    clip_id = "9001"
    for f in range(1, n_frames + 1):
        frame = np.full((height, width, 3), GREEN_BGR, dtype=np.uint8)
        image_id = new_image_id(clip_id, f)
        images.append(ImageRecord(image_id=image_id, file_name=frame_file_name(f), width=width, height=height,
                                  is_labeled=True, has_labeled_person=True))
        for t, (role, team, jersey, colour, (x0, y0), (vx, vy)) in enumerate(actors, start=1):
            x = float(np.clip(x0 + vx * (f - 1), 0, width - bw))
            y = float(np.clip(y0 + vy * (f - 1), 0, height - bh))
            xi, yi = int(round(x)), int(round(y))
            cv2.rectangle(frame, (xi, yi), (xi + bw - 1, yi + bh - 1), colour, thickness=-1)
            box = BBoxImage(x=float(xi), y=float(yi), w=float(bw), h=float(bh))
            bl, br = project(h_px2m, np.array([[box.x, box.y + box.h], [box.x + box.w, box.y + box.h]]))
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
                    extra={"synthetic": True, "note": "coloured rectangles; code-path smoke test only"})
    clip = GSRClip(path=clip_dir / LABELS_FILE, info=info, images=images, objects=objects, categories=cats)
    write_labels_json(clip)
    return SyntheticClip(clip_dir=clip_dir, labels_path=clip.path, width=width, height=height,
                         n_frames=n_frames, homography=h_px2m)
