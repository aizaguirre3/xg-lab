"""Re-linking track fragments after the clip, by place, time and appearance.

ByteTrack follows a player from box to box and starts a new id whenever the chain breaks: a
player hidden behind another, a missed detection for longer than its buffer, a jump the motion
model did not expect. On the 58 SoccerNet GSR validation clips it gave 2,957 tracks for 1,350
labelled people (IDF1 0.63). Because the pipeline sees the whole clip before it writes anything,
a fragment that ends can be joined to one that starts later, if the second starts where the first
player could have run to in the time between, looks like them, and plays for the same team.

Positions are the feet (bottom-middle of the box) in the coordinates of the shot's reference
frame (``MotionCompensatedTracker``), so the camera's pan does not move them; distances are
measured in body heights (the box height), which keeps the rule the same for a near and a far
player. One assignment over all fragment ends and starts (Hungarian, infeasible pairs excluded)
links each end to at most one start; chains form on their own, since a link always goes forward
in time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Tracklet:
    tid: int
    frames: np.ndarray  # (n,) frame indices, increasing
    feet: np.ndarray  # (n, 2) bottom-middle of the box in the shot's reference coordinates
    heights: np.ndarray  # (n,) box heights, pixels (the scale for distances)
    shot: int  # shot of the fragment (a fragment never spans a cut)
    embedding: Optional[np.ndarray] = None  # mean appearance, unit length
    team: Optional[str] = None
    role: Optional[str] = None

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def end_state(self, k: int = 10):
        """Last foot position and velocity per frame (over the last ``k`` frames), and the last height."""
        f, p = self.frames[-k:], self.feet[-k:]
        v = (p[-1] - p[0]) / float(f[-1] - f[0]) if len(f) > 1 and f[-1] > f[0] else np.zeros(2)
        return p[-1], v, float(np.median(self.heights[-k:]))


@dataclass
class StitchConfig:
    """Defaults chosen on five held-out GSR train clips (fine-tuned detector, motion-compensated
    ByteTrack): IDF1 0.689 -> 0.781, 252 -> 148 tracks for 118 people, ID switches 481 -> 433. IDF1
    rose with ``max_gap`` up to 300 frames and stayed flat beyond; at such gaps the radius covers most
    of the view, so time, role and appearance decide."""

    max_gap: int = 300  # frames between the end of one fragment and the start of the next (12 s at 25 fps)
    radius0: float = 0.8  # body heights allowed at a gap of one frame
    speed: float = 0.3  # body heights per frame the allowed radius grows by
    max_velocity_frames: int = 12  # the end velocity is extrapolated over at most this many frames
    max_appearance: float = 0.25  # cosine distance above which two fragments are not the same player
    appearance_weight: float = 1.0


def stitch(tracklets: Sequence[Tracklet], cfg: StitchConfig = StitchConfig()) -> Dict[int, int]:
    """``{tid: id of the chain it joins}`` for every tracklet (its own id when it joins nothing)."""
    from scipy.optimize import linear_sum_assignment

    tl = list(tracklets)
    n = len(tl)
    out = {t.tid: t.tid for t in tl}
    if n < 2:
        return out
    big = 1e6
    cost = np.full((n, n), big)
    for i, a in enumerate(tl):
        p, v, h = a.end_state()
        for j, b in enumerate(tl):
            gap = b.start - a.end
            if i == j or gap < 1 or gap > cfg.max_gap or a.shot != b.shot:
                continue
            if a.team is not None and b.team is not None and a.team != b.team:
                continue
            if a.role is not None and b.role is not None and a.role != b.role:
                continue
            pred = p + v * min(gap, cfg.max_velocity_frames)
            scale = 0.5 * (h + float(np.median(b.heights[:10])))
            d = float(np.linalg.norm(b.feet[0] - pred)) / max(scale, 1.0)
            radius = cfg.radius0 + cfg.speed * gap
            if d > radius:
                continue
            app = 0.0
            if a.embedding is not None and b.embedding is not None:
                app = 1.0 - float(np.dot(a.embedding, b.embedding))
                if app > cfg.max_appearance:
                    continue
            cost[i, j] = d / radius + cfg.appearance_weight * app / cfg.max_appearance
    rows, cols = linear_sum_assignment(cost)
    succ = {int(i): int(j) for i, j in zip(rows, cols) if cost[i, j] < big}
    pred_of = {j: i for i, j in succ.items()}
    for i in range(n):
        if i in pred_of:
            continue  # not the head of a chain
        head, k = tl[i].tid, i
        while k in succ:
            k = succ[k]
            out[tl[k].tid] = head
    return out


def mean_embeddings(tids: Sequence[int], feats: np.ndarray) -> Dict[int, np.ndarray]:
    """Unit-length mean of each track's crop embeddings."""
    tids = np.asarray(tids)
    feats = np.asarray(feats, float)
    out: Dict[int, np.ndarray] = {}
    for t in np.unique(tids):
        m = feats[tids == t].mean(axis=0)
        nrm = float(np.linalg.norm(m))
        out[int(t)] = m / nrm if nrm > 0 else m
    return out


def tracklets_from_frames(per_frame: Sequence, to_ref: Sequence[np.ndarray], shots: Sequence[int],
                          person_class: str = "person") -> List[Tracklet]:
    """Tracklets from the pipeline's per-frame detections (``(frame, w, h, FrameDetections)``), the map
    of each frame's pixels into its shot's reference frame and each frame's shot index."""
    rows: Dict[int, List[tuple]] = {}
    for (fidx, _w, _h, dets), h_ref, shot in zip(per_frame, to_ref, shots):
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            if tid <= 0 or dets.class_name[i] != person_class:
                continue
            x0, y0, x1, y1 = (float(v) for v in dets.xyxy[i])
            foot = np.array([(x0 + x1) / 2.0, y1, 1.0]) @ np.asarray(h_ref, float).T
            rows.setdefault(tid, []).append((fidx, foot[:2] / foot[2], y1 - y0, shot))
    out = []
    for tid, r in rows.items():
        r.sort(key=lambda t: t[0])
        shot = r[0][3]
        r = [t for t in r if t[3] == shot]  # a track that outlived a cut keeps its first shot only
        out.append(Tracklet(tid=tid, frames=np.array([t[0] for t in r]), feet=np.array([t[1] for t in r]),
                            heights=np.array([t[2] for t in r], float), shot=shot))
    return out
