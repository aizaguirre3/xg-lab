#!/usr/bin/env python3
"""Replay re-linking and team assignment from saved clip caches and score them, without the GPU.

A Colab run with ``run_clips.py --cache`` leaves ``<out>/<clip>/cache.pkl`` beside each prediction:
the detections and track ids before re-linking, the map of every frame into its shot's reference
frame, the crop embeddings and the calibration. This rebuilds the clip's GSR output with the given
``StitchConfig`` (or none) and team rule, scores it against the labels with
``eval_gsr_tracking.score_clip``, and prints the pooled numbers, so the thresholds can be tuned on
train clips in seconds.

The caches are pickles written by this repository's own pipeline on our own Colab VM and pulled
back by our runner; loading a pickle runs code, so never point this at a cache from anywhere else.

    python tools/replay_tracking.py --cache data/gsr_pred/train_cache --labels data/gsr_train_full \
        --stitch max_gap=50 max_appearance=0.35 --out analysis/replay.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_gsr_tracking import rates, score_clip  # noqa: E402
from varlab_vision import pipeline as pl  # noqa: E402
from varlab_vision.calibration import CalibrationResult  # noqa: E402
from varlab_vision.gsr_format import LABELS_FILE, write_labels_json  # noqa: E402
from varlab_vision.loaders import objects_frame, read_gsr_clip  # noqa: E402
from varlab_vision.stitching import StitchConfig, mean_embeddings, stitch, tracklets_from_frames  # noqa: E402


def _calib(rows):
    out = []
    for status, h in rows:
        if h is None:
            out.append(CalibrationResult(status=status))
        else:
            hi = np.linalg.inv(h)
            out.append(CalibrationResult(status=status, h_m2px=h, h_px2m=hi / hi[2, 2]))
    return out


def replay_clip(cache: dict, name: str, stitch_cfg: StitchConfig | None, team_k: int = 2, seed: int = 0,
                whiten: int = 4):
    """The clip's GSR predictions rebuilt from its cache; returns (clip, n fragments, n tracks)."""
    per_frame, tids, feats, cxs = cache["per_frame"], list(cache["tids"]), np.asarray(cache["feats"]), cache["cxs"]
    calib = _calib(cache["calib"]) if cache["calib"] else None  # none when the run had no calibrator
    n_before = len({int(t) for t in tids})
    if stitch_cfg is not None and len(tids):
        tl = tracklets_from_frames(per_frame, cache["to_ref"], cache["shots"], pl.CLASS_PERSON)
        emb = mean_embeddings(tids, feats)
        roles = pl.track_roles(per_frame)
        for t in tl:
            t.embedding, t.role = emb.get(t.tid), roles.get(t.tid)
        id_map = stitch(tl, stitch_cfg)
        per_frame = [(f, w, h, pl._remap_ids(d, id_map)) for f, w, h, d in per_frame]
        tids = [id_map.get(t, t) for t in tids]
    roles = pl.track_roles(per_frame)
    outfield = np.array([roles.get(t, "player") == "player" for t in tids], bool)
    team_of = {}
    if int(outfield.sum()) >= team_k:
        cluster = pl.team_clusters(feats, outfield, team_k, seed, whiten)
        team_of = pl.assign_teams(np.array(tids), cluster, np.array(cxs), roles=roles,
                                  pitch_x=pl.track_pitch_x(per_frame, calib) if calib else None)
    pipe = pl.VisionPipeline.__new__(pl.VisionPipeline)
    pipe.frame_rate = 25.0
    pipe.detector, pipe.tracker, pipe.embedder, pipe.pose, pipe.projector, pipe.calibrator = (None,) * 6
    pipe.backends = lambda: cache.get("backends", {})
    clip = pipe._to_gsr(name, "", per_frame, team_of, calib, roles)
    return clip, n_before, len({int(t) for t in tids})


def _cfg(pairs):
    if pairs is None:
        return None
    names = {f.name: f.type for f in fields(StitchConfig)}
    kw = {}
    for p in pairs:
        k, v = p.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown StitchConfig field {k}")
        kw[k] = int(v) if names[k] in ("int", int) else float(v)
    return StitchConfig(**kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="folder with <clip>/cache.pkl")
    ap.add_argument("--labels", required=True, help="folder with <clip>/Labels-GameState.json (ground truth)")
    ap.add_argument("--stitch", nargs="*", default=None, help="StitchConfig fields as key=value; omit to not stitch")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = _cfg(args.stitch)
    clips = args.clips or sorted(p.parent.name for p in Path(args.cache).glob("*/cache.pkl"))
    totals: dict = {}
    per_clip = {}
    team_pairs = team_agree = frags = tracks = 0
    for clip in clips:
        with open(Path(args.cache) / clip / "cache.pkl", "rb") as fh:
            cache = pickle.load(fh)
        built, n_before, n_after = replay_clip(cache, clip, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            write_labels_json(built, Path(tmp) / LABELS_FILE)
            pred = objects_frame(read_gsr_clip(Path(tmp) / LABELS_FILE))
        gt = objects_frame(read_gsr_clip(Path(args.labels) / clip / LABELS_FILE))
        untracked = pred["track_id"].isna() & (pred["role"] != "ball")  # as eval_gsr_tracking: tracks only
        s = score_clip(gt, pred[~untracked])
        for k, v in s.items():  # pooled as eval_gsr_tracking.main pools them
            if k in ("team_pairs", "team_agree") or v is None:
                continue
            if isinstance(v, dict):
                d = totals.setdefault(k, {kk: 0 for kk in v})
                for kk, vv in v.items():
                    d[kk] += vv
            else:
                totals[k] = totals.get(k, 0) + v
        team_pairs += s.get("team_pairs") or 0
        team_agree += s.get("team_agree") or 0
        frags += n_before
        tracks += n_after
        per_clip[clip] = {**rates(s), "fragments": n_before, "tracks": n_after}
        print(f"{clip}: idf1 {per_clip[clip].get('idf1')} team {per_clip[clip].get('team_accuracy')} "
              f"tracks {n_before} -> {n_after}", flush=True)
    overall = {**rates({**totals, "team_pairs": team_pairs, "team_agree": team_agree}), "fragments": frags, "tracks": tracks,
               "stitch": None if cfg is None else cfg.__dict__}
    print(json.dumps(overall, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"overall": overall, "per_clip": per_clip}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
