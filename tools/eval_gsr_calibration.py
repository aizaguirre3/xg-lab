#!/usr/bin/env python3
"""Measure the pitch calibration on real SoccerNet GSR frames against the dataset's labels.

For every clip in ``<split>.zip`` of ``SoccerNet/SN-GSR-2025`` (public, not gated),
take every ``--every``-th frame, read just that JPEG out of the remote zip,
calibrate it from scratch (``calibration.calibrate_frame``, the global search; no
tracking between the sampled frames) and compare the pitch position it gives each
labelled person's feet (bottom-middle of the box) with the label's ``bbox_pitch``.
The labels' own positions come from SoccerNet's annotated camera, so the error
measures agreement with that annotation, not absolute truth.

    python tools/eval_gsr_calibration.py --labels data/gsr_valid --every 50 \\
        --out analysis/gsr_calibration_valid.json --workers 8

People mask: the calibrator leaves the pixels of the people out of the marking mask. In the
pipeline those boxes come from the detector; here the *labelled* person boxes of the frame
stand in for the detections (``--person-mask labels``, the default; only their image boxes are
used, never their pitch positions), and ``--person-mask none`` calibrates without any.

Headline numbers in the summary:

* ``precision_1m``: of the frames the calibrator claims (any status but ``uncalibrated``) that
  have labelled people, the share whose median foot error is under 1 m;
* ``within_1m_of_all_labelled_boxes``: labelled people placed within 1 m, over every sampled
  frame (an uncalibrated frame places nobody);
* ``calibrated_share`` and the claimed frames by median error (under 1 m, 1 to 5 m, over 5 m).

``--tracked`` feeds a clip's sampled frames in order through a ``FrameCalibrator`` (tracking, the
filter and the frame stride, as the pipeline does) instead of calibrating each from scratch;
use it with ``--every 1`` on a few clips.

Labels are read from ``--labels/<clip>/Labels-GameState.json`` when present (fetch them
once with ``varlab_vision.remote_zip.extract_clip(..., labels_only=True)``),
otherwise out of the zip. Frames are kept in memory only, unless ``--frame-cache`` names a
(gitignored) folder to keep them in for the next run. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from varlab_vision import calibration as cal  # noqa: E402
from varlab_vision.gsr_format import LABELS_FILE  # noqa: E402
from varlab_vision.loaders import objects_frame, read_gsr_clip  # noqa: E402
from varlab_vision.remote_zip import clip_index, extract_clip, hf_dataset_file_url, open_remote_zip  # noqa: E402

PERSON_ROLES = ("player", "goalkeeper", "referee", "other")


def summarise(errs: np.ndarray) -> dict:
    if len(errs) == 0:
        return {"n": 0}
    return {"n": int(len(errs)), "median_m": round(float(np.median(errs)), 3),
            "p90_m": round(float(np.percentile(errs, 90)), 3),
            "within_1m": round(float(np.mean(errs < 1.0)), 4), "within_2m": round(float(np.mean(errs < 2.0)), 4),
            "within_5m": round(float(np.mean(errs < 5.0)), 4)}


def frame_result(jpeg: bytes, gt: np.ndarray, use_boxes: bool) -> dict:
    """Calibrate one frame; ``gt`` is (N, 6): x, y, w, h (pixels) and x_pitch, y_pitch (metres) per person."""
    import cv2

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    boxes = np.stack([gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]], axis=1) if len(gt) else None
    t0 = time.perf_counter()
    res = cal.calibrate_frame(img, person_boxes=boxes if use_boxes else None)
    out = {"status": res.status, "seconds": round(time.perf_counter() - t0, 3), "n_lines": res.n_lines,
           "n_inliers": res.n_inliers, "n_curves": len(res.arc_matches), "note": res.note[:160],
           "refused": res.note.startswith("refused")}
    if res.ok and len(gt):
        feet = np.stack([gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3]], axis=1)
        err = np.linalg.norm(res.project(feet) - gt[:, 4:6], axis=1)
        err = err[np.isfinite(err)]
        out["errors_m"] = [round(float(e), 3) for e in err]
        out["median_m"] = round(float(np.median(err)), 3) if len(err) else None
    return out


def _work(args):
    jpeg, gt, use_boxes = args
    return frame_result(jpeg, gt, use_boxes)


def _errors_row(res, gt: np.ndarray, seconds: float) -> dict:
    out = {"status": res.status, "seconds": round(seconds, 4), "n_inliers": res.n_inliers, "note": res.note[:160],
           "refused": res.note.startswith("refused")}
    if res.ok and len(gt):
        feet = np.stack([gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3]], axis=1)
        err = np.linalg.norm(res.project(feet) - gt[:, 4:6], axis=1)
        err = err[np.isfinite(err)]
        out["errors_m"] = [round(float(e), 3) for e in err]
        out["median_m"] = round(float(np.median(err)), 3) if len(err) else None
    return out


def _work_tracked(frames: list) -> list:
    """One clip in order through a FrameCalibrator: ``frames`` is [(jpeg, gt, use_boxes), ...]."""
    import cv2

    fc = cal.FrameCalibrator()
    gts, out, t_prev = [], [], time.perf_counter()
    for jpeg, gt, use_boxes in frames:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        boxes = np.stack([gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]], axis=1) if len(gt) else None
        gts.append(gt)
        for res in fc.push(img, person_boxes=boxes if use_boxes else None):
            now = time.perf_counter()
            out.append(_errors_row(res, gts[len(out)], now - t_prev))
            t_prev = now
    for res in fc.flush():
        out.append(_errors_row(res, gts[len(out)], 0.0))
    return out


def summary_of(rows: list, n_labelled_total: int) -> dict:
    """The headline numbers (see the module docstring) over per-frame rows."""
    n = len(rows)
    claimed = [r for r in rows if r["status"] != "uncalibrated"]
    meds = np.array([r["median_m"] for r in claimed if r.get("median_m") is not None], float)
    errs = np.array([e for r in rows for e in r.get("errors_m", [])], float)
    statuses: dict = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "frames": n, "statuses": statuses,
        "calibrated_share": round(len(claimed) / max(1, n), 4),
        "refused": int(sum(r.get("refused", False) for r in rows)),
        "claimed_frames_with_people": int(len(meds)),
        "claimed_by_median": {"under_1m": int((meds < 1).sum()), "1_to_5m": int(((meds >= 1) & (meds < 5)).sum()),
                              "over_5m": int((meds >= 5).sum())},
        "precision_1m": round(float(np.mean(meds < 1)), 4) if len(meds) else None,
        "precision_2m": round(float(np.mean(meds < 2)), 4) if len(meds) else None,
        "foot_errors_on_calibrated_frames": summarise(errs),
        "labelled_boxes_on_sampled_frames": int(n_labelled_total),
        "within_1m_of_all_labelled_boxes": round(float(np.sum(errs < 1.0)) / max(1, n_labelled_total), 4),
        "within_2m_of_all_labelled_boxes": round(float(np.sum(errs < 2.0)) / max(1, n_labelled_total), 4),
        "seconds_per_frame_median": round(float(np.median([r["seconds"] for r in rows])), 3) if n else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="SoccerNet/SN-GSR-2025")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--labels", default=None, help="folder with <clip>/Labels-GameState.json already fetched")
    ap.add_argument("--every", type=int, default=50, help="calibrate frames 1, 1+every, 1+2*every, ...")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every clip in the split")
    ap.add_argument("--person-mask", choices=("labels", "none"), default="labels",
                    help="leave the labelled person boxes out of the marking mask (stand-in for detections)")
    ap.add_argument("--workers", type=int, default=1, help="processes for the calibration (frames are read first)")
    ap.add_argument("--frame-cache", default=None, help="folder to keep the read JPEGs in (gitignored data/...)")
    ap.add_argument("--tracked", action="store_true", help="each clip in order through a FrameCalibrator")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    url = hf_dataset_file_url(args.dataset, f"{args.split}.zip", args.revision)
    zf, rf = open_remote_zip(url, block_size=1 << 20, max_blocks=2)  # sparse reads: small blocks
    index = clip_index(zf)
    names = {i.filename for i in zf.infolist()}
    clips = args.clips or list(index)
    cache = Path(args.frame_cache) if args.frame_cache else None
    t_start = time.time()
    jobs, keys, n_labelled = [], [], {}
    for clip in clips:
        lab = Path(args.labels) / clip / LABELS_FILE if args.labels else None
        if lab is None or not lab.is_file():
            lab = extract_clip(zf, clip, Path(args.out).parent / "_labels", labels_only=True, index=index) / LABELS_FILE
        gt_clip = read_gsr_clip(lab)
        df = objects_frame(gt_clip)
        df = df[df["role"].isin(PERSON_ROLES)]
        for fi in range(1, gt_clip.info.seq_length + 1, args.every):
            member = f"{index[clip]}img1/{fi:06d}.jpg"
            if member not in names:
                continue
            cached = cache / clip / f"{fi:06d}.jpg" if cache else None
            if cached is not None and cached.is_file():
                jpeg = cached.read_bytes()
            else:
                jpeg = zf.read(member)
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(jpeg)
            g = df[df["frame"] == fi][["x", "y", "w", "h", "x_pitch", "y_pitch"]].to_numpy(float)
            jobs.append((jpeg, g, args.person_mask == "labels"))
            keys.append((clip, fi))
            n_labelled[clip] = n_labelled.get(clip, 0) + len(g)
    print(f"read {len(jobs)} frames of {len(clips)} clips ({time.time() - t_start:.0f} s)", flush=True)
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")  # one BLAS thread per worker process
    if args.tracked:  # one job per clip, its frames in order
        by_clip: dict = {}
        for (clip, _fi), job in zip(keys, jobs):
            by_clip.setdefault(clip, []).append(job)
        work, fn, chunk = list(by_clip.values()), _work_tracked, 1
    else:
        work, fn, chunk = jobs, _work, 4
    if args.workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(fn, work, chunksize=chunk)
    else:
        results = [fn(j) for j in work]
    if args.tracked:
        results = [r for clip_rows in results for r in clip_rows]
    per_clip: dict = {}
    for (clip, fi), r, (_j, g, _b) in zip(keys, results, jobs):
        r["frame"] = fi
        r["n_labelled"] = int(len(g))
        per_clip.setdefault(clip, {"rows": []})["rows"].append(r)
    for k, (clip, c) in enumerate(per_clip.items(), start=1):
        s = summary_of(c["rows"], n_labelled[clip])
        c.update({"frames": s["frames"], "calibrated": s["frames"] - s["statuses"].get("uncalibrated", 0),
                  "precision_1m": s["precision_1m"], "within_1m": s["within_1m_of_all_labelled_boxes"],
                  "errors": s["foot_errors_on_calibrated_frames"]})
        print(f"[{k}/{len(per_clip)}] {clip}: {c['calibrated']}/{c['frames']} calibrated, precision "
              f"{c['precision_1m']}, people within 1 m {c['within_1m']}", flush=True)
    summary = {"dataset": args.dataset, "split": args.split, "every": args.every, "clips": len(per_clip),
               "mode": "tracked (FrameCalibrator, frame stride)" if args.tracked else "each frame from scratch",
               "person_mask": ("labelled person boxes (stand-in for the detector's)" if args.person_mask == "labels"
                               else "none"),
               **summary_of(results, sum(n_labelled.values())),
               "network_mb": round(rf.bytes_fetched / 1e6, 1), "wall_seconds": round(time.time() - t_start, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
