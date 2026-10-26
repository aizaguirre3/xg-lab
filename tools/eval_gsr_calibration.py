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
        --out analysis/gsr_calibration_valid.json

Labels are read from ``--labels/<clip>/Labels-GameState.json`` when present (fetch them
once with ``varlab_vision.remote_zip.extract_clip(..., labels_only=True)``),
otherwise out of the zip. Frames are never written to disk. CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
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


def frame_result(img: np.ndarray, gt) -> dict:
    t0 = time.perf_counter()
    res = cal.calibrate_frame(img)
    out = {"status": res.status, "seconds": round(time.perf_counter() - t0, 3), "n_lines": res.n_lines,
           "n_inliers": res.n_inliers, "note": res.note[:80]}
    if res.ok and len(gt):
        feet = np.stack([gt["x"] + gt["w"] / 2, gt["y"] + gt["h"]], axis=1)
        err = np.linalg.norm(res.project(feet) - gt[["x_pitch", "y_pitch"]].to_numpy(float), axis=1)
        err = err[np.isfinite(err)]
        out["errors_m"] = [round(float(e), 3) for e in err]
        out["median_m"] = round(float(np.median(err)), 3) if len(err) else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="SoccerNet/SN-GSR-2025")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--labels", default=None, help="folder with <clip>/Labels-GameState.json already fetched")
    ap.add_argument("--every", type=int, default=50, help="calibrate frames 1, 1+every, 1+2*every, ...")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every clip in the split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import cv2

    url = hf_dataset_file_url(args.dataset, f"{args.split}.zip", args.revision)
    zf, rf = open_remote_zip(url, block_size=1 << 20, max_blocks=2)  # sparse reads: small blocks
    index = clip_index(zf)
    names = {i.filename for i in zf.infolist()}
    clips = args.clips or list(index)
    per_clip, all_errs, statuses = {}, [], {}
    n_labelled_total = 0  # labelled person boxes on the sampled frames, calibrated or not
    t_start = time.time()
    for k, clip in enumerate(clips, start=1):
        lab = Path(args.labels) / clip / LABELS_FILE if args.labels else None
        if lab is None or not lab.is_file():
            lab = extract_clip(zf, clip, Path(args.out).parent / "_labels", labels_only=True, index=index) / LABELS_FILE
        gt_clip = read_gsr_clip(lab)
        df = objects_frame(gt_clip)
        df = df[df["role"].isin(PERSON_ROLES)]
        frames = list(range(1, gt_clip.info.seq_length + 1, args.every))
        rows = []
        for fi in frames:
            member = f"{index[clip]}img1/{fi:06d}.jpg"
            if member not in names:
                continue
            img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
            gt_f = df[df["frame"] == fi]
            n_labelled_total += len(gt_f)
            r = frame_result(img, gt_f)
            r["frame"] = fi
            r["n_labelled"] = int(len(gt_f))
            rows.append(r)
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            all_errs.extend(r.get("errors_m", []))
        errs = np.array([e for r in rows for e in r.get("errors_m", [])])
        per_clip[clip] = {"frames": len(rows), "calibrated": sum(r["status"] != "uncalibrated" for r in rows),
                          "errors": summarise(errs), "rows": rows}
        print(f"[{k}/{len(clips)}] {clip}: {per_clip[clip]['calibrated']}/{len(rows)} calibrated, "
              f"median {per_clip[clip]['errors'].get('median_m')} m, within 1 m "
              f"{per_clip[clip]['errors'].get('within_1m')} ({time.time() - t_start:.0f} s)", flush=True)
    errs = np.array(all_errs)
    n_frames = sum(statuses.values())
    summary = {"dataset": args.dataset, "split": args.split, "every": args.every, "clips": len(per_clip),
               "frames": n_frames, "statuses": statuses,
               "calibrated_share": round(1 - statuses.get("uncalibrated", 0) / max(1, n_frames), 4),
               "foot_errors_on_calibrated_frames": summarise(errs),
               "labelled_boxes_on_sampled_frames": int(n_labelled_total),
               "within_1m_of_all_labelled_boxes": round(float(np.sum(errs < 1.0)) / max(1, n_labelled_total), 4),
               "within_2m_of_all_labelled_boxes": round(float(np.sum(errs < 2.0)) / max(1, n_labelled_total), 4),
               "seconds_per_frame_median": round(float(np.median([r["seconds"] for c in per_clip.values()
                                                                  for r in c["rows"]])), 3) if n_frames else None,
               "network_mb": round(rf.bytes_fetched / 1e6, 1), "wall_seconds": round(time.time() - t_start, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
