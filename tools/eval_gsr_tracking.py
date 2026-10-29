#!/usr/bin/env python3
"""Score the vision pipeline's detection, tracking and team assignment on SoccerNet GSR clips.

Image-space metrics, so they need no pitch calibration:

* people: per frame, predicted boxes are matched to labelled people (player,
  goalkeeper, referee, other) by IoU with the Hungarian algorithm, a match needs
  IoU >= 0.5. Precision, recall and F1 of the boxes; recall per labelled role.
* tracking: MOTA = 1 - (FN + FP + ID switches) / labelled boxes, and IDF1, the
  identity F1 of Ristani et al. (2016): trajectories are paired one to one to
  maximise the number of frames in which a pair overlaps with IoU >= 0.5.
* teams: on matched player and goalkeeper boxes, agreement of the predicted
  left / right label with the labelled one, under the better of the two possible
  namings per clip (the pipeline names its two clusters by image side, the labels
  by pitch side, so only the grouping is comparable).
* the pipeline also writes detections the tracker did not assign to a track
  (``track_id`` null). Tracking and team metrics use the tracker's output only
  (boxes with a track id), as MOT evaluation does; ``detector_*`` repeats precision
  and recall over every predicted box.
* ball: a labelled ball counts as found when a predicted ball centre lies within
  ``--ball-px`` pixels of it.
* timing: per-stage seconds from each clip's ``report.json``.

    python tools/eval_gsr_tracking.py --pred data/gsr_pred/valid --labels data/gsr_valid \\
        --out data/gsr_eval/tracking_valid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from varlab_vision.gsr_format import LABELS_FILE  # noqa: E402
from varlab_vision.loaders import objects_frame, read_gsr_clip  # noqa: E402

PERSON_ROLES = ("player", "goalkeeper", "referee", "other")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N, 4) and (M, 4) boxes given as x, y, w, h."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def score_clip(gt: pd.DataFrame, pr: pd.DataFrame, iou_thr: float = 0.5, ball_px: float = 15.0) -> dict:
    g_all, p_all = gt[gt["role"].isin(PERSON_ROLES)], pr[pr["role"] != "ball"]
    gb, pb = gt[gt["role"] == "ball"], pr[pr["role"] == "ball"]
    frames = sorted(set(gt["frame"]) | set(pr["frame"]))
    tp = fp = fn = idsw = 0
    role_tp = {r: 0 for r in PERSON_ROLES}
    role_n = {r: int((g_all["role"] == r).sum()) for r in PERSON_ROLES}
    last_match: dict = {}
    co = {}  # (gt_track, pred_track) -> frames overlapping with IoU >= thr
    team_pairs = []
    ball_found = ball_n = ball_pred_ok = ball_pred_n = 0
    g_by = {f: d for f, d in g_all.groupby("frame")}
    p_by = {f: d for f, d in p_all.groupby("frame")}
    gb_by = {f: d for f, d in gb.groupby("frame")}
    pb_by = {f: d for f, d in pb.groupby("frame")}
    for f in frames:
        g = g_by.get(f, g_all.iloc[:0])
        p = p_by.get(f, p_all.iloc[:0])
        iou = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), p[["x", "y", "w", "h"]].to_numpy(float))
        g_ids = g["track_id"].to_numpy(); p_ids = p["track_id"].to_numpy()
        if iou.size:
            gi, pj = np.nonzero(iou >= iou_thr)
            for a, b in zip(gi, pj):
                key = (g_ids[a], p_ids[b])
                co[key] = co.get(key, 0) + 1
            r, c = linear_sum_assignment(-iou)
            keep = iou[r, c] >= iou_thr
            r, c = r[keep], c[keep]
        else:
            r = c = np.zeros(0, int)
        tp += len(r); fn += len(g) - len(r); fp += len(p) - len(r)
        roles = g["role"].to_numpy()
        for a, b in zip(r, c):
            role_tp[roles[a]] += 1
            gid, pid = g_ids[a], p_ids[b]
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
            gt_team, pr_team = g["team"].iloc[a], p["team"].iloc[b]
            if roles[a] in ("player", "goalkeeper") and gt_team in ("left", "right") and pr_team in ("left", "right"):
                team_pairs.append((gt_team, pr_team))
        # ball
        gbf, pbf = gb_by.get(f), pb_by.get(f)
        if gbf is not None and len(gbf):
            ball_n += len(gbf)
            if pbf is not None and len(pbf):
                gc = np.stack([gbf["x"] + gbf["w"] / 2, gbf["y"] + gbf["h"] / 2], axis=1)
                pc = np.stack([pbf["x"] + pbf["w"] / 2, pbf["y"] + pbf["h"] / 2], axis=1)
                d = np.linalg.norm(gc[:, None] - pc[None], axis=2)
                ball_found += int((d.min(axis=1) <= ball_px).sum())
        if pbf is not None and len(pbf):
            ball_pred_n += len(pbf)
            if gbf is not None and len(gbf):
                gc = np.stack([gbf["x"] + gbf["w"] / 2, gbf["y"] + gbf["h"] / 2], axis=1)
                pc = np.stack([pbf["x"] + pbf["w"] / 2, pbf["y"] + pbf["h"] / 2], axis=1)
                ball_pred_ok += int((np.linalg.norm(pc[:, None] - gc[None], axis=2).min(axis=1) <= ball_px).sum())
    # IDF1: one-to-one trajectory pairing maximising overlapping frames
    g_tracks = sorted(set(g_all["track_id"])); p_tracks = sorted(set(p_all["track_id"]))
    gi = {t: i for i, t in enumerate(g_tracks)}; pi = {t: i for i, t in enumerate(p_tracks)}
    m = np.zeros((len(g_tracks), len(p_tracks)))
    for (gt_t, p_t), n in co.items():
        m[gi[gt_t], pi[p_t]] = n
    idtp = float(m[linear_sum_assignment(-m)].sum()) if m.size else 0.0
    n_gt, n_pr = len(g_all), len(p_all)
    team_acc = None
    if team_pairs:
        same = sum(a == b for a, b in team_pairs)
        team_acc = max(same, len(team_pairs) - same) / len(team_pairs)
    return {"gt_boxes": n_gt, "pred_boxes": n_pr, "tp": tp, "fp": fp, "fn": fn, "idsw": idsw, "idtp": idtp,
            "gt_tracks": len(g_tracks), "pred_tracks": len(p_tracks),
            "role_tp": role_tp, "role_n": role_n, "team_pairs": len(team_pairs),
            "team_agree": (None if team_acc is None else round(team_acc * len(team_pairs))),
            "ball_n": ball_n, "ball_found": ball_found, "ball_pred_n": ball_pred_n, "ball_pred_ok": ball_pred_ok}


def rates(s: dict) -> dict:
    prec = s["tp"] / max(1, s["tp"] + s["fp"]); rec = s["tp"] / max(1, s["tp"] + s["fn"])
    out = {"precision": round(prec, 4), "recall": round(rec, 4),
           "f1": round(2 * prec * rec / max(1e-12, prec + rec), 4),
           "mota": round(1 - (s["fn"] + s["fp"] + s["idsw"]) / max(1, s["gt_boxes"]), 4),
           "idf1": round(2 * s["idtp"] / max(1, s["gt_boxes"] + s["pred_boxes"]), 4),
           "id_switches": s["idsw"], "gt_tracks": s["gt_tracks"], "pred_tracks": s["pred_tracks"],
           "recall_by_role": {r: (round(s["role_tp"][r] / s["role_n"][r], 4) if s["role_n"][r] else None)
                              for r in PERSON_ROLES},
           "team_accuracy": (round(s["team_agree"] / s["team_pairs"], 4) if s["team_pairs"] else None),
           "ball_recall": (round(s["ball_found"] / s["ball_n"], 4) if s["ball_n"] else None),
           "ball_precision": (round(s["ball_pred_ok"] / s["ball_pred_n"], 4) if s["ball_pred_n"] else None)}
    if "det_tp" in s:
        out["detector_precision"] = round(s["det_tp"] / max(1, s["det_tp"] + s["det_fp"]), 4)
        out["detector_recall"] = round(s["det_tp"] / max(1, s["det_tp"] + s["det_fn"]), 4)
        out["untracked_boxes"] = s.get("untracked_boxes", 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="folder with <clip>/Labels-GameState.json (+ report.json)")
    ap.add_argument("--labels", required=True, help="folder with <clip>/Labels-GameState.json ground truth")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--ball-px", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    pred_root, gt_root = Path(args.pred), Path(args.labels)
    clips = sorted(p.name for p in pred_root.iterdir() if (p / LABELS_FILE).is_file() and (gt_root / p.name / LABELS_FILE).is_file())
    per_clip, totals, timings = {}, None, []
    for clip in clips:
        gt = objects_frame(read_gsr_clip(gt_root / clip / LABELS_FILE))
        pr_all = objects_frame(read_gsr_clip(pred_root / clip / LABELS_FILE))
        untracked = pr_all["track_id"].isna() & (pr_all["role"] != "ball")
        pr = pr_all[~untracked]
        s = score_clip(gt, pr, args.iou, args.ball_px)
        det_only = pr_all.copy()
        det_only.loc[untracked, "track_id"] = -np.arange(1, int(untracked.sum()) + 1)  # each its own id
        d = score_clip(gt, det_only, args.iou, args.ball_px)
        s["untracked_boxes"] = int(untracked.sum())
        s["det_tp"], s["det_fp"], s["det_fn"] = d["tp"], d["fp"], d["fn"]
        per_clip[clip] = {**rates(s), "counts": s}
        if totals is None:
            totals = {k: (dict(v) if isinstance(v, dict) else v) for k, v in s.items()}
        else:
            for k, v in s.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        totals[k][kk] += vv
                elif v is None:
                    continue
                else:
                    totals[k] = (totals[k] or 0) + v
        rep = pred_root / clip / "report.json"
        if rep.is_file():
            r = json.loads(rep.read_text())
            st = r.get("stages", {})
            timings.append({"clip": clip, "wall_s": r.get("wall_seconds"),
                            "download_s": (r.get("download") or {}).get("seconds"),
                            "download_mb": (r.get("download") or {}).get("mb"),
                            **{f"{k}_s": (st.get(k) or {}).get("seconds") for k in ("read", "detect", "track", "team", "total")}})
        print(f"{clip}: P {per_clip[clip]['precision']:.3f} R {per_clip[clip]['recall']:.3f} "
              f"MOTA {per_clip[clip]['mota']:.3f} IDF1 {per_clip[clip]['idf1']:.3f} "
              f"team {per_clip[clip]['team_accuracy']}", flush=True)
    if totals is None:
        print("no clips with both predictions and labels"); return 1
    # team accuracy must be pooled per clip (each clip picks its own naming), not from summed counts
    team_pairs = sum(c["counts"]["team_pairs"] for c in per_clip.values())
    team_agree = sum(c["counts"]["team_agree"] or 0 for c in per_clip.values())
    overall = rates({**totals, "team_pairs": team_pairs, "team_agree": team_agree})
    tdf = pd.DataFrame(timings)
    timing = {c: round(float(tdf[c].median()), 2) for c in tdf.columns if c != "clip" and tdf[c].notna().any()} if len(tdf) else {}
    out = {"clips": len(per_clip), "iou_threshold": args.iou, "overall": overall,
           "median_seconds_per_clip": timing, "per_clip": per_clip, "timings": timings}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"clips": len(per_clip), "overall": overall, "median_seconds_per_clip": timing}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
