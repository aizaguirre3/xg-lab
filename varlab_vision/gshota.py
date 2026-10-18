"""GS-HOTA: HOTA with the Game-State similarity, written from the public definition.

Definition read from the sn-gamestate README
(https://github.com/SoccerNet/sn-gamestate) and Section 5 of arXiv:2404.11335
(https://arxiv.org/html/2404.11335):

    Sim(P, G)    = LocSim(P, G) * IdSim(P, G)
    LocSim(P, G) = exp(ln(0.05) * ||P - G||^2 / tau^2),   tau = 5 m
    IdSim(P, G)  = 1 if role, team and jersey all match, else 0

with P and G the bottom-middle pitch positions in metres, then the HOTA
formulation of Luiten et al. (IJCV 2021): for alpha in {0.05, 0.10, ..., 0.95}
match predictions to ground truth per frame with the Hungarian algorithm on
``global alignment score x similarity``, keep pairs whose similarity exceeds
alpha, accumulate TP/FN/FP (DetA) and per-identity-pair association counts
(AssA), HOTA_alpha = sqrt(DetA_alpha * AssA_alpha) and GS-HOTA is the mean
over alpha. This is an independent re-implementation of the published maths,
not a copy of the sn-gamestate / TrackEval code.

Attribute matching. The README says attributes missing from the ground truth
are ignored (referees have no jersey) but also that non-player roles need a null
team and jersey and that a wrong attribute makes a false positive. The two
readings differ when the ground truth has a null and the prediction does not.
``null_is_value=True`` (default) treats null as a value that must be matched;
``null_is_value=False`` skips attributes that are null in the ground truth.
Official numbers come from the SoccerNet evaluation code; this adapter exists
for local development loops and its tests check the maths on toy data.

Ground truth and predictions are DataFrames in the layout produced by
:func:`varlab_vision.loaders.objects_frame` (``frame``, ``track_id``, ``role``,
``team``, ``jersey``, ``x_pitch``, ``y_pitch``). The ball is excluded by
default: GSR scores athletes, and ball records have no pitch position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

TAU_M = 5.0
ALPHAS: Tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))
DEFAULT_ROLES: Tuple[str, ...] = ("player", "goalkeeper", "referee", "other")


def loc_sim(dist_m: np.ndarray, tau: float = TAU_M) -> np.ndarray:
    """Gaussian kernel on pitch distance: 1 at 0 m, 0.05 at tau."""
    d = np.asarray(dist_m, dtype=float)
    return np.exp(math.log(0.05) * (d ** 2) / (tau ** 2))


def _norm_attr(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v)
    return None if s in ("", "None", "nan") else s


def id_sim_matrix(gt: pd.DataFrame, pr: pd.DataFrame, null_is_value: bool = True) -> np.ndarray:
    """(n_gt, n_pred) binary attribute agreement."""
    g_role = [_norm_attr(v) for v in gt["role"]]
    g_team = [_norm_attr(v) for v in gt["team"]]
    g_jersey = [_norm_attr(v) for v in gt["jersey"]]
    p_role = [_norm_attr(v) for v in pr["role"]]
    p_team = [_norm_attr(v) for v in pr["team"]]
    p_jersey = [_norm_attr(v) for v in pr["jersey"]]
    out = np.zeros((len(gt), len(pr)), dtype=float)
    for i in range(len(gt)):
        for j in range(len(pr)):
            if g_role[i] != p_role[j]:
                continue
            ok = True
            for ga, pa in ((g_team[i], p_team[j]), (g_jersey[i], p_jersey[j])):
                if ga is None and not null_is_value:
                    continue
                if ga != pa:
                    ok = False
                    break
            out[i, j] = 1.0 if ok else 0.0
    return out


def similarity_matrix(gt: pd.DataFrame, pr: pd.DataFrame, tau: float = TAU_M,
                      null_is_value: bool = True) -> np.ndarray:
    """Sim = LocSim * IdSim for one frame."""
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)), dtype=float)
    g = gt[["x_pitch", "y_pitch"]].to_numpy(dtype=float)
    p = pr[["x_pitch", "y_pitch"]].to_numpy(dtype=float)
    d = np.sqrt(((g[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
    ls = loc_sim(d, tau)
    ls = np.where(np.isnan(ls), 0.0, ls)  # a missing pitch position never matches
    return ls * id_sim_matrix(gt, pr, null_is_value=null_is_value)


@dataclass
class HOTAResult:
    hota: float
    deta: float
    assa: float
    loca: float
    per_alpha: pd.DataFrame  # alpha, HOTA, DetA, AssA, TP, FN, FP
    n_gt: int
    n_pred: int
    n_frames: int
    extras: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"GS-HOTA {self.hota:.4f}  DetA {self.deta:.4f}  AssA {self.assa:.4f}  "
                f"LocA {self.loca:.4f}  (gt {self.n_gt}, pred {self.n_pred}, frames {self.n_frames})")


def _prepare(df: pd.DataFrame, roles: Sequence[str]) -> pd.DataFrame:
    need = {"frame", "track_id", "role", "team", "jersey", "x_pitch", "y_pitch"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df[df["role"].isin(list(roles))].copy()
    out = out[out["track_id"].notna()]
    out["track_id"] = out["track_id"].astype(int)
    out["frame"] = out["frame"].astype(int)
    return out.reset_index(drop=True)


def gs_hota(gt: pd.DataFrame, pred: pd.DataFrame, tau: float = TAU_M,
            roles: Sequence[str] = DEFAULT_ROLES, alphas: Sequence[float] = ALPHAS,
            null_is_value: bool = True, frames: Optional[Iterable[int]] = None) -> HOTAResult:
    """Compute GS-HOTA for one clip (or one concatenated set of clips with disjoint ids).

    ``frames`` restricts evaluation to the given frame indices (e.g. labelled
    frames only); by default every frame present in either table is scored, so
    a prediction on an unlabelled frame counts as a false positive. Filter
    predictions to labelled frames before calling when that is not wanted.
    """
    gt = _prepare(gt, roles)
    pr = _prepare(pred, roles)
    if frames is None:
        frame_ids = sorted(set(gt["frame"]).union(set(pr["frame"])))
    else:
        frame_ids = sorted(set(int(f) for f in frames))
        gt = gt[gt["frame"].isin(frame_ids)].reset_index(drop=True)
        pr = pr[pr["frame"].isin(frame_ids)].reset_index(drop=True)

    gt_ids = sorted(gt["track_id"].unique().tolist())
    pr_ids = sorted(pr["track_id"].unique().tolist())
    g_index = {t: i for i, t in enumerate(gt_ids)}
    p_index = {t: i for i, t in enumerate(pr_ids)}
    n_g, n_p = len(gt_ids), len(pr_ids)

    gt_by_frame = {f: d for f, d in gt.groupby("frame")}
    pr_by_frame = {f: d for f, d in pr.groupby("frame")}

    # Pass 1: per-frame similarities and the global alignment score.
    sims: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    potential = np.zeros((n_g, n_p), dtype=float)
    gt_count = np.zeros(n_g, dtype=float)
    pr_count = np.zeros(n_p, dtype=float)
    for f in frame_ids:
        g = gt_by_frame.get(f)
        p = pr_by_frame.get(f)
        gi = np.array([g_index[t] for t in g["track_id"]], dtype=int) if g is not None else np.zeros(0, int)
        pi = np.array([p_index[t] for t in p["track_id"]], dtype=int) if p is not None else np.zeros(0, int)
        if g is not None and p is not None:
            s = similarity_matrix(g, p, tau=tau, null_is_value=null_is_value)
        else:
            s = np.zeros((len(gi), len(pi)), dtype=float)
        sims[f] = (gi, pi, s)
        gt_count[gi] += 1
        pr_count[pi] += 1
        if s.size:
            denom = s.sum(axis=0)[None, :] + s.sum(axis=1)[:, None] - s
            with np.errstate(divide="ignore", invalid="ignore"):
                sim_iou = np.where(denom > 0, s / denom, 0.0)
            np.add.at(potential, (gi[:, None], pi[None, :]), sim_iou)
    with np.errstate(divide="ignore", invalid="ignore"):
        gas_denom = gt_count[:, None] + pr_count[None, :] - potential
        global_alignment = np.where(gas_denom > 0, potential / gas_denom, 0.0)

    # Pass 2: Hungarian matching per frame, thresholded per alpha.
    eps = np.finfo(float).eps
    tp = np.zeros(len(alphas)); fn = np.zeros(len(alphas)); fp = np.zeros(len(alphas))
    loc_sum = np.zeros(len(alphas))
    match_counts = [np.zeros((n_g, n_p), dtype=float) for _ in alphas]
    for f in frame_ids:
        gi, pi, s = sims[f]
        if len(gi) == 0 or len(pi) == 0:
            fn += len(gi)
            fp += len(pi)
            continue
        score = global_alignment[np.ix_(gi, pi)] * s
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            keep = s[rows, cols] >= alpha + eps
            r, c = rows[keep], cols[keep]
            n_tp = len(r)
            tp[a] += n_tp
            fn[a] += len(gi) - n_tp
            fp[a] += len(pi) - n_tp
            loc_sum[a] += s[r, c].sum()
            if n_tp:
                np.add.at(match_counts[a], (gi[r], pi[c]), 1.0)

    rows_out = []
    for a, alpha in enumerate(alphas):
        mc = match_counts[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            ass_a = np.where(mc > 0, mc / (gt_count[:, None] + pr_count[None, :] - mc), 0.0)
        assa = float((mc * ass_a).sum() / max(1.0, tp[a]))
        deta = float(tp[a] / max(1.0, tp[a] + fn[a] + fp[a]))
        loca = float(loc_sum[a] / max(1.0, tp[a]))
        rows_out.append({"alpha": alpha, "HOTA": math.sqrt(deta * assa), "DetA": deta, "AssA": assa,
                         "LocA": loca, "TP": tp[a], "FN": fn[a], "FP": fp[a]})
    per_alpha = pd.DataFrame(rows_out)
    return HOTAResult(
        hota=float(per_alpha["HOTA"].mean()),
        deta=float(per_alpha["DetA"].mean()),
        assa=float(per_alpha["AssA"].mean()),
        loca=float(per_alpha["LocA"].mean()),
        per_alpha=per_alpha,
        n_gt=int(len(gt)),
        n_pred=int(len(pr)),
        n_frames=len(frame_ids),
    )


def gs_hota_clips(pairs: Iterable[Tuple[str, pd.DataFrame, pd.DataFrame]], **kw) -> pd.DataFrame:
    """Score several clips; one row per clip plus a mean row (the GSR leaderboard pools
    per-clip HOTA components differently, so treat the mean as indicative)."""
    rows = []
    for name, g, p in pairs:
        r = gs_hota(g, p, **kw)
        rows.append({"clip": name, "GS-HOTA": r.hota, "DetA": r.deta, "AssA": r.assa, "LocA": r.loca,
                     "n_gt": r.n_gt, "n_pred": r.n_pred, "n_frames": r.n_frames})
    df = pd.DataFrame(rows)
    if len(df):
        mean = df[["GS-HOTA", "DetA", "AssA", "LocA"]].mean()
        df = pd.concat([df, pd.DataFrame([{"clip": "MEAN", **mean.to_dict()}])], ignore_index=True)
    return df
