"""Colab-side functions for training the jersey-number reader (``varlab_vision/jersey.py``).

Sent with ``colab exec -f`` AFTER ``line_seg_worker.py`` (whose ``setup`` and ``_download`` it
uses) by ``colab/train_jersey.py``, which then calls

    setup(ref)
    prepare_jersey(every=5, holdout=[...])  # torso crops of every labelled person, with the track's number
    train_jersey(minutes=..., epochs=...)   # checkpoint after every epoch; per-track accuracy on the holdout

No horizontal flip: a mirrored number is another number (or none).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

JWORK = Path("/content/jersey")
ROLES = {"player", "goalkeeper", "referee"}  # referees have no number: the "none" examples


def _jersey_clip(args):
    zip_path, clip, prefix, every, min_h = args
    import zipfile

    import cv2
    import numpy as np

    from varlab_vision.jersey import torso

    out = JWORK / "data" / clip
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        raw = json.loads(zf.read(prefix + "Labels-GameState.json"))
        name_of = {im["image_id"]: im["file_name"] for im in raw["images"]}
        frames = sorted(set(name_of.values()))
        keep = {f for k, f in enumerate(frames) if k % every == 0}
        by_frame = {}
        for a in raw["annotations"]:
            at = a.get("attributes") or {}
            if a.get("supercategory") != "object" or at.get("role") not in ROLES:
                continue
            fname = name_of[a["image_id"]]
            if fname in keep and (a.get("bbox_image") or {}).get("h", 0) >= min_h:
                by_frame.setdefault(fname, []).append(a)
        for fname, anns in by_frame.items():
            img = cv2.imdecode(np.frombuffer(zf.read(prefix + "img1/" + fname), np.uint8), cv2.IMREAD_COLOR)
            for a in anns:
                b = a["bbox_image"]
                x0, y0 = max(0, int(b["x"])), max(0, int(b["y"]))
                x1, y1 = min(img.shape[1], int(b["x"] + b["w"])), min(img.shape[0], int(b["y"] + b["h"]))
                if x1 - x0 < 8 or y1 - y0 < 16:
                    continue
                path = out / f"{fname[:-4]}_{a['track_id']}.jpg"
                cv2.imwrite(str(path), torso(img[y0:y1, x0:x1]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                jersey = (a.get("attributes") or {}).get("jersey")
                rows.append({"path": str(path), "clip": clip, "track": a["track_id"],
                             "jersey": None if a["attributes"].get("role") == "referee" else jersey})
    return rows


def prepare_jersey(every: int = 5, holdout=(), min_h: int = 40, split: str = "train", workers: int = 8,
                   dataset: str = "SoccerNet/SN-GSR-2025") -> dict:
    import zipfile
    from multiprocessing import get_context

    from varlab_vision.remote_zip import clip_index, hf_dataset_file_url

    JWORK.mkdir(parents=True, exist_ok=True)
    zip_path = JWORK / f"{split}.zip"
    t0 = time.time()
    _download(hf_dataset_file_url(dataset, f"{split}.zip"), zip_path)  # noqa: F821 (from line_seg_worker)
    with zipfile.ZipFile(zip_path) as zf:
        index = clip_index(zf)
    jobs = [(str(zip_path), c, p, every, min_h) for c, p in index.items()]
    with get_context("fork").Pool(workers) as pool:
        rows = [r for part in pool.map(_jersey_clip, jobs) for r in part]
    hold = set(holdout)
    for r in rows:
        r["split"] = "val" if r["clip"] in hold else "train"
    (JWORK / "index.json").write_text(json.dumps(rows))
    n_num = sum(r["jersey"] not in (None, "") for r in rows)
    print(f"prepared {len(rows)} crops ({n_num} with a number) of {len(index)} clips in {time.time() - t0:.0f} s", flush=True)
    return {"crops": len(rows), "with_number": n_num}


def train_jersey(minutes: float = 10.0, epochs: int = 15, batch: int = 128, lr: float = 1e-3, workers: int = 8) -> dict:
    import random

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from varlab_vision.jersey import build_model, decide, encode, number_log_probs, to_input

    rows = json.loads((JWORK / "index.json").read_text())

    class Crops(Dataset):
        def __init__(self, items, augment):
            self.items, self.augment = items, augment

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            r = self.items[i]
            img = cv2.imread(r["path"])
            if self.augment:
                img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-20, 20), 0, 255).astype(np.uint8)
                if random.random() < 0.3:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
            n, d1, d2 = encode(r["jersey"])
            return torch.from_numpy(to_input(img)), torch.tensor([n, d1, d2])

    tr = DataLoader(Crops([r for r in rows if r["split"] == "train"], True), batch_size=batch, shuffle=True,
                    num_workers=workers, drop_last=True)
    val_rows = [r for r in rows if r["split"] == "val"]
    va = DataLoader(Crops(val_rows, False), batch_size=batch, shuffle=False, num_workers=workers)
    dev = "cuda"
    model = build_model(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ckpt = JWORK / "ckpt.pt"
    state = {"epoch": 0, "best": -1.0}
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"]); state = saved["state"]
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    total_steps = epochs * len(tr)
    t0 = time.time()
    log = []
    while state["epoch"] < epochs and time.time() - t0 < minutes * 60:
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr:
            step = state["epoch"] * len(tr) + nb
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + np.cos(np.pi * min(1.0, step / total_steps)))
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                ln, d1, d2 = model(x)
                loss = ce(ln.float(), y[:, 0]) + ce(d1.float(), y[:, 1]) + ce(d2.float(), y[:, 2])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        model.eval()
        lps = []
        with torch.inference_mode():
            for x, _y in va:
                ln, d1, d2 = (torch.log_softmax(t.float(), 1).cpu().numpy() for t in model(x.to(dev)))
                lps.append(number_log_probs(ln, d1, d2))
        lp = np.concatenate(lps) if lps else np.zeros((0, 101))
        tracks = {}
        for r, l in zip(val_rows, lp):
            tracks.setdefault((r["clip"], r["track"]), (r["jersey"], []))[1].append(l)
        numbered = [(lab, decide(np.array(ls))) for lab, ls in tracks.values() if lab not in (None, "")]
        answered = [(lab, got) for lab, got in numbered if got is not None]
        acc = sum(str(int(lab)) == got for lab, got in answered) / max(1, len(answered))
        coverage = len(answered) / max(1, len(numbered))
        state["epoch"] += 1
        row = {"epoch": state["epoch"], "loss": round(tot / max(1, nb), 4), "track_precision": round(acc, 4),
               "track_coverage": round(coverage, 4), "numbered_tracks": len(numbered), "seconds": round(time.time() - t0, 1)}
        log.append(row)
        with open(JWORK / "log.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        score = acc * coverage
        if score > state["best"]:
            state["best"] = score
            torch.save({"model": model.state_dict()}, JWORK / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, ckpt)
        print(json.dumps(row), flush=True)
    return {"epoch": state["epoch"], "best": state["best"]}
