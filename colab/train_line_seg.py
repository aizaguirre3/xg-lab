#!/usr/bin/env python3
"""Mac-side runner: train the pitch-line segmenter on a Colab GPU in chunks, pulling the checkpoint back after each.

    python colab/train_line_seg.py --ref <commit-sha> --out data/models/line_seg --gpu L4 \
        --holdout SNGS-060 SNGS-064 SNGS-072 SNGS-099 SNGS-103 SNGS-130 --budget-min 50

Same pattern as ``run_clips.py``: one session with a budget under an hour, ``colab exec`` of
``line_seg_worker.py`` in a fresh kernel, ``prepare`` once, then ``train`` in chunks of
``--chunk-min`` minutes with ``ckpt.pt``, ``best.pt`` and ``log.jsonl`` downloaded after every
chunk, ``colab stop`` in a ``finally``. The held-out clips are the ones the calibration is tuned
on, so their scores are not flattered by the network having seen them. No token is needed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_clips import ColabSession, printing_runner, subprocess_runner  # noqa: E402

WORKER = HERE / "line_seg_worker.py"
REMOTE = "/content/lineseg"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="commit to clone on the VM (must be pushed)")
    p.add_argument("--out", required=True, help="local folder for the checkpoints and the log (gitignored)")
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--chunk-min", type=float, default=8.0)
    p.add_argument("--budget-min", type=float, default=50.0)
    p.add_argument("--session", default="varlab-lineseg")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    session = ColabSession(args.session, args.gpu, printing_runner if args.dry_run else subprocess_runner)
    out = Path(args.out)
    t0 = time.monotonic()
    session.new()
    try:
        session.exec_file(WORKER, timeout=120)
        session.exec_checked(f"setup({args.ref!r})", timeout=600)
        session.exec_checked(f"prepare(every={args.every}, holdout={list(args.holdout)!r})", timeout=1800)
        for name in ("ckpt.pt", "best.pt", "log.jsonl"):  # a resumed run carries on from the last chunk
            if (out / name).is_file():
                session.upload_large(out / name, f"{REMOTE}/{name}")
        while True:
            left = args.budget_min - (time.monotonic() - t0) / 60.0
            if left < args.chunk_min + 3:
                print(f"[train_line_seg] budget reached ({args.budget_min} min); resume with a new run")
                break
            res = session.exec_checked(f"r = train(minutes={args.chunk_min}, epochs={args.epochs}); "
                                    "print('EPOCH', r['epoch'], 'BEST', r['best_miou'])",
                                    timeout=args.chunk_min * 60 + 600)
            for name in ("ckpt.pt", "best.pt", "log.jsonl"):
                session.download(f"{REMOTE}/{name}", out / name)
            if f"EPOCH {args.epochs} " in res.stdout or args.dry_run:
                break
    finally:
        session.stop()
        print(session.sessions())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
