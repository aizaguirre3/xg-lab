#!/usr/bin/env python3
"""Mac-side runner: train the jersey-number reader on a Colab GPU, pulling the checkpoint back after each chunk.

    python colab/train_jersey.py --ref <commit-sha> --out data/models/jersey --gpu L4 \
        --holdout SNGS-060 SNGS-064 SNGS-072 SNGS-099 SNGS-103 --budget-min 45

Same pattern as ``train_line_seg.py``: ``line_seg_worker.py`` and then ``jersey_worker.py`` are
exec'd into one fresh kernel (the second uses the first's ``setup`` and ``_download``),
``prepare_jersey`` once, ``train_jersey`` in chunks with ``ckpt.pt``, ``best.pt`` and ``log.jsonl``
downloaded after each, ``colab stop`` in a ``finally``.
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

REMOTE = "/content/jersey"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--chunk-min", type=float, default=8.0)
    p.add_argument("--budget-min", type=float, default=45.0)
    p.add_argument("--session", default="varlab-jersey")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    session = ColabSession(args.session, args.gpu, printing_runner if args.dry_run else subprocess_runner)
    out = Path(args.out)
    t0 = time.monotonic()
    session.new()
    try:
        session.exec_file(HERE / "line_seg_worker.py", timeout=120)
        session.exec_file(HERE / "jersey_worker.py", timeout=120)
        session.exec_checked(f"setup({args.ref!r})", timeout=600)
        session.exec_checked(f"prepare_jersey(every={args.every}, holdout={list(args.holdout)!r})", timeout=1800)
        for name in ("ckpt.pt", "best.pt", "log.jsonl"):
            if (out / name).is_file():
                session.upload(out / name, f"{REMOTE}/{name}")
        while True:
            if args.budget_min - (time.monotonic() - t0) / 60.0 < args.chunk_min + 3:
                print(f"[train_jersey] budget reached ({args.budget_min} min); resume with a new run")
                break
            res = session.exec_checked(f"r = train_jersey(minutes={args.chunk_min}, epochs={args.epochs}); "
                                    "print('EPOCH', r['epoch'], 'BEST', r['best'])", timeout=args.chunk_min * 60 + 600)
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
