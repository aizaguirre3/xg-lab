"""Read single clips out of a large remote zip with HTTP range requests.

SoccerNet publishes the Game State Reconstruction data on Hugging Face as one zip
per split (``SoccerNet/SN-GSR-2025``: ``valid.zip`` is 11.2 GB for 49 clips; checked
2026-10-26, public and not gated). Downloading a whole split for every short Colab
session is wasteful, and a session killed mid-download loses everything. A zip
keeps its table of contents (the central directory) at the end of the file and
every member at a known offset, so this module reads the directory once and then
fetches only the byte ranges of the members it needs, in large blocks.

Nothing downloaded is executed. Member names are checked before anything is
written (no absolute paths, no ``..``), so a malicious archive cannot write
outside the destination folder.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple

HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
DEFAULT_BLOCK = 32 * 2 ** 20  # 32 MiB: a clip is a few hundred MB of contiguous members

Fetch = Callable[[int, int], bytes]  # (start, end_exclusive) -> bytes


def hf_dataset_file_url(repo: str, filename: str, revision: str = "main") -> str:
    return HF_RESOLVE_URL.format(repo=repo, revision=revision, filename=filename)


def http_range_fetcher(url: str, token: Optional[str] = None, timeout: float = 120.0,
                       retries: int = 5, user_agent: str = "varlab-remote-zip/1",
                       log: Optional[Callable[[str], None]] = None) -> Tuple[Fetch, int]:
    """A ``fetch(start, end)`` that issues one HTTP range request per call, and the file size.

    Redirects (Hugging Face sends a 302 to its storage) are followed by urllib, which
    keeps the ``Range`` header. The size comes from the ``Content-Range`` of a
    one-byte request. Transient failures are retried with backoff.
    """
    headers = {"User-Agent": user_agent}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def get(start: int, end_inclusive: int) -> Tuple[bytes, str]:
        req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end_inclusive}"})
        delay = 2.0
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status != 206:
                        raise IOError(f"expected 206 Partial Content, got {resp.status} for {url}")
                    return resp.read(), resp.headers.get("Content-Range", "")
            except (urllib.error.URLError, TimeoutError, ConnectionError, IOError) as e:
                if attempt == retries - 1:
                    raise IOError(f"range {start}-{end_inclusive} of {url} failed after {retries} tries: {e}") from e
                if log is not None:
                    log(f"range {start}-{end_inclusive}: {type(e).__name__}: {e}; retry {attempt + 1}/{retries - 1} in {delay:.0f} s")
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    _, cr = get(0, 0)
    m = re.match(r"bytes \d+-\d+/(\d+)", cr)
    if not m:
        raise IOError(f"no total size in Content-Range {cr!r} for {url}")
    size = int(m.group(1))

    def fetch(start: int, end: int) -> bytes:
        data, _ = get(start, end - 1)
        if len(data) != end - start:
            raise IOError(f"short read: asked {end - start} bytes at {start}, got {len(data)}")
        return data

    return fetch, size


class RangeFile:
    """A read-only, seekable file over ``fetch(start, end)`` with an LRU cache of aligned blocks.

    Implements exactly what :class:`zipfile.ZipFile` uses: ``read``, ``seek``, ``tell``,
    ``seekable``. ``bytes_fetched`` and ``requests`` count the network traffic.
    """

    def __init__(self, fetch: Fetch, size: int, block_size: int = DEFAULT_BLOCK, max_blocks: int = 4):
        self._fetch = fetch
        self.size = int(size)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self._pos = 0
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self.bytes_fetched = 0
        self.requests = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            pos = offset
        elif whence == 1:
            pos = self._pos + offset
        elif whence == 2:
            pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def _block(self, k: int) -> bytes:
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        start = k * self.block_size
        end = min(self.size, start + self.block_size)
        data = self._fetch(start, end)
        self.bytes_fetched += len(data)
        self.requests += 1
        self._cache[k] = data
        while len(self._cache) > self.max_blocks:
            self._cache.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if self._pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        out = bytearray()
        while n > 0:
            k, off = divmod(self._pos, self.block_size)
            blk = self._block(k)
            take = min(n, len(blk) - off)
            out += blk[off:off + take]
            self._pos += take
            n -= take
        return bytes(out)

    def close(self) -> None:
        self._cache.clear()


def open_remote_zip(url: str, token: Optional[str] = None, block_size: int = DEFAULT_BLOCK,
                    max_blocks: int = 4, log: Optional[Callable[[str], None]] = None) -> Tuple[zipfile.ZipFile, RangeFile]:
    fetch, size = http_range_fetcher(url, token=token, log=log)
    rf = RangeFile(fetch, size, block_size=block_size, max_blocks=max_blocks)
    return zipfile.ZipFile(rf), rf


def _safe_relpath(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts) or "\\" in name or ":" in name:
        raise ValueError(f"unsafe member name in archive: {name!r}")
    return p


def clip_index(zf: zipfile.ZipFile, labels_file: str = "Labels-GameState.json") -> Dict[str, str]:
    """``{clip_name: member_prefix}`` for every folder in the archive that holds a labels file.

    The prefix is the folder path inside the archive (for example ``valid/SNGS-021/``),
    so the code does not depend on how deep the split folders sit.
    """
    out: Dict[str, str] = {}
    for info in zf.infolist():
        p = _safe_relpath(info.filename)
        if p.name == labels_file and len(p.parts) >= 2:
            out[p.parts[-2]] = str(p.parent) + "/"
    return dict(sorted(out.items()))


def extract_members(zf: zipfile.ZipFile, prefix: str, dest_root: Path | str,
                    keep: Optional[Callable[[str], bool]] = None, strip_prefix: Optional[str] = None) -> List[Path]:
    """Extract the members under ``prefix`` in archive order (contiguous reads), safely.

    Each member is written to ``dest_root / (name without strip_prefix)``. ``keep`` filters
    member names. Returns the written files.
    """
    dest_root = Path(dest_root).resolve()
    members = [i for i in zf.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
    if keep is not None:
        members = [i for i in members if keep(i.filename)]
    members.sort(key=lambda i: i.header_offset)
    written: List[Path] = []
    for info in members:
        rel = _safe_relpath(info.filename)
        if strip_prefix:
            rel = PurePosixPath(*rel.parts[len(PurePosixPath(strip_prefix).parts):])
        target = (dest_root / Path(*rel.parts)).resolve()
        if dest_root != target and dest_root not in target.parents:
            raise ValueError(f"member {info.filename!r} would escape {dest_root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
        written.append(target)
    return written


def extract_clip(zf: zipfile.ZipFile, clip: str, dest_dir: Path | str, labels_only: bool = False,
                 index: Optional[Dict[str, str]] = None) -> Path:
    """Write one clip (``Labels-GameState.json`` and ``img1/``) to ``dest_dir/<clip>/``; return that folder."""
    index = index if index is not None else clip_index(zf)
    if clip not in index:
        raise KeyError(f"{clip} is not in the archive ({len(index)} clips)")
    prefix = index[clip]
    parent = str(PurePosixPath(prefix).parent)
    keep = (lambda name: name.endswith("Labels-GameState.json")) if labels_only else None
    extract_members(zf, prefix, dest_dir, keep=keep, strip_prefix=None if parent == "." else parent)
    return Path(dest_dir) / clip
