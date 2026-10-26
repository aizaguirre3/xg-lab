"""Reading clips out of a remote zip with range requests (no network: the "remote" is in memory)."""
from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path

import pytest

from varlab_vision import remote_zip as rz


def _zip_bytes(members: dict, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _gsr_like_zip() -> bytes:
    """The SN-GSR-2025 layout: top-level clip folders plus a sequences_info.json."""
    members = {"sequences_info.json": json.dumps({"version": "1.3"})}
    for clip in ("SNGS-001", "SNGS-002"):
        members[f"{clip}/Labels-GameState.json"] = json.dumps({"info": {"name": clip}})
        for i in range(1, 4):
            members[f"{clip}/img1/{i:06d}.jpg"] = bytes([i]) * (500 + 37 * i)  # odd sizes cross block edges
    return _zip_bytes(members)


def _range_file(data: bytes, block_size: int = 64, max_blocks: int = 2) -> rz.RangeFile:
    return rz.RangeFile(lambda s, e: data[s:e], len(data), block_size=block_size, max_blocks=max_blocks)


def test_range_file_reads_seeks_and_caches_like_a_file():
    data = bytes(range(256)) * 10
    rf = _range_file(data, block_size=100, max_blocks=2)
    assert rf.read(5) == data[:5] and rf.tell() == 5
    rf.seek(250)
    assert rf.read(120) == data[250:370]  # spans three blocks
    rf.seek(-10, 2)
    assert rf.read() == data[-10:] and rf.read(4) == b""
    rf.seek(3, 0); rf.seek(4, 1)
    assert rf.read(1) == data[7:8]
    n = rf.requests
    rf.seek(7); rf.read(1)
    assert rf.requests == n  # block 0 still cached
    with pytest.raises(ValueError):
        rf.seek(-1)


def test_clip_index_and_extract_clip_reproduce_the_archive(tmp_path: Path):
    data = _gsr_like_zip()
    zf = zipfile.ZipFile(_range_file(data))
    index = rz.clip_index(zf)
    assert index == {"SNGS-001": "SNGS-001/", "SNGS-002": "SNGS-002/"}
    out = rz.extract_clip(zf, "SNGS-002", tmp_path, index=index)
    assert out == tmp_path / "SNGS-002"
    ref = zipfile.ZipFile(io.BytesIO(data))
    for i in range(1, 4):
        assert (out / "img1" / f"{i:06d}.jpg").read_bytes() == ref.read(f"SNGS-002/img1/{i:06d}.jpg")
    assert json.loads((out / "Labels-GameState.json").read_text())["info"]["name"] == "SNGS-002"
    assert not (tmp_path / "SNGS-001").exists()
    with pytest.raises(KeyError):
        rz.extract_clip(zf, "SNGS-999", tmp_path, index=index)


def test_labels_only_skips_the_frames(tmp_path: Path):
    zf = zipfile.ZipFile(_range_file(_gsr_like_zip()))
    out = rz.extract_clip(zf, "SNGS-001", tmp_path, labels_only=True)
    assert (out / "Labels-GameState.json").is_file() and not (out / "img1").exists()


def test_a_clip_is_fetched_in_few_large_requests(tmp_path: Path):
    data = _gsr_like_zip()
    rf = _range_file(data, block_size=1 << 20, max_blocks=4)
    zf = zipfile.ZipFile(rf)
    rz.extract_clip(zf, "SNGS-001", tmp_path)
    assert rf.requests <= 2 and rf.bytes_fetched <= len(data)


@pytest.mark.parametrize("bad", ["../evil.txt", "SNGS-001/../../evil.txt", "/abs/evil.txt", "C:/evil.txt",
                                 "SNGS-001\\..\\evil.txt"])
def test_unsafe_member_names_are_refused(tmp_path: Path, bad: str):
    zf = zipfile.ZipFile(_range_file(_zip_bytes({bad: b"x", "SNGS-001/Labels-GameState.json": b"{}"})))
    with pytest.raises(ValueError):
        rz.clip_index(zf)
    with pytest.raises(ValueError):
        rz.extract_members(zf, "", tmp_path / "dest")
    assert not (tmp_path / "evil.txt").exists()


class _FakeResponse:
    def __init__(self, data: bytes, start: int, end: int, total: int, status: int = 206):
        self._data = data
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_range_fetcher_parses_the_size_retries_and_checks_lengths(monkeypatch):
    blob = bytes(range(200))
    calls = {"n": 0, "fail_once": True}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        assert req.headers["User-agent"].startswith("varlab-remote-zip")
        assert "Authorization" not in req.headers
        rng = req.headers["Range"]
        s, e = (int(v) for v in rng.split("=")[1].split("-"))
        if s == 50 and calls["fail_once"]:
            calls["fail_once"] = False
            raise urllib.error.URLError("transient")
        return _FakeResponse(blob[s:e + 1], s, e, len(blob))

    monkeypatch.setattr(rz.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rz.time, "sleep", lambda s: None)
    fetch, size = rz.http_range_fetcher("https://example.invalid/valid.zip")
    assert size == 200
    assert fetch(50, 60) == blob[50:60]  # first try fails, the retry succeeds
    assert fetch(0, 200) == blob

    def short(req, timeout=None):
        s, e = (int(v) for v in req.headers["Range"].split("=")[1].split("-"))
        return _FakeResponse(blob[s:e], s, e, len(blob))  # one byte short

    monkeypatch.setattr(rz.urllib.request, "urlopen", short)
    with pytest.raises(IOError):
        rz.http_range_fetcher("https://example.invalid/valid.zip")[0](10, 20)


def test_http_range_fetcher_sends_the_token_only_when_given(monkeypatch):
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.headers.get("Authorization"))
        return _FakeResponse(b"x", 0, 0, 1)

    monkeypatch.setattr(rz.urllib.request, "urlopen", fake_urlopen)
    rz.http_range_fetcher("https://example.invalid/f.zip", token="hf_abc")
    assert seen == ["Bearer hf_abc"]


def test_hf_url():
    assert rz.hf_dataset_file_url("SoccerNet/SN-GSR-2025", "valid.zip") == \
        "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/valid.zip"
