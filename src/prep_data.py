"""Build the 6-class dataset WITHOUT a Kaggle login, reliably.

Source: the `freshness_fruit.zip` (3 GB, ZIP64) in the HF dataset repo
`Densu341/Fresh-rotten-fruit`. Instead of downloading all 3 GB, we let Python's
zipfile parse the central directory over a range-backed file object (ZIP64
aware), then fetch ONLY the images for our 6 target classes with one HTTP range
request per image, decompressing locally. ~1800 small images, not 3 GB.

Usage:  python src/prep_data.py --per-class 400
"""
from __future__ import annotations
import argparse, io, os, struct, time, urllib.request, zipfile, zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

URL = ("https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit/"
       "resolve/main/freshness_fruit.zip")
HDR = {"User-Agent": "Mozilla/5.0"}
TARGET = {"freshapples", "freshbanana", "freshoranges",
          "rottenapples", "rottenbanana", "rottenoranges"}
IMG_EXT = (".png", ".jpg", ".jpeg")


def ranged(start, end, retries=6, timeout=90):
    """GET bytes[start:end] inclusive, with backoff."""
    delay = 2
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, headers={**HDR, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(delay); delay = min(delay * 2, 20)


class RangeReader(io.RawIOBase):
    """Seekable file-like object over the remote zip. The tail (EOCD + central
    directory) is cached so zipfile can parse the index with one fetch."""

    def __init__(self, size, tail_bytes=6_000_000):
        self.size = size
        self.pos = 0
        self.tail_start = max(0, size - tail_bytes)
        self.tail = ranged(self.tail_start, size - 1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        start, end = self.pos, min(self.pos + n, self.size)
        if start >= self.tail_start:
            data = self.tail[start - self.tail_start:end - self.tail_start]
        else:
            data = ranged(start, end - 1)
        self.pos = end
        return data


def fetch_member(info, out_path):
    """One ranged read of a member (local header + data) -> save JPEG."""
    try:
        span = 30 + len(info.filename.encode("utf-8")) + 256 + info.compress_size
        blob = ranged(info.header_offset, info.header_offset + span - 1, timeout=60)
        if blob[0:4] != b"\x50\x4b\x03\x04":
            return False
        fnl = struct.unpack("<H", blob[26:28])[0]
        exl = struct.unpack("<H", blob[28:30])[0]
        ds = 30 + fnl + exl
        data = blob[ds:ds + info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            raw = data
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(data, -15)
        else:
            return False
        Image.open(io.BytesIO(raw)).convert("RGB").save(out_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def cls_of(fn):
    parts = fn.strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--out", default=os.path.join("data", "raw"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    for n in TARGET:
        os.makedirs(os.path.join(args.out, n), exist_ok=True)

    print("Reading zip directory from CDN ...", flush=True)
    req = urllib.request.Request(URL, method="HEAD", headers=HDR)
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    zf = zipfile.ZipFile(RangeReader(size))
    infos = zf.infolist()
    print(f"  {len(infos)} entries, zip {size/1e9:.2f} GB", flush=True)

    chosen = defaultdict(list)
    for info in infos:
        fn = info.filename
        if info.is_dir() or not fn.lower().endswith(IMG_EXT):
            continue
        c = cls_of(fn)
        if c in TARGET and len(chosen[c]) < args.per_class:
            chosen[c].append(info)
    for c in sorted(TARGET):
        print(f"  selected {c:16s} {len(chosen[c])}", flush=True)

    tasks = [(c, i, info) for c, infos_ in chosen.items() for i, info in enumerate(infos_)]
    print(f"Fetching {len(tasks)} images with {args.workers} workers ...", flush=True)
    ok = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_member, info, os.path.join(args.out, c, f"{i:04d}.jpg")): c
                for c, i, info in tasks}
        done = 0
        for fut in as_completed(futs):
            if fut.result():
                ok[futs[fut]] += 1
            done += 1
            if done % 200 == 0:
                print("  fetched", done, "/", len(tasks), flush=True)

    print("Done. Saved per class:", flush=True)
    total = 0
    for c in sorted(TARGET):
        total += ok[c]
        print(f"  {c:16s} {ok[c]}", flush=True)
    print("Total saved:", total, flush=True)


if __name__ == "__main__":
    main()
