"""Download one shard of Thai Wikipedia (wikimedia/wikipedia, 20231101.th) and dump it as
plain text lines for trdg.corpus --text.

    pip install pyarrow
    python tools/fetch_thai_wiki.py                 # -> out/thai_wiki.txt (~87 MB download)
    python tools/fetch_thai_wiki.py --shard 0       # bigger shard (172 MB)

Lines are paragraphs split on Thai sentence spacing, NFC-normalised, 10-200 chars.
"""

import argparse
import os
import re
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trdg.corpus import normalize  # noqa: E402

SHARDS = [
    "train-00000-of-00003.parquet",  # 172 MB
    "train-00001-of-00003.parquet",  # 112 MB
    "train-00002-of-00003.parquet",  # 87 MB
]
BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.th/"
_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+|\s{2,}")
_JUNK = re.compile(r"[=\[\]{}|<>]")


def download(url, dst):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(tmp, dst)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--out", default="out/thai_wiki.txt")
    ap.add_argument("--cache", default="out/cache")
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--max_lines", type=int, default=0, help="stop after this many lines (0 = all)")
    a = ap.parse_args(argv)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pip install pyarrow")

    path = download(BASE + SHARDS[a.shard], os.path.join(a.cache, SHARDS[a.shard]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    seen = set()
    with open(a.out, "w", encoding="utf8", newline="\n") as f:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=2000):
            for text in batch.column(0).to_pylist():
                for para in text.split("\n"):
                    for sent in _SPLIT.split(para):
                        s = normalize(sent)
                        if not (a.min_len <= len(s) <= a.max_len) or _JUNK.search(s):
                            continue
                        if not any("฀" <= c <= "๿" for c in s):
                            continue
                        h = hash(s)
                        if h in seen:
                            continue
                        seen.add(h)
                        f.write(s + "\n")
                        n += 1
                        if a.max_lines and n >= a.max_lines:
                            print(f"wrote {n} lines -> {a.out}")
                            return 0
    print(f"wrote {n} lines -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
