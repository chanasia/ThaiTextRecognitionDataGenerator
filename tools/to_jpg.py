"""Re-encode a trdg output folder (PNG) as JPEG and rewrite the label files.

PNG of noisy paper backgrounds barely compresses (~53 KB per 64-px line, 200k = 10.8 GB).
Measured on 300 random lines:  RGB JPEG q92 21.5 KB · gray q90 17.3 KB · gray, height 48,
q90 11.5 KB (2.3 GB per 200k). The trainer resizes to height 48 anyway, so:

    python tools/to_jpg.py out/rec out/jpg/rec --gray --height 48 -q 90 -t 8
    tar -a -cf out/rec_200k_jpg.zip -C out/jpg rec
"""

import argparse
import os
import sys
from multiprocessing import Pool

from PIL import Image

LABEL_FILES = ("labels.txt", "train.txt", "val.txt")


def _convert(job):
    src, dst, quality, gray, height = job
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im = Image.open(src).convert("L" if gray else "RGB")
        if height and im.height != height:
            im = im.resize((max(8, round(im.width * height / im.height)), height), Image.Resampling.LANCZOS)
        im.save(dst, format="JPEG", quality=quality, subsampling=0)
        return None
    except Exception as e:  # report, do not kill the pool
        return f"{src}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="trdg output dir containing base/ and labels.txt")
    ap.add_argument("dst", help="destination dir (created)")
    ap.add_argument("-q", "--quality", type=int, default=90)
    ap.add_argument("-t", "--workers", type=int, default=4)
    ap.add_argument("--gray", action="store_true", help="save 8-bit grayscale (the lines are grey anyway)")
    ap.add_argument("--height", type=int, default=0, help="resize to this height first (e.g. 48 = what the trainer uses)")
    a = ap.parse_args(argv)

    rows = []
    with open(os.path.join(a.src, "labels.txt"), encoding="utf8") as f:
        for line in f:
            path, text = line.rstrip("\n").split("\t", 1)
            rows.append((path, text))
    jobs = [(os.path.join(a.src, p), os.path.join(a.dst, os.path.splitext(p)[0] + ".jpg"), a.quality, a.gray, a.height)
            for p, _ in rows]

    os.makedirs(a.dst, exist_ok=True)
    failed = []
    with Pool(a.workers) as pool:
        for i, err in enumerate(pool.imap_unordered(_convert, jobs, chunksize=64), 1):
            if err:
                failed.append(err)
            if i % 10000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    failed_src = {e.split(":", 1)[0] for e in failed}

    for name in LABEL_FILES:
        src_file = os.path.join(a.src, name)
        if not os.path.isfile(src_file):
            continue
        with open(src_file, encoding="utf8") as f, open(os.path.join(a.dst, name), "w", encoding="utf8", newline="\n") as g:
            for line in f:
                path, text = line.rstrip("\n").split("\t", 1)
                if os.path.join(a.src, path) in failed_src:
                    continue
                g.write(f"{os.path.splitext(path)[0]}.jpg\t{text}\n")
    charset = os.path.join(a.src, "charset.txt")
    if os.path.isfile(charset):
        with open(charset, encoding="utf8") as f, open(os.path.join(a.dst, "charset.txt"), "w", encoding="utf8", newline="\n") as g:
            g.write(f.read())

    print(f"converted {len(jobs) - len(failed)} images -> {a.dst}" + (f", {len(failed)} failed" if failed else ""))
    for e in failed[:10]:
        print("  ", e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
