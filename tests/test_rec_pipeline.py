"""Smallest checks that fail if the recognizer-data pipeline breaks.

    python tests/test_rec_pipeline.py          # plain
    python -m pytest tests/                    # pytest
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from trdg.corpus import DEFAULT_CHARSET, CorpusBuilder, normalize, safe_cut  # noqa: E402
from trdg.augment import augment  # noqa: E402


def test_normalize_thai():
    assert normalize("กํา") == "กำ"                     # nikhahit+aa -> sara am
    assert normalize("a​b  c d") == "ab c d"            # zero-width, nbsp, whitespace
    assert normalize("เก้า") == "เก้า"                        # NFC keeps already-canonical text


def test_safe_cut_keeps_clusters():
    s = "สวัสดีครับวันนี้อากาศดี"
    for n in range(4, len(s)):
        cut = safe_cut(s, n)
        assert len(cut) <= n
        assert cut[-1] not in "เแโใไ", cut                          # never end on a leading vowel
        rest = s[len(cut):]
        assert rest[0] not in "ะัาำิีึืุู็่้๊๋์ํ๎ๅๆฯ", (cut, rest)   # never split before a mark
    assert safe_cut("hello world foo", 9) == "hello"              # prefers a space


def test_corpus_builder():
    words = ["สวัสดี", "ครับ", "วันนี้", "อากาศ", "ดี", "ประชุม", "ทีม", "ห้อง", "บ้าน", "ที่"]
    b = CorpusBuilder(words, ["hello", "server", "docker"], min_len=6, max_len=40, seed=7)
    lines = list(b.lines(200))
    assert len(lines) == 200
    assert len(set(lines)) == 200
    for l in lines:
        assert 6 <= len(l) <= 40, l
        assert set(l) <= DEFAULT_CHARSET, l
        assert l == normalize(l)
    # the mix must contain Thai, Latin and digits
    joined = "".join(lines)
    assert any("฀" <= c <= "๿" for c in joined)
    assert any(c.isascii() and c.isalpha() for c in joined)
    assert any(c.isdigit() for c in joined)


def test_augment_keeps_size_without_geometric():
    img = Image.fromarray(np.full((64, 320, 3), 235, np.uint8))
    for preset in ("doc", "heavy"):
        for _ in range(10):
            out = augment(img, preset=preset, prob_scale=3.0, geometric=False)
            assert out.size == img.size and out.mode == "RGB"
    assert augment(img, preset="off") is img
    gray = img.convert("L")
    assert augment(gray, preset="heavy", prob_scale=3.0).mode == "L"


def test_end_to_end_generation():
    """Renders a few mixed lines with augmentation and checks the label files (needs uharfbuzz)."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        with open(corpus, "w", encoding="utf8") as f:
            f.write("\n".join([
                "ประชุมทีม Backend วันศุกร์ 10:30",
                "ชื่อบิดา สมชาย ใจดี",
                "ราคา 1,250.00 บาท (VAT 7%)",
                "Email: somchai.p@example.co.th",
                "ที่อยู่ 1/23 ถนนตัวอย่าง อ.เมือง",
                "อยู่ที่ 52% ที่เหลือ Other",
            ]) + "\n")
        out = os.path.join(tmp, "out")
        cmd = [sys.executable, os.path.join(ROOT, "trdg", "run.py"), "-i", corpus, "-f", "48", "-e", "png",
               "-t", "1", "-aug", "heavy", "--seed", "3", "--output_dir", out,
               "-fd", os.path.join(ROOT, "trdg", "fonts", "th_doc")]
        subprocess.run(cmd, check=True, cwd=ROOT, timeout=600)

        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(out, "labels.txt"), encoding="utf8")]
        assert 4 <= len(rows) <= 6, rows
        for path, text in rows:
            p = os.path.join(out, path)
            assert os.path.isfile(p), p
            assert text == normalize(text)
            assert Image.open(p).size[1] == 48
        train = open(os.path.join(out, "train.txt"), encoding="utf8").read().splitlines()
        val = open(os.path.join(out, "val.txt"), encoding="utf8").read().splitlines()
        assert len(train) + len(val) == len(rows)
        charset = open(os.path.join(out, "charset.txt"), encoding="utf8").read().splitlines()
        assert "ก" not in charset or all(len(c) == 1 for c in charset)
        assert " " not in charset and "ท" in charset and "B" in charset


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
