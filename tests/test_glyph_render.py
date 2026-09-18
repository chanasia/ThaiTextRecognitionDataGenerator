"""Fails if the renderer stops drawing the glyphs HarfBuzz shaped (needs uharfbuzz + fontTools only).

    python tests/test_glyph_render.py
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from trdg import computer_text_generator as ctg  # noqa: E402

FONTS = os.path.join(ROOT, "trdg", "fonts")


def render(text, font):
    img, mask, positions = ctg.generate(text, font, "#000000", 64, 0, 1, 0, True, False)
    return img, mask, positions


def ink(img):
    return np.asarray(img.split()[-1]) > 0


def test_thanthakhat_above_upper_vowel_is_drawn():
    # uni0E4C.small sits above sara i; the old renderer drew the character and lost it
    for font in ("th_doc/Kanit-Regular.ttf", "th_doc/IBMPlexSansThai-Regular.ttf", "th_doc/DilleniaUPC.ttf"):
        with_mark, _, pos = render("สิทธิ์", os.path.join(FONTS, font))
        without, _, _ = render("สิทธิ", os.path.join(FONTS, font))
        assert with_mark.height > without.height, font          # the mark adds a row of ink on top
        assert len(pos) == 6, (font, len(pos))                   # one entry per shaped glyph
        # role naming differs per font (uni0E4C.small vs unnamed glyphs); it must be a mark above the base
        assert pos[-1]["upper_diacritic_bbox"] or pos[-1]["upper_tone_bbox"], font


def test_lower_vowel_under_descender_consonant():
    img, _, pos = render("กตัญญู", os.path.join(FONTS, "th_doc/Kanit-Regular.ttf"))
    assert pos[-1]["lower_bbox"] is not None
    assert ink(img).any()


def test_unsupported_text_is_skipped_not_mislabelled():
    latin_only = os.path.join(FONTS, "latin", sorted(os.listdir(os.path.join(FONTS, "latin")))[0])
    img, mask, pos = ctg._generate_horizontal_text_original("สวัสดี", latin_only, "#000000", 64, 1, 0, True, False)
    assert img is None and pos == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
