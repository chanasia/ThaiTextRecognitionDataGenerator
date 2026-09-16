"""Document-style augmentations applied to the final composited image.

These target the failure modes seen on real Thai documents with off-the-shelf OCR:
  * dotted / dashed / solid form lines crossing lower vowels (สระอุ อู)
  * thin strokes losing tone marks after downscaling (screenshots, low-DPI scans)
  * JPEG artefacts, noise, uneven contrast, binarised faxes

Non-geometric ops keep the image size and pixel positions, so character bounding boxes
stay valid. Geometric ops (shear, aspect) move pixels and are only applied when the caller
allows it (i.e. not in COCO / bbox mode).

    from trdg.augment import augment
    img = augment(img, preset="doc")
"""

import io
import random as rnd
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

# probability of each op, per preset
PRESETS: Dict[str, Dict[str, float]] = {
    "off": {},
    "doc": dict(underline=0.35, border=0.15, morph=0.30, lowres=0.35, motion=0.05,
                noise=0.30, contrast=0.40, jpeg=0.40, threshold=0.05, invert=0.02,
                shear=0.10, aspect=0.15),
    "heavy": dict(underline=0.50, border=0.25, morph=0.45, lowres=0.50, motion=0.15,
                  noise=0.45, contrast=0.55, jpeg=0.50, threshold=0.10, invert=0.05,
                  shear=0.30, aspect=0.30),
}
GEOMETRIC = ("shear", "aspect")


def _bg_color(arr: np.ndarray) -> tuple:
    """Median colour of the image border = paper colour."""
    edge = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def _ink_color(arr: np.ndarray, r) -> tuple:
    """Dark grey close to the text colour, slightly randomised."""
    g = int(np.percentile(arr.mean(axis=2), 5))
    g = max(0, min(120, g + r.randint(-20, 40)))
    return (g, g, g)


# ------------------------------------------------------------------ non-geometric ops
def underline(arr: np.ndarray, r) -> np.ndarray:
    """Dotted / dashed / solid line at or under the baseline, as in printed forms."""
    h, w = arr.shape[:2]
    y = int(h * r.uniform(0.66, 0.97))
    thick = 1 if h < 48 else r.randint(1, 2)
    x0 = 0 if r.random() < 0.6 else r.randint(0, w // 3)
    x1 = w if r.random() < 0.6 else r.randint(max(x0 + w // 3, x0 + 8), w)
    color = _ink_color(arr, r)
    style = r.choice(["dotted", "dotted", "dashed", "solid"])
    out = arr.copy()
    if style == "solid":
        out[y:y + thick, x0:x1] = color
        return out
    if style == "dotted":
        dot, gap = r.randint(1, 2), r.randint(2, 5)
    else:
        dot, gap = r.randint(4, 12), r.randint(3, 8)
    x = x0 + r.randint(0, dot + gap)
    while x < x1:
        out[y:y + thick, x:min(x + dot, x1)] = color
        x += dot + gap
    return out


def border(arr: np.ndarray, r) -> np.ndarray:
    """Table-cell edges: top/bottom rules and/or vertical lines near the edges."""
    h, w = arr.shape[:2]
    color = _ink_color(arr, r)
    thick = 1 if h < 48 else r.randint(1, 2)
    out = arr.copy()
    if r.random() < 0.7:
        inset = r.randint(0, 2)
        out[inset:inset + thick, :] = color
    if r.random() < 0.7:
        inset = r.randint(0, 2)
        out[h - inset - thick:h - inset, :] = color
    if r.random() < 0.5:
        x = r.randint(0, 4)
        out[:, x:x + thick] = color
    if r.random() < 0.5:
        x = w - r.randint(1, 5)
        out[:, x:x + thick] = color
    return out


def _ink_fraction(arr: np.ndarray) -> float:
    """Share of pixels that differ clearly from the paper colour (polarity-agnostic)."""
    g = arr.mean(axis=2)
    return float((np.abs(g - np.median(g)) > 40).mean())


def morph(arr: np.ndarray, r) -> np.ndarray:
    """Thicken (ink spread) or thin (light print) strokes. Thinning is what eats tone marks."""
    dark_on_light = arr.mean() > 127
    thin = r.random() < 0.6
    if thin:
        k = r.choice([(2, 2), (2, 1), (1, 2), (2, 2)])   # 3x3 wipes out light-weight fonts
    else:
        k = r.choice([(2, 2), (2, 1), (1, 2), (3, 3), (2, 3), (3, 2)])
    kernel = np.ones(k, np.uint8)
    # erode shrinks bright regions -> thickens dark text; dilate does the opposite
    out = cv2.dilate(arr, kernel, iterations=1) if dark_on_light == thin else cv2.erode(arr, kernel, iterations=1)
    if thin and _ink_fraction(out) < 0.5 * _ink_fraction(arr):
        return arr  # strokes fell apart (thin font): keep the original
    return out


def lowres(arr: np.ndarray, r) -> np.ndarray:
    """Downscale then upscale: screenshots, 100-dpi scans, chat images."""
    h, w = arr.shape[:2]
    f = r.uniform(0.35, 0.75)
    small = cv2.resize(arr, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
    up = r.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_LINEAR])
    return cv2.resize(small, (w, h), interpolation=up)


def motion(arr: np.ndarray, r) -> np.ndarray:
    k = r.randint(3, 7)
    kernel = np.zeros((k, k), np.float32)
    if r.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= k
    return cv2.filter2D(arr, -1, kernel)


def noise(arr: np.ndarray, r) -> np.ndarray:
    out = arr.astype(np.float32)
    if r.random() < 0.7:
        sigma = r.uniform(3, 14)
        out += np.random.default_rng(r.getrandbits(32)).normal(0, sigma, out.shape)
    else:
        amount = r.uniform(0.001, 0.012)
        g = np.random.default_rng(r.getrandbits(32))
        m = g.random(out.shape[:2])
        out[m < amount / 2] = 0
        out[m > 1 - amount / 2] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast(arr: np.ndarray, r) -> np.ndarray:
    alpha = r.uniform(0.55, 1.35)
    beta = r.uniform(-40, 40)
    out = arr.astype(np.float32) * alpha + beta
    if r.random() < 0.3:  # gamma
        gamma = r.uniform(0.6, 1.6)
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if np.std(out.mean(axis=2)) < 18:  # text would be washed out: keep the original
        return arr
    return out


def jpeg(arr: np.ndarray, r) -> np.ndarray:
    q = r.randint(20, 75)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def threshold(arr: np.ndarray, r) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if r.random() < 0.5:
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block = r.choice([11, 15, 21, 31])
        b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, r.randint(2, 10))
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def invert(arr: np.ndarray, r) -> np.ndarray:
    return 255 - arr


# ------------------------------------------------------------------ geometric ops
def shear(arr: np.ndarray, r) -> np.ndarray:
    """Horizontal shear: italic-ish print or a skewed scan. Width grows to keep all pixels."""
    h, w = arr.shape[:2]
    s = r.uniform(-0.25, 0.25)
    pad = int(abs(s) * h) + 1
    m = np.float32([[1, s, pad if s < 0 else 0], [0, 1, 0]])
    return cv2.warpAffine(arr, m, (w + pad, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=_bg_color(arr))


def aspect(arr: np.ndarray, r) -> np.ndarray:
    """Stretch / squash width: condensed print, fax, resized screenshots."""
    h, w = arr.shape[:2]
    f = r.uniform(0.7, 1.35)
    return cv2.resize(arr, (max(8, int(w * f)), h), interpolation=cv2.INTER_LINEAR if f > 1 else cv2.INTER_AREA)


ORDER = [
    ("shear", shear), ("aspect", aspect),
    ("morph", morph), ("underline", underline), ("border", border),
    ("lowres", lowres), ("motion", motion), ("noise", noise), ("contrast", contrast),
    ("jpeg", jpeg), ("threshold", threshold), ("invert", invert),
]


def augment(
    image: Image.Image,
    preset: str = "doc",
    prob_scale: float = 1.0,
    geometric: bool = True,
    rng: Optional[rnd.Random] = None,
) -> Image.Image:
    """Apply the preset's ops, each with its own probability. Returns an image in the input mode."""
    probs = PRESETS.get(preset)
    if probs is None:
        raise ValueError(f"unknown augment preset '{preset}', choose from {list(PRESETS)}")
    if not probs or prob_scale <= 0:
        return image
    r = rng or rnd
    mode = image.mode
    src = np.array(image.convert("RGB"))
    arr = src
    for name, fn in ORDER:
        p = probs.get(name, 0.0) * prob_scale
        if p <= 0 or (name in GEOMETRIC and not geometric):
            continue
        if r.random() < p:
            arr = fn(arr, r)
    # Safety net: the combination must not wash the text out (light fonts + lowres + noise ...)
    if np.std(arr.mean(axis=2)) < 15 or _ink_fraction(arr) < 0.45 * _ink_fraction(src):
        arr = src
    out = Image.fromarray(arr)
    return out if mode == "RGB" else out.convert(mode)
