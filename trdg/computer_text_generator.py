"""Text image generation with Thai language support and Character-level Fallback."""

import io
import math
import random as rnd
import os
from typing import Tuple, List, Dict

from PIL import Image, ImageColor, ImageDraw, ImageFont
import uharfbuzz as hb
from fontTools.ttLib import TTFont

from trdg.vector_engine import FontVectorEngineHB
from trdg.utils import get_text_bbox, get_text_height
from trdg.thai_utils import has_upper_vowel, has_lower_vowel, split_grapheme_clusters

_vector_engines = {}
_font_cmap_cache = {}  # Cache for supported characters in each font
_fallback_font_list = None
_FONT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

def _get_vector_engine(font_path: str, size: int) -> FontVectorEngineHB:
    """Get or create a cached vector engine."""
    key = (font_path, size)
    if key not in _vector_engines:
        _vector_engines[key] = FontVectorEngineHB(font_path, size)
    return _vector_engines[key]

def _get_font_cmap(font_path: str) -> set:
    """Get a set of all supported character codepoints for a given font."""
    if font_path not in _font_cmap_cache:
        try:
            ttfont = TTFont(font_path)
            cmap = ttfont.getBestCmap()
            _font_cmap_cache[font_path] = set(cmap.keys()) if cmap else set()
        except Exception as e:
            print(f"[Warning] Could not read cmap for {font_path}: {e}")
            _font_cmap_cache[font_path] = set()
    return _font_cmap_cache[font_path]

def _fallback_candidates() -> List[str]:
    """All bundled fonts, Thai first, then Latin (paths resolved from the package, not the CWD)."""
    global _fallback_font_list
    if _fallback_font_list is None:
        found = []
        for sub in ("th", "th_doc", "latin"):
            d = os.path.join(_FONT_ROOT, sub)
            if os.path.isdir(d):
                found += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith((".ttf", ".otf"))]
        _fallback_font_list = found
    return _fallback_font_list

def _get_fallback_font(segment: str, exclude: str) -> str:
    """Random font that covers every non-space character of `segment`, or None."""
    needed = {ord(c) for c in segment if not c.isspace()}
    if not needed:
        return None
    ok = [p for p in _fallback_candidates() if p != exclude and needed <= _get_font_cmap(p)]
    return rnd.choice(ok) if ok else None

def _split_text_by_font_support(text: str, primary_font_path: str) -> List[Tuple[str, bool]]:
    """
    Splits text based on whether characters are supported by the primary font.
    Returns: List of (segment, is_supported)
    """
    if not text:
        return []

    cmap = _get_font_cmap(primary_font_path)
    segments = []
    current_segment = ""
    current_support = None

    for char in text:
        # Check if font supports this specific character
        # Space (U+0020) is usually treated as supported to maintain context
        char_code = ord(char)
        is_supported = (char_code in cmap) or (char == ' ')

        if current_support is None:
            current_support = is_supported
            current_segment = char
        elif is_supported == current_support:
            current_segment += char
        else:
            segments.append((current_segment, current_support))
            current_segment = char
            current_support = is_supported

    if current_segment:
        segments.append((current_segment, current_support))

    return segments

def generate(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        orientation: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        word_split: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """Generate text image with automatic font fallback based on character support."""

    if orientation == 0:
        return _generate_horizontal_text(
            text, font, text_color, font_size, space_width,
            character_spacing, fit, word_split, stroke_width, stroke_fill
        )
    elif orientation == 1:
        # Vertical generation (simplified fallback could be added here if needed)
        return _generate_vertical_text(
            text, font, text_color, font_size, space_width,
            character_spacing, fit, stroke_width, stroke_fill
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))

def _generate_horizontal_text(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        word_split: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """Horizontal text generation with character-level font fallback logic."""

    # Split text into segments that the font can render and those it cannot
    segments = _split_text_by_font_support(text, font)

    # Check if any segment needs a fallback
    needs_fallback = any(not supported for _, supported in segments)

    if not needs_fallback:
        # Optimization: If all characters are supported, use original logic directly
        return _generate_horizontal_text_original(
            text, font, text_color, font_size, space_width,
            character_spacing, fit, word_split, stroke_width, stroke_fill
        )

    # Mixed rendering with per-segment fallback fonts
    segment_images = []
    segment_masks = []
    all_char_positions = []
    cumulative_width = 0

    for seg_text, is_supported in segments:
        if is_supported:
            seg_font = font
        else:
            seg_font = _get_fallback_font(seg_text, font)
            if seg_font is None:
                # No bundled font can draw this: caller skips the sample instead of rendering tofu
                return None, None, []

        seg_img, seg_mask, seg_positions = _generate_horizontal_text_original(
            seg_text, seg_font, text_color, font_size, space_width,
            character_spacing, fit, word_split, stroke_width, stroke_fill
        )

        if seg_img:
            segment_images.append(seg_img)
            segment_masks.append(seg_mask)

            # Adjust positions with cumulative offset
            for pos in seg_positions:
                if pos.get('bbox'):
                    x1, y1, x2, y2 = pos['bbox']
                    pos['bbox'] = (x1 + cumulative_width, y1, x2 + cumulative_width, y2)
                for key in ['base_bbox', 'leading_bbox', 'upper_vowel_bbox',
                           'upper_tone_bbox', 'upper_diacritic_bbox', 'lower_bbox', 'trailing_bbox']:
                    if pos.get(key):
                        x1, y1, x2, y2 = pos[key]
                        pos[key] = (x1 + cumulative_width, y1, x2 + cumulative_width, y2)
                all_char_positions.append(pos)

            cumulative_width += seg_img.width

    if not segment_images:
        return None, None, []

    # Combine segments
    max_height = max(img.height for img in segment_images)
    total_width = sum(img.width for img in segment_images)

    combined_img = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))
    combined_mask = Image.new("RGB", (total_width, max_height), (0, 0, 0))

    x_offset = 0
    for img, mask in zip(segment_images, segment_masks):
        # Using simple top-alignment. Baseline alignment would be better but requires more engine data.
        combined_img.paste(img, (x_offset, 0), img)
        combined_mask.paste(mask, (x_offset, 0))
        x_offset += img.width

    if fit:
        bbox = combined_img.getbbox()
        if bbox:
            return combined_img.crop(bbox), combined_mask.crop(bbox), all_char_positions

    return combined_img, combined_mask, all_char_positions

_GID_BASE = 0xF0000   # supplementary PUA: chr(_GID_BASE + glyph id) draws that glyph
_glyph_fonts = {}
_glyph_font_bytes = {}


def _glyph_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Pillow font whose cmap maps U+F0000+gid -> glyph gid.

    Pillow has no draw-by-glyph-index API, so a copy of the font gets a cmap in which every
    glyph id is reachable as one codepoint. Drawing chr(_GID_BASE + gid) with the BASIC layout
    then rasterizes exactly the glyph HarfBuzz picked (mark variants, PUA positional forms,
    ligatures) without any re-shaping or glyph-name guessing.
    """
    key = (font_path, size)
    if key not in _glyph_fonts:
        if font_path not in _glyph_font_bytes:
            from fontTools.ttLib.tables._c_m_a_p import cmap_format_12, table__c_m_a_p
            tt = TTFont(font_path)
            sub = cmap_format_12(12)
            sub.platformID, sub.platEncID, sub.language = 3, 10, 0
            sub.cmap = {_GID_BASE + gid: name for gid, name in enumerate(tt.getGlyphOrder())}
            cmap = table__c_m_a_p()
            cmap.tableVersion, cmap.tables = 0, [sub]
            tt["cmap"] = cmap
            buf = io.BytesIO()
            tt.save(buf)
            _glyph_font_bytes[font_path] = buf.getvalue()
        _glyph_fonts[key] = ImageFont.truetype(
            io.BytesIO(_glyph_font_bytes[font_path]), size, layout_engine=ImageFont.Layout.BASIC)
    return _glyph_fonts[key]


_ROLE_KEY = {"BASE": "base_bbox", "LEADING_VOWEL": "leading_bbox", "UPPER_VOWEL": "upper_vowel_bbox",
             "TONE": "upper_tone_bbox", "UPPER_DIACRITIC": "upper_diacritic_bbox", "LOWER_VOWEL": "lower_bbox",
             "TRAILING_VOWEL": "trailing_bbox", "SARA_AA": "trailing_bbox", "NIKHAHIT": "upper_diacritic_bbox"}


def _random_color(spec: str):
    colors = [ImageColor.getrgb(c) for c in spec.split(",")]
    return tuple(rnd.randint(min(colors[0][i], colors[-1][i]), max(colors[0][i], colors[-1][i])) for i in range(3))


def _generate_horizontal_text_original(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        word_split: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """Shape with HarfBuzz, draw every shaped glyph by glyph id at the shaped position.

    The image shows exactly the glyphs the shaper chose, so image == label by construction:
    the label is always the input text. Anything the font cannot draw faithfully (.notdef,
    a blank glyph for a visible character, a dotted-circle for an orphan mark) returns None
    so the caller skips the sample instead of writing a wrong label.
    """
    try:
        engine = _get_vector_engine(font, font_size)
        gfont = _glyph_font(font, font_size)
    except Exception as e:
        print(f"[Error] Could not load font {font}: {e}")
        return None, None, []

    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = "ltr"
    if any("฀" <= c <= "๿" for c in text):
        buf.script, buf.language = "thai", "tha"
    else:
        buf.script, buf.language = "latn", "en"
    # liga off: "fi"/"fl" ligature glyphs have no single character and would vanish from the image
    hb.shape(engine.hb_font, buf, {"kern": True, "liga": False, "ccmp": True, "locl": True, "mark": True, "mkmk": True})

    glyphs = []                       # (gid, glyph name, origin x, origin y, ink bbox) in y-down image space
    x = y = 0.0
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gid = info.codepoint
        name = engine.ttfont.getGlyphName(gid)
        if gid == 0 or engine.reverse_cmap.get(name) == "◌":
            return None, None, []
        gx, gy = x + pos.x_offset / 64, -(y + pos.y_offset / 64)
        l, t, r, b = gfont.getbbox(chr(_GID_BASE + gid), anchor="ls", stroke_width=stroke_width)
        ink = None
        if r > l and b > t:
            ink = (gx + l, gy + t, gx + r, gy + b)
            x0, y0, x1, y1 = min(x0, ink[0]), min(y0, ink[1]), max(x1, ink[2]), max(y1, ink[3])
        elif not text[info.cluster].isspace():
            return None, None, []
        glyphs.append((gid, name, gx, gy, ink))
        x += pos.x_advance / 64 + (character_spacing if character_spacing > 0 else 0)
        y += pos.y_advance / 64
    if x0 == float("inf"):
        return None, None, []

    width, height = int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))
    txt_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (width, height), (0, 0, 0))
    draw, mask_draw = ImageDraw.Draw(txt_img), ImageDraw.Draw(txt_mask, mode="RGB")
    fill, stroke_color = _random_color(text_color), _random_color(stroke_fill)

    char_positions = []
    for gid, name, gx, gy, ink in glyphs:
        ox, oy = gx - x0, gy - y0
        draw.text((ox, oy), chr(_GID_BASE + gid), fill=fill, font=gfont, anchor="ls",
                  stroke_width=stroke_width, stroke_fill=stroke_color)
        if ink is None:
            continue
        n = len(char_positions) + 1
        mask_color = (n // (255 * 255), n // 255, n % 255)
        info = {"glyph_name": name, "bbox": None, "base_bbox": None, "leading_bbox": None, "upper_vowel_bbox": None,
                "upper_tone_bbox": None, "upper_diacritic_bbox": None, "lower_bbox": None, "trailing_bbox": None}
        boxes = []
        for comp in engine.decompose_glyph(name):
            bb, role = comp.get("bbox"), comp.get("role")
            if not bb:
                continue
            box = (int(ox + bb[0]), int(oy - bb[3]), int(ox + bb[2]), int(oy - bb[1]))
            boxes.append(box)
            if role in _ROLE_KEY:
                info[_ROLE_KEY[role]] = box
            mask_draw.rectangle(box, fill=mask_color)
        if not boxes:
            boxes = [(int(ink[0] - x0), int(ink[1] - y0), int(ink[2] - x0), int(ink[3] - y0))]
            mask_draw.rectangle(boxes[0], fill=mask_color)
        info["bbox"] = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        char_positions.append(info)

    if fit:
        bbox = txt_img.getbbox()
        return txt_img.crop(bbox), txt_mask.crop(bbox), char_positions
    return txt_img, txt_mask, char_positions


def _generate_vertical_text(text: str, font: str, text_color: str, font_size: int, space_width: int, character_spacing: int, fit: bool, stroke_width: int = 0, stroke_fill: str = "#282828") -> Tuple:
    """Vertical generation (Placeholder for simplified vertical logic)."""
    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)
    graphemes = split_grapheme_clusters(text)
    left, top, right, bottom = get_text_bbox(image_font, text)
    x_offset_base = -left
    space_height = int(get_text_height(image_font, " ") * space_width)
    g_heights = [get_text_height(image_font, g) if g != " " else space_height for g in graphemes]
    t_w, t_h = right - left, sum(g_heights) + character_spacing * len(graphemes)
    txt_img = Image.new("RGBA", (t_w, t_h), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (t_w, t_h), (0, 0, 0, 0))
    txt_img_draw, txt_mask_draw = ImageDraw.Draw(txt_img), ImageDraw.Draw(txt_mask)
    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    fill = tuple(rnd.randint(colors[0][i], colors[-1][i]) for i in range(3))
    char_positions = []
    for i, g in enumerate(graphemes):
        g_l, g_t, g_r, g_b = get_text_bbox(image_font, g)
        y_pos = sum(g_heights[0:i]) + i * character_spacing - g_t
        char_positions.append({"grapheme": g, "bbox": (x_offset_base, y_pos + g_t, x_offset_base + (g_r - g_l), y_pos + g_b), "is_upper_vowel": has_upper_vowel(g), "is_lower_vowel": has_lower_vowel(g)})
        txt_img_draw.text((x_offset_base, y_pos), g, fill=fill, font=image_font, stroke_width=stroke_width, stroke_fill=stroke_fill)
        txt_mask_draw.text((x_offset_base, y_pos), g, fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255, 255), font=image_font, stroke_width=stroke_width, stroke_fill=stroke_fill)
    return (txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), char_positions) if fit else (txt_img, txt_mask, char_positions)