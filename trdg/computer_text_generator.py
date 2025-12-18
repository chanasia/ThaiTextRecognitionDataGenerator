"""Text image generation with Thai language support and Character-level Fallback."""

import math
import random as rnd
import os
import warnings
from typing import Tuple, List, Dict

from PIL import Image, ImageColor, ImageDraw, ImageFont
import uharfbuzz as hb
from fontTools.ttLib import TTFont

from trdg.vector_engine import FontVectorEngineHB
from trdg.utils import get_text_bbox, get_text_height
from trdg.thai_utils import contains_thai, THAI_TONE_MARKS, THAI_UPPER_DIACRITICS, THAI_UPPER_VOWELS, has_upper_vowel, \
    has_lower_vowel, split_grapheme_clusters

_vector_engines = {}
_font_cmap_cache = {}  # Cache for supported characters in each font

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

def _get_random_latin_font(latin_font_dir: str = "fonts/latin") -> str:
    """Get random Latin font from directory."""
    try:
        if not os.path.exists(latin_font_dir):
            return None
        fonts = [f for f in os.listdir(latin_font_dir) if f.endswith(('.ttf', '.otf'))]
        if not fonts:
            return None
        return os.path.join(latin_font_dir, rnd.choice(fonts))
    except:
        return None

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

    # Mixed rendering with fallback
    latin_fallback_font = _get_random_latin_font()
    if not latin_fallback_font:
        warnings.warn("Fallback needed but no Latin font found in fonts/latin. Tofu might occur.")
        return _generate_horizontal_text_original(
            text, font, text_color, font_size, space_width,
            character_spacing, fit, word_split, stroke_width, stroke_fill
        )

    segment_images = []
    segment_masks = []
    all_char_positions = []
    cumulative_width = 0

    for seg_text, is_supported in segments:
        seg_font = font if is_supported else latin_fallback_font

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
    """Original horizontal text generation logic using Vector Engine."""

    try:
        engine = _get_vector_engine(font, font_size)
    except Exception as e:
        print(f"[Error] Could not load Vector Engine for {font}: {e}")
        return None, None, []

    dumb_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.BASIC)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = 'ltr'

    # Auto-detect script for HarfBuzz shaping
    has_thai = any('\u0E00' <= c <= '\u0E7F' for c in text)
    if has_thai:
        buf.script = 'thai'
        buf.language = 'tha'
    else:
        buf.script = 'latn'
        buf.language = 'en'

    features = {"kern": True, "liga": True, "ccmp": True, "locl": True, "mark": True, "mkmk": True}
    hb.shape(engine.hb_font, buf, features)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    cursor_x, cursor_y = 0, 0
    glyph_layout_data = []
    cluster_highest_y = -float('inf')
    overlap_padding = font_size * 0.05
    char_usage_index = 0

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        original_glyph_name = engine.ttfont.getGlyphName(info.codepoint)
        char_str = engine.get_char_from_glyph_name(original_glyph_name)
        temp_comps = engine.decompose_glyph(original_glyph_name)
        temp_roles = [c.get('role', 'UNKNOWN') for c in temp_comps]

        is_high = False
        if temp_comps and temp_comps[0].get('bbox'):
            if temp_comps[0]['bbox'][1] > font_size * 0.4: is_high = True

        derived_role = 'BASE'
        if 'TONE' in temp_roles or (is_high and 'NIKHAHIT' not in temp_roles and 'UPPER_VOWEL' not in temp_roles):
            derived_role = 'TONE'
        elif 'NIKHAHIT' in temp_roles: derived_role = 'NIKHAHIT'
        elif 'UPPER_VOWEL' in temp_roles: derived_role = 'UPPER_VOWEL'
        elif 'SARA_AA' in temp_roles: derived_role = 'SARA_AA'

        need_recovery = False
        if not char_str: need_recovery = True
        elif len(char_str) == 1:
            if char_str == '\u0E33': need_recovery = False
            elif derived_role in ['TONE', 'NIKHAHIT', 'UPPER_VOWEL'] and char_str not in THAI_TONE_MARKS and char_str not in THAI_UPPER_DIACRITICS and char_str not in THAI_UPPER_VOWELS:
                need_recovery = True

        if need_recovery:
            char_str = ""
            if derived_role == 'TONE':
                for idx in range(char_usage_index, len(text)):
                    if text[idx] in THAI_TONE_MARKS:
                        char_str = text[idx]
                        if idx == char_usage_index: char_usage_index += 1
                        else:
                            skipped_am = any(text[k] == '\u0E33' for k in range(char_usage_index, idx))
                            if not skipped_am: char_usage_index = idx + 1
                        break
            elif derived_role == 'NIKHAHIT':
                for idx in range(char_usage_index, len(text)):
                    if text[idx] == '\u0E33' or text[idx] == '\u0E4D':
                        char_str = '\u0E4D'
                        if text[idx] == '\u0E4D': char_usage_index = idx + 1
                        break
            elif derived_role == 'SARA_AA':
                for idx in range(char_usage_index, len(text)):
                    if text[idx] == '\u0E32' or text[idx] == '\u0E33':
                        char_str = '\u0E32'
                        char_usage_index = idx + 1
                        break
            else:
                if info.cluster < len(text): char_str = text[info.cluster]
        else:
            if char_usage_index < len(text):
                if text[char_usage_index] == char_str: char_usage_index += 1
                elif text[char_usage_index] == '\u0E33':
                    if char_str == '\u0E4D': pass
                    elif char_str in ['\u0E32', '\u0E33']: char_usage_index += 1

        x_offset = pos.x_offset / 64
        y_offset = pos.y_offset / 64
        x_advance = pos.x_advance / 64
        y_advance = pos.y_advance / 64
        if character_spacing > 0: x_advance += character_spacing

        current_draw_x, current_draw_y = cursor_x + x_offset, cursor_y + y_offset
        items_to_process = []
        if char_str == '\u0E33':
            ni_glyph = engine.get_glyph_name_from_char('\u0E4D')
            aa_glyph = engine.get_glyph_name_from_char('\u0E32')
            if ni_glyph: items_to_process.append(('\u0E4D', ni_glyph))
            if aa_glyph: items_to_process.append(('\u0E32', aa_glyph))
        elif len(char_str) > 1:
            for c in char_str: items_to_process.append((c, engine.get_glyph_name_from_char(c)))
        else:
            items_to_process.append((char_str, engine.get_glyph_name_from_char(char_str) if char_str else original_glyph_name))

        for sub_char, sub_glyph_name in items_to_process:
            components = engine.decompose_glyph(sub_glyph_name)
            roles = [c.get('role', 'UNKNOWN') for c in components]
            is_base = any(r in roles for r in ['BASE', 'LEADING_VOWEL', 'LOWER_VOWEL'])
            is_upper = any(r in roles for r in ['NIKHAHIT', 'UPPER_VOWEL', 'UPPER_DIACRITIC'])
            is_tone = 'TONE' in roles
            if is_base: cluster_highest_y = -float('inf')

            ink_top, ink_bottom, has_ink = -float('inf'), float('inf'), False
            for c in components:
                if c.get('bbox'):
                    has_ink = True
                    ink_bottom = min(ink_bottom, c['bbox'][1])
                    ink_top = max(ink_top, c['bbox'][3])

            item_draw_y = current_draw_y
            if has_ink:
                current_abs_top, current_abs_bottom = item_draw_y + ink_top, item_draw_y + ink_bottom
                if is_tone and cluster_highest_y > -float('inf'):
                    if current_abs_bottom < (cluster_highest_y + overlap_padding):
                        lift = (cluster_highest_y + overlap_padding) - current_abs_bottom
                        item_draw_y += lift
                        current_abs_top += lift
                if is_upper or is_tone: cluster_highest_y = max(cluster_highest_y, current_abs_top)

            for comp in components:
                bbox = comp.get('bbox')
                if bbox:
                    min_x, max_x = min(min_x, current_draw_x + bbox[0]), max(max_x, current_draw_x + bbox[2])
                    min_y, max_y = min(min_y, item_draw_y + bbox[1]), max(max_y, item_draw_y + bbox[3])

            if not components and x_advance > 0:
                min_x, max_x = min(min_x, current_draw_x), max(max_x, current_draw_x + x_advance)
                min_y, max_y = min(min_y, current_draw_y), max(max_y, current_draw_y + font_size * 0.7)

            glyph_layout_data.append({"glyph_name": sub_glyph_name, "char_str": sub_char, "components": components, "draw_x": current_draw_x, "draw_y": item_draw_y, "x_advance": x_advance})

        cursor_x += x_advance
        cursor_y += y_advance

    if min_x == float('inf'): min_x, max_x, min_y, max_y = 0, 1, 0, font_size
    text_width, text_height = int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))
    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))
    txt_img_draw, txt_mask_draw = ImageDraw.Draw(txt_img), ImageDraw.Draw(txt_mask, mode="RGB")

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    fill = tuple(rnd.randint(min(colors[0][i], colors[-1][i]), max(colors[0][i], colors[-1][i])) for i in range(3))
    s_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_fill_color = tuple(rnd.randint(min(s_colors[0][i], s_colors[-1][i]), max(s_colors[0][i], s_colors[-1][i])) for i in range(3))

    start_offset_x, font_top_y = -min_x, max_y
    char_positions, mask_idx = [], 0

    for g_data in glyph_layout_data:
        base_draw_x, base_draw_y = g_data['draw_x'] + start_offset_x, g_data['draw_y']
        glyph_char = g_data['char_str']
        if glyph_char:
            txt_img_draw.text((base_draw_x, font_top_y - base_draw_y), glyph_char, fill=fill, font=dumb_font, anchor="ls", stroke_width=stroke_width, stroke_fill=stroke_fill_color)

        char_info = {"glyph_name": g_data['glyph_name'], "bbox": None, "base_bbox": None, "leading_bbox": None, "upper_vowel_bbox": None, "upper_tone_bbox": None, "upper_diacritic_bbox": None, "lower_bbox": None, "trailing_bbox": None}
        all_comp_bboxes, mask_color = [], ((mask_idx + 1) // (255 * 255), (mask_idx + 1) // 255, (mask_idx + 1) % 255)

        for comp in g_data['components']:
            bbox, role = comp.get('bbox'), comp.get('role')
            if bbox:
                f_bbox = (int(base_draw_x + bbox[0]), int(font_top_y - (base_draw_y + bbox[3])), int(base_draw_x + bbox[2]), int(font_top_y - (base_draw_y + bbox[1])))
                all_comp_bboxes.append(f_bbox)
                role_map = {"BASE": "base_bbox", "LEADING_VOWEL": "leading_bbox", "UPPER_VOWEL": "upper_vowel_bbox", "TONE": "upper_tone_bbox", "UPPER_DIACRITIC": "upper_diacritic_bbox", "LOWER_VOWEL": "lower_bbox", "TRAILING_VOWEL": "trailing_bbox", "SARA_AA": "trailing_bbox", "NIKHAHIT": "upper_diacritic_bbox"}
                if role in role_map: char_info[role_map[role]] = f_bbox
                txt_mask_draw.rectangle(f_bbox, fill=mask_color)

        if all_comp_bboxes:
            char_info['bbox'] = (min(b[0] for b in all_comp_bboxes), min(b[1] for b in all_comp_bboxes), max(b[2] for b in all_comp_bboxes), max(b[3] for b in all_comp_bboxes))
            char_positions.append(char_info)
            mask_idx += 1

    return (txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), char_positions) if fit else (txt_img, txt_mask, char_positions)

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