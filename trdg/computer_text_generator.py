"""Text image generation with Thai language support."""

import math
import random as rnd
import os
import warnings
from typing import Tuple, List, Dict

from PIL import Image, ImageColor, ImageDraw, ImageFont
import uharfbuzz as hb

from trdg.vector_engine import FontVectorEngineHB
from trdg.utils import get_text_bbox, get_text_height
from trdg.thai_utils import contains_thai, THAI_TONE_MARKS, THAI_UPPER_DIACRITICS, THAI_UPPER_VOWELS, has_upper_vowel, \
    has_lower_vowel, split_grapheme_clusters

_vector_engines = {}

def _get_vector_engine(font_path: str, size: int) -> FontVectorEngineHB:
    """Get or create a cached vector engine."""
    key = (font_path, size)
    if key not in _vector_engines:
        _vector_engines[key] = FontVectorEngineHB(font_path, size)
    return _vector_engines[key]

def _check_latin_support(font_path: str) -> bool:
    """Check if font supports Latin characters (A-Z, a-z)."""
    try:
        from fontTools.ttLib import TTFont
        ttfont = TTFont(font_path)
        cmap = ttfont.getBestCmap()
        if not cmap:
            return False

        # Check A-Z (U+0041 to U+005A)
        for code in range(0x0041, 0x005B):
            if code not in cmap:
                return False

        # Check a-z (U+0061 to U+007A)
        for code in range(0x0061, 0x007B):
            if code not in cmap:
                return False

        return True
    except:
        return False

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

def _split_text_by_script(text: str) -> List[Tuple[str, str]]:
    """Split text into segments by script (Thai/Latin).

    Returns: List of (segment, script_type) where script_type is 'thai' or 'latin'
    """
    if not text:
        return []

    segments = []
    current_segment = ""
    current_type = None

    for char in text:
        # Detect script type
        if '\u0E00' <= char <= '\u0E7F':  # Thai Unicode range
            char_type = 'thai'
        elif ('A' <= char <= 'Z') or ('a' <= char <= 'z'):
            char_type = 'latin'
        else:
            # Numbers, symbols, spaces follow previous type or default to thai
            char_type = current_type if current_type else 'thai'

        # Group consecutive same-type characters
        if char_type == current_type:
            current_segment += char
        else:
            if current_segment:
                segments.append((current_segment, current_type))
            current_segment = char
            current_type = char_type

    if current_segment:
        segments.append((current_segment, current_type))

    return segments

def _render_thai_mask_components(
        txt_mask_draw: ImageDraw.ImageDraw,
        image_font: ImageFont.FreeTypeFont,
        components: Dict,
        x_pos: int,
        y_offset: int,
        base_idx: int,
        stroke_width: int,
        stroke_fill_color: Tuple[int, int, int]
) -> None:
    """Render Thai grapheme components to mask image. (Legacy)"""
    def get_mask_color(idx: int) -> Tuple[int, int, int]:
        return ((idx + 1) // (255 * 255), (idx + 1) // 255, (idx + 1) % 255)

    mask_idx = base_idx

    if components['base'] or components['leading']:
        base_text = (components['leading'] if components['leading'] else '') + \
                   (components['base'] if components['base'] else '')
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_text,
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['upper_vowel']:
        txt_mask_draw.text(
            (x_pos, y_offset),
            components['base'] + components['upper_vowel'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['upper_diacritic']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_for_render + components['upper_diacritic'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['upper_tone']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_for_render + components['upper_tone'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['lower']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_pos, y_offset),
            base_for_render + components['lower'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        mask_idx += 1000

    if components['trailing']:
        txt_mask_draw.text(
            (x_pos, y_offset),
            (components['base'] if components['base'] else '') + components['trailing'],
            fill=get_mask_color(mask_idx),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

def _render_grapheme_to_mask(
        txt_mask_draw: ImageDraw.ImageDraw,
        image_font: ImageFont.FreeTypeFont,
        components: Dict,
        x_offset: int,
        y_offset: int,
        char_index: int,
        stroke_width: int,
        stroke_fill_color: Tuple[int, int, int]
) -> int:
    """Render Thai grapheme components to mask for word_split mode. Returns updated char_index. (Legacy)"""
    def get_mask_color(idx):
        return ((idx + 1) // (255 * 255), (idx + 1) // 255, (idx + 1) % 255)

    if components['base'] or components['leading']:
        base_text = (components['leading'] if components['leading'] else '') + \
                   (components['base'] if components['base'] else '')
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_text,
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['upper_vowel']:
        txt_mask_draw.text(
            (x_offset, y_offset),
            components['base'] + components['upper_vowel'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['upper_diacritic']:
        upper_d = components['upper_diacritic']
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_for_render + upper_d,
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['upper_tone']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_for_render + components['upper_tone'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['lower']:
        base_for_render = components['base'] if components['base'] else ''
        txt_mask_draw.text(
            (x_offset, y_offset),
            base_for_render + components['lower'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    if components['trailing']:
        txt_mask_draw.text(
            (x_offset, y_offset),
            (components['base'] if components['base'] else '') + components['trailing'],
            fill=get_mask_color(char_index),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        char_index += 1

    return char_index


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
    """
    Generate text image with optional Thai language support.

    Returns (image, mask, char_positions) tuple.
    """

    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            stroke_width,
            stroke_fill,
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
    """
    Generate horizontal text image using Vector Engine for exact layout.
    """

    # Check Latin support
    has_latin_support = _check_latin_support(font)
    has_latin_text = any(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)

    if has_latin_text and not has_latin_support:
        warnings.warn(f"Font '{font}' does not support Latin characters. Using fallback Latin font.")
        latin_font = _get_random_latin_font()
        if not latin_font:
            warnings.warn("No Latin fallback font found in fonts/latin directory.")
    else:
        latin_font = None

    # If no fallback needed, use original logic
    if not latin_font:
        return _generate_horizontal_text_original(
            text, font, text_color, font_size, space_width,
            character_spacing, fit, word_split, stroke_width, stroke_fill
        )

    # Mixed rendering with fallback
    segments = _split_text_by_script(text)

    # Render each segment separately then combine
    segment_images = []
    segment_masks = []
    all_char_positions = []
    cumulative_width = 0

    for segment_text, script_type in segments:
        segment_font = latin_font if script_type == 'latin' else font

        seg_img, seg_mask, seg_positions = _generate_horizontal_text_original(
            segment_text, segment_font, text_color, font_size, space_width,
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
    """Original horizontal text generation logic."""

    try:
        engine = _get_vector_engine(font, font_size)
    except Exception as e:
        print(f"[Error] Could not load Vector Engine for {font}: {e}")
        return None, None, []

    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)
    dumb_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.BASIC)

    # Shape Text
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = 'ltr'

    # Detect script to prevent Latin chars being treated as Thai diacritics
    has_thai = any('\u0E00' <= c <= '\u0E7F' for c in text)
    has_latin = any(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)

    if has_thai and not has_latin:
        buf.script = 'thai'
        buf.language = 'tha'
    elif has_latin and not has_thai:
        buf.script = 'latn'
        buf.language = 'en'
    else:
        buf.script = 'zyyy'  # Common script for mixed content
        buf.language = 'en'

    features = {
        "kern": True, "liga": True, "ccmp": True,
        "locl": True, "mark": True, "mkmk": True
    }

    hb.shape(engine.hb_font, buf, features)

    # Calculate Layout & Overlap Correction
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    cursor_x, cursor_y = 0, 0
    glyph_layout_data = []

    # Variables for Overlap Correction
    cluster_highest_y = -float('inf')
    overlap_padding = font_size * 0.05
    char_usage_index = 0

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        original_glyph_name = engine.ttfont.getGlyphName(info.codepoint)

        # Resolve Char (Strict Priority)
        char_str = engine.get_char_from_glyph_name(original_glyph_name)

        temp_comps = engine.decompose_glyph(original_glyph_name)
        temp_roles = [c.get('role', 'UNKNOWN') for c in temp_comps]

        is_high = False
        if temp_comps and temp_comps[0].get('bbox'):
            if temp_comps[0]['bbox'][1] > font_size * 0.4: is_high = True

        derived_role = 'BASE'
        if 'TONE' in temp_roles or (is_high and 'NIKHAHIT' not in temp_roles and 'UPPER_VOWEL' not in temp_roles):
            derived_role = 'TONE'
        elif 'NIKHAHIT' in temp_roles:
            derived_role = 'NIKHAHIT'
        elif 'UPPER_VOWEL' in temp_roles:
            derived_role = 'UPPER_VOWEL'
        elif 'SARA_AA' in temp_roles:
            derived_role = 'SARA_AA'

        # Decision
        need_recovery = False
        if not char_str:
            need_recovery = True
        elif len(char_str) == 1:
            if char_str == '\u0E33':
                need_recovery = False  # Found actual Sara Am glyph
            elif derived_role in ['TONE', 'NIKHAHIT',
                                  'UPPER_VOWEL'] and char_str not in THAI_TONE_MARKS and char_str not in THAI_UPPER_DIACRITICS and char_str not in THAI_UPPER_VOWELS:
                need_recovery = True

        # Recovery Execution
        if need_recovery:
            char_str = ""  # Reset
            if derived_role == 'TONE':
                for idx in range(char_usage_index, len(text)):
                    if text[idx] in THAI_TONE_MARKS:
                        char_str = text[idx]

                        # Handle Tone jumping over Sara Am
                        if idx == char_usage_index:
                            char_usage_index += 1
                        else:
                            # Check if we skipped Sara Am
                            skipped_am = False
                            for k in range(char_usage_index, idx):
                                if text[k] == '\u0E33': skipped_am = True
                            if not skipped_am:
                                char_usage_index = idx + 1
                        break

            elif derived_role == 'NIKHAHIT':
                for idx in range(char_usage_index, len(text)):
                    if text[idx] == '\u0E33' or text[idx] == '\u0E4D':
                        char_str = '\u0E4D'
                        # Wait for Aa if it is Am
                        if text[idx] == '\u0E4D': char_usage_index = idx + 1
                        break
            elif derived_role == 'SARA_AA':
                for idx in range(char_usage_index, len(text)):
                    if text[idx] == '\u0E32' or text[idx] == '\u0E33':
                        char_str = '\u0E32'
                        char_usage_index = idx + 1  # Consume Am/Aa
                        break
            else:
                if info.cluster < len(text):
                    char_str = text[info.cluster]
        else:
            # Normal advance
            if char_usage_index < len(text):
                if text[char_usage_index] == char_str:
                    char_usage_index += 1
                elif text[char_usage_index] == '\u0E33':
                    if char_str == '\u0E4D':
                        pass
                    elif char_str == '\u0E32':
                        char_usage_index += 1
                    elif char_str == '\u0E33':
                        char_usage_index += 1

        # HarfBuzz Units -> Pixels
        x_offset = pos.x_offset / 64
        y_offset = pos.y_offset / 64
        x_advance = pos.x_advance / 64
        y_advance = pos.y_advance / 64

        # Apply spacing adjustment
        if character_spacing > 0:
            x_advance += character_spacing

        # Base Position of this glyph
        current_draw_x = cursor_x + x_offset
        current_draw_y = cursor_y + y_offset

        # --- EXPLOSION LOGIC ---
        items_to_process = []

        # Special case: Sara Am glyph (\u0E33) needs forced explosion
        if char_str == '\u0E33':
            ni_glyph = engine.get_glyph_name_from_char('\u0E4D')
            aa_glyph = engine.get_glyph_name_from_char('\u0E32')
            if ni_glyph: items_to_process.append(('\u0E4D', ni_glyph))
            if aa_glyph: items_to_process.append(('\u0E32', aa_glyph))

        elif len(char_str) > 1:
            # Ligature
            for c in char_str:
                std_name = engine.get_glyph_name_from_char(c)
                items_to_process.append((c, std_name))
        else:
            # Standard
            std_name = engine.get_glyph_name_from_char(char_str) if char_str else original_glyph_name
            items_to_process.append((char_str, std_name))

        # Process each sub-item independently
        for sub_char, sub_glyph_name in items_to_process:
            # Measure this sub-glyph
            components = engine.decompose_glyph(sub_glyph_name)

            # OVERLAP CORRECTION LOGIC
            roles = [c.get('role', 'UNKNOWN') for c in components]
            is_base = 'BASE' in roles or 'LEADING_VOWEL' in roles or 'LOWER_VOWEL' in roles
            is_upper = 'NIKHAHIT' in roles or 'UPPER_VOWEL' in roles or 'UPPER_DIACRITIC' in roles
            is_tone = 'TONE' in roles

            if is_base:
                cluster_highest_y = -float('inf')

            ink_top = -float('inf')
            ink_bottom = float('inf')
            has_ink = False

            for c in components:
                if c.get('bbox'):
                    has_ink = True
                    ink_bottom = min(ink_bottom, c['bbox'][1])
                    ink_top = max(ink_top, c['bbox'][3])

            # Lift Logic
            item_draw_y = current_draw_y

            if has_ink:
                current_abs_top = item_draw_y + ink_top
                current_abs_bottom = item_draw_y + ink_bottom

                # Check & Fix Overlap
                if is_tone and cluster_highest_y > -float('inf'):
                    if current_abs_bottom < (cluster_highest_y + overlap_padding):
                        lift_amount = (cluster_highest_y + overlap_padding) - current_abs_bottom
                        item_draw_y += lift_amount
                        current_abs_top += lift_amount
                        current_abs_bottom += lift_amount

                # Update Highest Y
                if is_upper or is_tone:
                    cluster_highest_y = max(cluster_highest_y, current_abs_top)

            # Global Bounds Calculation
            for comp in components:
                bbox = comp.get('bbox')
                if bbox:
                    bx1, by1, bx2, by2 = bbox
                    global_x1 = current_draw_x + bx1
                    global_x2 = current_draw_x + bx2
                    global_y1 = item_draw_y + by1
                    global_y2 = item_draw_y + by2
                    min_x = min(min_x, global_x1)
                    max_x = max(max_x, global_x2)
                    min_y = min(min_y, global_y1)
                    max_y = max(max_y, global_y2)

            # Safety Fallback
            if not components and x_advance > 0:
                min_x = min(min_x, current_draw_x)
                max_x = max(max_x, current_draw_x + x_advance)
                min_y = min(min_y, current_draw_y)
                max_y = max(max_y, current_draw_y + font_size * 0.7)

            # Append flattened item to layout data
            glyph_layout_data.append({
                "glyph_name": sub_glyph_name,
                "char_str": sub_char,
                "components": components,
                "draw_x": current_draw_x,
                "draw_y": item_draw_y,
                "x_advance": x_advance
            })

        cursor_x += x_advance
        cursor_y += y_advance

    # Safety Fallback
    if min_x == float('inf'): min_x, max_x = 0, 0
    if min_y == float('inf'): min_y, max_y = 0, font_size

    #Tight Fit
    text_height = int(math.ceil(max_y - min_y))
    text_width = int(math.ceil(max_x - min_x))

    if text_width <= 0: text_width = 1
    if text_height <= 0: text_height = 1

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")

    # Colors
    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]
    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]
    stroke_fill_color = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    start_offset_x = -min_x
    font_top_y = max_y

    # Drawing Loop (Character-by-Character)
    char_positions = []
    mask_idx = 0

    for g_data in glyph_layout_data:
        base_draw_x = g_data['draw_x'] + start_offset_x
        base_draw_y = g_data['draw_y']

        # Draw using the recovered character
        glyph_char = g_data['char_str']

        if glyph_char:
            txt_img_draw.text(
                (base_draw_x, font_top_y - base_draw_y),
                glyph_char,
                fill=fill,
                font=dumb_font,  # Use Basic Layout
                anchor="ls",
                stroke_width=stroke_width,
                stroke_fill=stroke_fill_color
            )

        # Build Output Data
        components = g_data['components']

        char_info = {
            "glyph_name": g_data['glyph_name'],
            "bbox": None,
            "base_bbox": None,
            "leading_bbox": None,
            "upper_vowel_bbox": None,
            "upper_tone_bbox": None,
            "upper_diacritic_bbox": None,
            "lower_bbox": None,
            "trailing_bbox": None
        }

        all_comp_bboxes = []
        mask_color = ((mask_idx + 1) // (255 * 255), (mask_idx + 1) // 255, (mask_idx + 1) % 255)

        for comp in components:
            bbox = comp.get('bbox')
            role = comp.get('role')

            if bbox:
                bx1, by1, bx2, by2 = bbox

                img_x1 = base_draw_x + bx1
                img_x2 = base_draw_x + bx2
                img_y1 = font_top_y - (base_draw_y + by2)
                img_y2 = font_top_y - (base_draw_y + by1)

                final_bbox = (int(img_x1), int(img_y1), int(img_x2), int(img_y2))
                all_comp_bboxes.append(final_bbox)

                if role == "BASE":
                    char_info["base_bbox"] = final_bbox
                elif role == "LEADING_VOWEL":
                    char_info["leading_bbox"] = final_bbox
                elif role == "UPPER_VOWEL":
                    char_info["upper_vowel_bbox"] = final_bbox
                elif role == "TONE":
                    char_info["upper_tone_bbox"] = final_bbox
                elif role == "UPPER_DIACRITIC":
                    char_info["upper_diacritic_bbox"] = final_bbox
                elif role == "LOWER_VOWEL":
                    char_info["lower_bbox"] = final_bbox
                elif role == "TRAILING_VOWEL":
                    char_info["trailing_bbox"] = final_bbox
                elif role == "SARA_AA":
                    char_info["trailing_bbox"] = final_bbox
                elif role == "NIKHAHIT":
                    char_info["upper_diacritic_bbox"] = final_bbox

                txt_mask_draw.rectangle(final_bbox, fill=mask_color)

        if all_comp_bboxes:
            min_bx = min(b[0] for b in all_comp_bboxes)
            min_by = min(b[1] for b in all_comp_bboxes)
            max_bx = max(b[2] for b in all_comp_bboxes)
            max_by = max(b[3] for b in all_comp_bboxes)
            char_info['bbox'] = (min_bx, min_by, max_bx, max_by)

            char_positions.append(char_info)
            mask_idx += 1

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), char_positions
    else:
        return txt_img, txt_mask, char_positions

def _generate_vertical_text(
        text: str,
        font: str,
        text_color: str,
        font_size: int,
        space_width: int,
        character_spacing: int,
        fit: bool,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
) -> Tuple:
    """
    Generate vertical text image.

    Returns (image, mask, char_positions) tuple.
    """
    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)

    graphemes = split_grapheme_clusters(text)

    left, top, right, bottom = get_text_bbox(image_font, text)
    x_offset_base = -left

    space_height = int(get_text_height(image_font, " ") * space_width)

    grapheme_heights = [
        get_text_height(image_font, g) if g != " " else space_height for g in graphemes
    ]
    text_width = right - left
    text_height = sum(grapheme_heights) + character_spacing * len(graphemes)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill_color = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    char_positions = []

    for i, g in enumerate(graphemes):
        g_left, g_top, g_right, g_bottom = get_text_bbox(image_font, g)
        y_offset = -g_top
        y_pos = sum(grapheme_heights[0:i]) + i * character_spacing + y_offset

        char_positions.append({
            "grapheme": g,
            "bbox": (x_offset_base, y_pos + g_top, x_offset_base + (g_right - g_left), y_pos + g_bottom),
            "is_upper_vowel": has_upper_vowel(g),
            "is_lower_vowel": has_lower_vowel(g)
        })

        txt_img_draw.text(
            (x_offset_base, y_pos),
            g,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        txt_mask_draw.text(
            (x_offset_base, y_pos),
            g,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255, 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), char_positions
    else:
        return txt_img, txt_mask, char_positions