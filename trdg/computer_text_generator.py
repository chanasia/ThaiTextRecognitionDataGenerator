import random as rnd
import re
from typing import Tuple, List, Dict
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
import numpy as np

from trdg.utils import get_text_width, get_text_height, get_text_bbox

# Thai upper vowels and tone marks (above baseline)
THAI_UPPER_CHARS = set('\u0E31\u0E34\u0E35\u0E36\u0E37\u0E47\u0E48\u0E49\u0E4A\u0E4B\u0E4C\u0E4D')

# Thai lower vowels (below baseline)
THAI_LOWER_CHARS = set('\u0E38\u0E39\u0E3A')

# Sara Am (special case:  splits into upper + trailing)
SARA_AM = '\u0E33'  # ำ
NIKHAHIT = '\u0E4D'  # ◌ํ (upper part)
SARA_AA = '\u0E32'  # า (trailing part)


def _decompose_thai_grapheme(grapheme: str) -> Dict:
    """
    Decompose Thai grapheme into base + upper + lower components.

    Special handling for สระอำ (U+0E33) → splits into nikhahit + sara aa

    Returns:
        {
            'base': str,           # consonant or base character
            'upper': str,          # upper vowels/tones (empty if none)
            'lower': str,          # lower vowels (empty if none)
            'trailing': str,       # trailing characters like า from ำ
            'is_sara_am': bool     # True if contains sara am
        }
    """
    if not grapheme:
        return {'base': '', 'upper': '', 'lower': '', 'trailing': '', 'is_sara_am': False}

    # Special case: Sara Am
    if SARA_AM in grapheme:
        base = grapheme.replace(SARA_AM, '')
        return {
            'base': base if base else '',
            'upper': NIKHAHIT,
            'lower': '',
            'trailing': SARA_AA,
            'is_sara_am': True
        }

    # Normal case: separate base from diacritics
    base = ''
    upper = ''
    lower = ''

    for char in grapheme:
        if char in THAI_UPPER_CHARS:
            upper += char
        elif char in THAI_LOWER_CHARS:
            lower += char
        else:
            base += char

    return {
        'base': base,
        'upper': upper,
        'lower': lower,
        'trailing': '',
        'is_sara_am': False
    }


def _measure_sara_am_components(
        image_font: ImageFont,
        sara_am_char: str,
        x_offset: int,
        y_offset: int
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Measure sara am (ำ) by splitting into upper (nikhahit) and trailing (sara aa) parts.
    Uses gap detection to accurately separate components.
    Returns:  (upper_bbox, trailing_bbox)
    """
    left, top, right, bottom = image_font.getbbox(sara_am_char)
    width = right - left
    height = bottom - top

    # Render sara am with padding
    padding = 10
    img = Image.new('L', (width + padding * 2, height + padding * 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((padding - left, padding - top), sara_am_char, fill=255, font=image_font)
    pixels = np.array(img)

    # Find all text pixels
    rows, cols = np.where(pixels > 0)

    if len(rows) == 0:
        print("    [ERROR] No pixels found in sara am!")
        return None, None

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Detect gap:  sum pixels per row
    row_sums = np.sum(pixels > 0, axis=1)

    # Find rows with minimal pixels (gap between nikhahit and sara aa)
    # Look for gap in middle 30%-70% region
    search_start = min_row + int((max_row - min_row) * 0.2)
    search_end = min_row + int((max_row - min_row) * 0.8)

    gap_row = None
    min_pixel_count = float('inf')

    for row in range(search_start, search_end):
        if row_sums[row] < min_pixel_count:
            min_pixel_count = row_sums[row]
            gap_row = row

    print(f"    [DEBUG] Gap detected at row {gap_row}, pixel count: {min_pixel_count}")
    print(f"    [DEBUG] Row range: {min_row} to {max_row}")

    # Split at gap
    # Upper part (nikhahit): rows <= gap_row
    upper_rows = rows[rows <= gap_row]
    upper_cols = cols[rows <= gap_row]

    if len(upper_rows) > 0:
        upper_bbox = (
            x_offset + upper_cols.min() - padding + left,
            y_offset + upper_rows.min() - padding + top,
            x_offset + upper_cols.max() + 1 - padding + left,  # +1 for inclusive
            y_offset + upper_rows.max() + 1 - padding + top
        )
    else:
        upper_bbox = None

    # Lower part (sara aa): rows > gap_row
    lower_rows = rows[rows > gap_row]
    lower_cols = cols[rows > gap_row]

    if len(lower_rows) > 0:
        trailing_bbox = (
            x_offset + lower_cols.min() - padding + left,
            y_offset + lower_rows.min() - padding + top,
            x_offset + lower_cols.max() + 1 - padding + left,
            y_offset + lower_rows.max() + 1 - padding + top
        )
    else:
        trailing_bbox = None

    print(f"    [_measure_sara_am_components]")
    print(f"      Upper bbox: {upper_bbox}")
    print(f"      Trailing bbox: {trailing_bbox}")

    return upper_bbox, trailing_bbox


def _measure_component(
        image_font: ImageFont,
        base_char: str,
        component_char: str,
        x_offset: int,
        y_offset: int
) -> Tuple[int, int, int, int]:
    """
    Measure bbox of component by rendering and finding non-zero pixels.
    """
    if not component_char or not base_char:
        return None

    # Render composite to get actual pixels
    composite = base_char + component_char
    comp_left, comp_top, comp_right, comp_bottom = image_font.getbbox(composite)

    # Create temporary image to render
    width = comp_right - comp_left + 10
    height = comp_bottom - comp_top + 10

    # Render base alone
    base_img = Image.new('L', (width, height), 0)
    base_draw = ImageDraw.Draw(base_img)
    base_draw.text((5 - comp_left, 5 - comp_top), base_char, fill=255, font=image_font)
    base_pixels = np.array(base_img)

    # Render composite
    comp_img = Image.new('L', (width, height), 0)
    comp_draw = ImageDraw.Draw(comp_img)
    comp_draw.text((5 - comp_left, 5 - comp_top), composite, fill=255, font=image_font)
    comp_pixels = np.array(comp_img)

    # Find component pixels (difference)
    component_pixels = np.logical_and(comp_pixels > 0, base_pixels == 0)

    # Find bbox of component pixels
    rows, cols = np.where(component_pixels)

    if len(rows) == 0:
        return None

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Convert back to absolute coordinates
    x1 = x_offset + min_col - 5 + comp_left
    y1 = y_offset + min_row - 5 + comp_top
    x2 = x_offset + max_col - 5 + comp_left
    y2 = y_offset + max_row - 5 + comp_top

    return (x1, y1, x2, y2)


def _split_grapheme_clusters(text: str) -> List[str]:
    th_pattern = r'[\u0E00-\u0E7F][\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]*'
    clusters = []
    pos = 0

    for match in re.finditer(th_pattern, text):
        if match.start() > pos:
            clusters.extend(list(text[pos:match.start()]))
        clusters.append(match.group())
        pos = match.end()

    if pos < len(text):
        clusters.extend(list(text[pos:]))

    return clusters


def _has_upper_vowel(grapheme: str) -> bool:
    """Check if grapheme contains Thai upper vowel or tone mark"""
    return any(char in THAI_UPPER_CHARS for char in grapheme)


def _has_lower_vowel(grapheme: str) -> bool:
    """Check if grapheme contains Thai lower vowel"""
    return any(char in THAI_LOWER_CHARS for char in grapheme)


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
    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)

    graphemes = _split_grapheme_clusters(text)

    left, top, right, bottom = get_text_bbox(image_font, text)
    y_offset = -top

    if word_split:
        words = text.split(" ")
        text_parts = []
        for i, word in enumerate(words):
            text_parts.append(word)
            if i < len(words) - 1:
                text_parts.append(" ")

        part_widths = []
        for part in text_parts:
            if part == " ":
                part_widths.append(int(get_text_width(image_font, " ") * space_width))
            else:
                part_widths.append(get_text_width(image_font, part))

        text_width = sum(part_widths)
        text_height = bottom - top
    else:
        if character_spacing == 0:
            text_width = right - left
            text_height = bottom - top
        else:
            grapheme_widths = [get_text_width(image_font, g) for g in graphemes]
            text_width = sum(grapheme_widths) + character_spacing * max(0, len(graphemes) - 1)
            text_height = bottom - top

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")

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

    char_positions = []

    if word_split:
        x_offset = 0
        char_index = 0
        for part in text_parts:
            part_graphemes = _split_grapheme_clusters(part)

            for g in part_graphemes:
                g_width = get_text_width(image_font, g)
                components = _decompose_thai_grapheme(g)

                # Base bbox
                base_bbox = None
                if components['base']:
                    left, top, right, bottom = get_text_bbox(image_font, components['base'])
                    base_bbox = (x_offset, y_offset + top, x_offset + g_width, y_offset + bottom)

                # Upper vowel bbox and Trailing bbox (for sara am)
                upper_bbox = None
                trailing_bbox = None

                if components['upper']:
                    if components['is_sara_am']:
                        # ใช้ฟังก์ชันพิเศษสำหรับสระอำ
                        upper_bbox, trailing_bbox = _measure_sara_am_components(
                            image_font, g, x_offset, y_offset
                        )
                    elif components['base']:
                        upper_bbox = _measure_component(
                            image_font, components['base'], components['upper'], x_offset, y_offset
                        )
                    else:
                        # No base (standalone) - measure directly
                        left, top, right, bottom = get_text_bbox(image_font, components['upper'])
                        upper_bbox = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

                # Lower vowel bbox
                lower_bbox = None
                if components['lower']:
                    if components['base']:
                        lower_bbox = _measure_component(
                            image_font, components['base'], components['lower'], x_offset, y_offset
                        )
                    else:
                        left, top, right, bottom = get_text_bbox(image_font, components['lower'])
                        lower_bbox = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

                # Trailing bbox (for non-sara am cases)
                if components['trailing'] and not components['is_sara_am']:
                    trailing_left, trailing_top, trailing_right, trailing_bottom = get_text_bbox(
                        image_font, components['trailing']
                    )
                    base_width = get_text_width(image_font, components['base']) if components['base'] else 0
                    trailing_bbox = (
                        x_offset + base_width,
                        y_offset + trailing_top,
                        x_offset + base_width + (trailing_right - trailing_left),
                        y_offset + trailing_bottom
                    )

                if components['is_sara_am']:
                    print(f"\n=== Sara Am Debug:  '{g}' ===")
                    print(f"  Base: '{components['base']}', bbox: {base_bbox}")
                    print(f"  Upper (nikhahit): bbox: {upper_bbox}")
                    print(f"  Trailing (aa): bbox: {trailing_bbox}")
                    print(f"  x_offset: {x_offset}")

                char_positions.append({
                    "grapheme": g,
                    "base_bbox": base_bbox,
                    "upper_bbox": upper_bbox,
                    "lower_bbox": lower_bbox,
                    "trailing_bbox": trailing_bbox,
                    "is_sara_am": components['is_sara_am']
                })

                txt_img_draw.text(
                    (x_offset, y_offset),
                    g,
                    fill=fill,
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill_color,
                )
                txt_mask_draw.text(
                    (x_offset, y_offset),
                    g,
                    fill=((char_index + 1) // (255 * 255), (char_index + 1) // 255, (char_index + 1) % 255),
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill_color,
                )
                x_offset += g_width
                char_index += 1

            if part == " ":
                x_offset = x_offset - get_text_width(image_font, " ") + int(
                    get_text_width(image_font, " ") * space_width)
    else:
        if character_spacing == 0:
            x_offset = 0
            for i, g in enumerate(graphemes):
                g_width = get_text_width(image_font, g)
                components = _decompose_thai_grapheme(g)

                base_bbox = None
                if components['base']:
                    left, top, right, bottom = get_text_bbox(image_font, components['base'])
                    base_bbox = (x_offset, y_offset + top, x_offset + g_width, y_offset + bottom)

                # Upper vowel bbox and Trailing bbox (for sara am)
                upper_bbox = None
                trailing_bbox = None

                if components['upper']:
                    if components['is_sara_am']:
                        # ใช้ฟังก์ชันพิเศษสำหรับสระอำ
                        upper_bbox, trailing_bbox = _measure_sara_am_components(
                            image_font, g, x_offset, y_offset
                        )
                    elif components['base']:
                        upper_bbox = _measure_component(
                            image_font, components['base'], components['upper'], x_offset, y_offset
                        )
                    else:
                        left, top, right, bottom = get_text_bbox(image_font, components['upper'])
                        upper_bbox = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

                lower_bbox = None
                if components['lower']:
                    if components['base']:
                        lower_bbox = _measure_component(
                            image_font, components['base'], components['lower'], x_offset, y_offset
                        )
                    else:
                        left, top, right, bottom = get_text_bbox(image_font, components['lower'])
                        lower_bbox = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

                # Trailing bbox (for non-sara am cases)
                if components['trailing'] and not components['is_sara_am']:
                    trailing_left, trailing_top, trailing_right, trailing_bottom = get_text_bbox(
                        image_font, components['trailing']
                    )
                    base_width = get_text_width(image_font, components['base']) if components['base'] else 0
                    trailing_bbox = (
                        x_offset + base_width,
                        y_offset + trailing_top,
                        x_offset + base_width + (trailing_right - trailing_left),
                        y_offset + trailing_bottom
                    )

                if components['is_sara_am']:
                    print(f"\n=== Sara Am Debug: '{g}' ===")
                    print(f"  Base: '{components['base']}', bbox: {base_bbox}")
                    print(f"  Upper (nikhahit): bbox: {upper_bbox}")
                    print(f"  Trailing (aa): bbox: {trailing_bbox}")
                    print(f"  x_offset: {x_offset}")

                char_positions.append({
                    "grapheme": g,
                    "base_bbox": base_bbox,
                    "upper_bbox": upper_bbox,
                    "lower_bbox": lower_bbox,
                    "trailing_bbox": trailing_bbox,
                    "is_sara_am": components['is_sara_am']
                })

                txt_img_draw.text(
                    (x_offset, y_offset),
                    g,
                    fill=fill,
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill_color,
                )
                txt_mask_draw.text(
                    (x_offset, y_offset),
                    g,
                    fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill_color,
                )
                x_offset += g_width
        else:
            x_offset = 0
            for i, g in enumerate(graphemes):
                g_width = get_text_width(image_font, g)
                components = _decompose_thai_grapheme(g)

                base_bbox = None
                if components['base']:
                    left, top, right, bottom = get_text_bbox(image_font, components['base'])
                    base_bbox = (x_offset, y_offset + top, x_offset + g_width, y_offset + bottom)

                # Upper vowel bbox and Trailing bbox (for sara am)
                upper_bbox = None
                trailing_bbox = None

                if components['upper']:
                    if components['is_sara_am']:
                        # ใช้ฟังก์ชันพิเศษสำหรับสระอำ
                        upper_bbox, trailing_bbox = _measure_sara_am_components(
                            image_font, g, x_offset, y_offset
                        )
                    elif components['base']:
                        upper_bbox = _measure_component(
                            image_font, components['base'], components['upper'], x_offset, y_offset
                        )
                    else:
                        left, top, right, bottom = get_text_bbox(image_font, components['upper'])
                        upper_bbox = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

                lower_bbox = None
                if components['lower']:
                    if components['base']:
                        lower_bbox = _measure_component(
                            image_font, components['base'], components['lower'], x_offset, y_offset
                        )
                    else:
                        left, top, right, bottom = get_text_bbox(image_font, components['lower'])
                        lower_bbox = (x_offset, y_offset + top, x_offset + (right - left), y_offset + bottom)

                # Trailing bbox (for non-sara am cases)
                if components['trailing'] and not components['is_sara_am']:
                    trailing_left, trailing_top, trailing_right, trailing_bottom = get_text_bbox(
                        image_font, components['trailing']
                    )
                    base_width = get_text_width(image_font, components['base']) if components['base'] else 0
                    trailing_bbox = (
                        x_offset + base_width,
                        y_offset + trailing_top,
                        x_offset + base_width + (trailing_right - trailing_left),
                        y_offset + trailing_bottom
                    )

                if components['is_sara_am']:
                    print(f"\n=== Sara Am Debug: '{g}' ===")
                    print(f"  Base: '{components['base']}', bbox: {base_bbox}")
                    print(f"  Upper (nikhahit): bbox: {upper_bbox}")
                    print(f"  Trailing (aa): bbox: {trailing_bbox}")
                    print(f"  x_offset: {x_offset}")

                char_positions.append({
                    "grapheme": g,
                    "base_bbox": base_bbox,
                    "upper_bbox": upper_bbox,
                    "lower_bbox": lower_bbox,
                    "trailing_bbox": trailing_bbox,
                    "is_sara_am": components['is_sara_am']
                })

                txt_img_draw.text(
                    (x_offset, y_offset),
                    g,
                    fill=fill,
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill_color,
                )
                txt_mask_draw.text(
                    (x_offset, y_offset),
                    g,
                    fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill_color,
                )
                x_offset += g_width + character_spacing

    print(f"Char positions created at original text size: {txt_img.size}")
    print(f"Total char positions: {len(char_positions)}")

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
    image_font = ImageFont.truetype(font=font, size=font_size, layout_engine=ImageFont.Layout.RAQM)

    graphemes = _split_grapheme_clusters(text)

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
            "is_upper_vowel": _has_upper_vowel(g),
            "is_lower_vowel": _has_lower_vowel(g)
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