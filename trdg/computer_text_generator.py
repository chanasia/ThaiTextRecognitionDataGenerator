import random as rnd
import re
from typing import Tuple, List
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from trdg.utils import get_text_width, get_text_height, get_text_bbox


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

    if word_split:
        x_offset = 0
        char_index = 0
        for part in text_parts:
            part_graphemes = _split_grapheme_clusters(part)

            for g in part_graphemes:
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
                x_offset += get_text_width(image_font, g)
                char_index += 1

            if part == " ":
                x_offset = x_offset - get_text_width(image_font, " ") + int(
                    get_text_width(image_font, " ") * space_width)
    else:
        if character_spacing == 0:
            x_offset = 0
            for i, g in enumerate(graphemes):
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
                x_offset += get_text_width(image_font, g)
        else:
            x_offset = 0
            for i, g in enumerate(graphemes):
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
                x_offset += get_text_width(image_font, g) + character_spacing

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask


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

    for i, g in enumerate(graphemes):
        g_left, g_top, g_right, g_bottom = get_text_bbox(image_font, g)
        y_offset = -g_top

        txt_img_draw.text(
            (x_offset_base, sum(grapheme_heights[0:i]) + i * character_spacing + y_offset),
            g,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        txt_mask_draw.text(
            (x_offset_base, sum(grapheme_heights[0:i]) + i * character_spacing + y_offset),
            g,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255, 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask