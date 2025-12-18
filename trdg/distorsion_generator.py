import cv2
import math
import os
import random as rnd
import numpy as np
from typing import Tuple, List, Dict, Callable

from PIL import Image


def _update_bboxes_after_distortion(
        char_positions: List[Dict],
        vertical_offsets: List[int],
        horizontal_offsets: List[int],
        max_offset: int,
        vertical: bool,
        horizontal: bool
) -> List[Dict]:
    """Update bboxes to match distortion transformation by applying offsets to each corner"""
    new_chars = []

    for char in char_positions:
        new_char = char.copy()

        def distort_bbox(bbox):
            if not bbox:
                return bbox

            x1, y1, x2, y2 = bbox
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            new_corners = []

            for x, y in corners:
                new_x = x
                new_y = y

                if vertical and 0 <= x < len(vertical_offsets):
                    v_off = vertical_offsets[x]
                    new_y += max_offset + v_off
                    new_x += max_offset if horizontal else 0

                if horizontal and 0 <= new_y < len(horizontal_offsets):
                    h_off = horizontal_offsets[new_y]
                    new_x += h_off

                new_corners.append((new_x, new_y))

            if not new_corners:
                return bbox

            xs = [c[0] for c in new_corners]
            ys = [c[1] for c in new_corners]
            return (min(xs), min(ys), max(xs), max(ys))

        if 'bbox' in new_char:
            new_char['bbox'] = distort_bbox(new_char['bbox'])

        for key in ['base_bbox', 'leading_bbox', 'upper_vowel_bbox',
                    'upper_tone_bbox', 'upper_diacritic_bbox',
                    'lower_bbox', 'trailing_bbox']:
            if new_char.get(key):
                new_char[key] = distort_bbox(new_char[key])

        new_chars.append(new_char)

    return new_chars


def _apply_func_distorsion(
        image: Image,
        mask: Image,
        char_positions: List[Dict],
        vertical: bool,
        horizontal: bool,
        max_offset: int,
        func: Callable
) -> Tuple:
    """
    Apply a distortion to an image
    """

    # Nothing to do!
    if not vertical and not horizontal:
        return image, mask, char_positions

    # FIXME: From looking at the code I think both are already RGBA
    rgb_image = image.convert("RGBA")
    rgb_mask = mask.convert("RGB")

    img_arr = np.array(rgb_image)
    mask_arr = np.array(rgb_mask)

    vertical_offsets = [func(i) for i in range(img_arr.shape[1])]
    horizontal_offsets = [
        func(i)
        for i in range(
            img_arr.shape[0]
            + (
                (max(vertical_offsets) - min(min(vertical_offsets), 0))
                if vertical
                else 0
            )
        )
    ]

    new_img_arr = np.zeros(
        (
            img_arr.shape[0] + (2 * max_offset if vertical else 0),
            img_arr.shape[1] + (2 * max_offset if horizontal else 0),
            4,
        )
    )

    new_img_arr_copy = np.copy(new_img_arr)

    new_mask_arr = np.zeros(
        (
            # I keep img_arr to maximise the chance of
            # a breakage if img and mask don't match
            img_arr.shape[0] + (2 * max_offset if vertical else 0),
            img_arr.shape[1] + (2 * max_offset if horizontal else 0),
            3,
        )
    )

    new_mask_arr_copy = np.copy(new_mask_arr)

    if vertical:
        column_height = img_arr.shape[0]
        for i, o in enumerate(vertical_offsets):
            column_pos = (i + max_offset) if horizontal else i
            new_img_arr[
                max_offset + o: column_height + max_offset + o, column_pos, :
            ] = img_arr[:, i, :]
            new_mask_arr[
                max_offset + o: column_height + max_offset + o, column_pos, :
            ] = mask_arr[:, i, :]

    if horizontal:
        row_width = img_arr.shape[1]
        for i, o in enumerate(horizontal_offsets):
            if vertical:
                new_img_arr_copy[
                    i, max_offset + o: row_width + max_offset + o, :
                ] = new_img_arr[i, max_offset: row_width + max_offset, :]
                new_mask_arr_copy[
                    i, max_offset + o: row_width + max_offset + o, :
                ] = new_mask_arr[i, max_offset: row_width + max_offset, :]
            else:
                new_img_arr[
                    i, max_offset + o: row_width + max_offset + o, :
                ] = img_arr[i, :, :]
                new_mask_arr[
                    i, max_offset + o: row_width + max_offset + o, :
                ] = mask_arr[i, :, :]

    # Update char_positions to match distortion
    updated_chars = _update_bboxes_after_distortion(
        char_positions, vertical_offsets, horizontal_offsets,
        max_offset, vertical, horizontal
    )

    return (
        Image.fromarray(
            np.uint8(new_img_arr_copy if horizontal and vertical else new_img_arr)
        ).convert("RGBA"),
        Image.fromarray(
            np.uint8(new_mask_arr_copy if horizontal and vertical else new_mask_arr)
        ).convert("RGB"),
        updated_chars
    )


def sin(
        image: Image,
        mask: Image,
        char_positions: List[Dict] = None,
        vertical: bool = False,
        horizontal: bool = False
) -> Tuple:
    """
    Apply a sine distortion on one or both of the specified axis
    """
    if char_positions is None:
        char_positions = []

    max_offset = int(image.height ** 0.5)

    return _apply_func_distorsion(
        image,
        mask,
        char_positions,
        vertical,
        horizontal,
        max_offset,
        (lambda x: int(math.sin(math.radians(x)) * max_offset)),
    )


def cos(
        image: Image,
        mask: Image,
        char_positions: List[Dict] = None,
        vertical: bool = False,
        horizontal: bool = False
) -> Tuple:
    """
    Apply a cosine distortion on one or both of the specified axis
    """
    if char_positions is None:
        char_positions = []

    max_offset = int(image.height ** 0.5)

    return _apply_func_distorsion(
        image,
        mask,
        char_positions,
        vertical,
        horizontal,
        max_offset,
        (lambda x: int(math.cos(math.radians(x)) * max_offset)),
    )


def random(
        image: Image,
        mask: Image,
        char_positions: List[Dict] = None,
        vertical: bool = False,
        horizontal: bool = False
) -> Tuple:
    """
    Apply a random distortion on one or both of the specified axis
    """
    if char_positions is None:
        char_positions = []

    max_offset = int(image.height ** 0.4)

    return _apply_func_distorsion(
        image,
        mask,
        char_positions,
        vertical,
        horizontal,
        max_offset,
        (lambda x: rnd.randint(0, max_offset)),
    )