import argparse
import errno
import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random as rnd
import string
import sys
from multiprocessing import Pool
from tqdm import tqdm

from trdg.data_generator import FakeTextDataGenerator
from trdg.string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly,
)
from trdg.utils import load_dict, load_fonts


def margins(margin):
    margins = margin.split(",")
    if len(margins) == 1:
        return [int(margins[0])] * 4
    return [int(m) for m in margins]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic text data for text recognition."
    )
    parser.add_argument(
        "--output_dir", type=str, nargs="?", help="The output directory", default="out/"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default="",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), ar (Arabic), cn (Chinese), ja (Japanese) or hi (Hindi)",
        default="en",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created. If not specified, will use all lines from dict/input file.",
        default=None,
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False,
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1,
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images if horizontal, else the width",
        default=32,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this parameter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help='Define if the data will be "handwritten" by an RNN',
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-om",
        "--output_mask",
        type=int,
        help="Define if the generator will return masks for the text",
        default=0,
    )
    parser.add_argument(
        "-obb",
        "--output_bboxes",
        type=int,
        help="Define if the generator will return bounding boxes for the text, 1: Bounding box file, 2: Tesseract format",
        default=0,
    )
    parser.add_argument(
        "-oc",
        "--output_coco",
        action="store_true",
        help="Generate COCO format annotations (creates metadata files and splits train/val)",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distortion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0,
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distortion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0,
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1,
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=1,
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=0,
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
        default="#282828",
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=1.0,
    )
    parser.add_argument(
        "-cs",
        "--character_spacing",
        type=int,
        nargs="?",
        help="Define the width of the spaces between characters. 2 means two pixels",
        default=0,
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(5, 5, 5, 5),
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        help="Apply a tight crop around the rendered text",
        default=False,
    )
    parser.add_argument(
        "-ft", "--font", type=str, nargs="?", help="Define font to be used"
    )
    parser.add_argument(
        "-fd",
        "--font_dir",
        type=str,
        nargs="?",
        help="Define a font directory to be used",
    )
    parser.add_argument(
        "-id",
        "--image_dir",
        type=str,
        nargs="?",
        help="Define an image directory to use when background is set to image",
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images"),
    )
    parser.add_argument(
        "-ca",
        "--case",
        type=str,
        nargs="?",
        help="Generate upper or lowercase only. arguments: upper or lower. Example: --case upper",
    )
    parser.add_argument(
        "-dt", "--dict", type=str, nargs="?", help="Define the dictionary to be used"
    )
    parser.add_argument(
        "-ws",
        "--word_split",
        action="store_true",
        help="Split on words instead of on characters (preserves ligatures, no character spacing)",
        default=False,
    )
    parser.add_argument(
        "-stw",
        "--stroke_width",
        type=int,
        nargs="?",
        help="Define the width of the strokes",
        default=0,
    )
    parser.add_argument(
        "-stf",
        "--stroke_fill",
        type=str,
        nargs="?",
        help="Define the color of the contour of the strokes, if stroke_width is bigger than 0",
        default="#282828",
    )
    parser.add_argument(
        "-im",
        "--image_mode",
        type=str,
        nargs="?",
        help="Define the image mode to be used. RGB is default, L means 8-bit grayscale images, 1 means 1-bit binary images stored with one pixel per byte, etc.",
        default="RGB",
    )
    parser.add_argument(
        "-tr",
        "--train_ratio",
        type=float,
        nargs="?",
        help="Train/val split ratio when using --output_coco (default: 0.8)",
        default=0.8,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Define base and coco output paths
    base_dir = os.path.join(args.output_dir, "base")
    coco_dir = os.path.join(args.output_dir, "coco-output")

    try:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(coco_dir, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if args.dict:
        lang_dict = []
        if os.path.isfile(args.dict):
            with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
                lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
        else:
            sys.exit("Cannot open dict")
    else:
        lang_dict = load_dict(
            os.path.join(os.path.dirname(__file__), "dicts", args.language + ".txt")
        )

    # Remove duplicates from dictionary
    lang_dict = list(dict.fromkeys(lang_dict))

    if args.font_dir:
        fonts = [
            os.path.join(args.font_dir, p)
            for p in os.listdir(args.font_dir)
            if os.path.splitext(p)[1] == ".ttf"
        ]
    elif args.font:
        if os.path.isfile(args.font):
            fonts = [args.font]
        else:
            sys.exit("Cannot open font")
    else:
        fonts = load_fonts(args.language)

    # --- RAM Optimization: Input Streaming ---
    # We count lines first to support tqdm without loading everything to RAM
    string_count = 0
    if args.input_file != "":
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for _ in f: string_count += 1
        if args.count:
            string_count = min(string_count, args.count)

    # Generator to read file line-by-line (Streaming)
    def get_string_generator():
        if args.input_file != "":
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if args.count and i >= args.count:
                        break
                    yield line.strip()
        else:
            # Fallback for dict/random if not using input_file
            # Note: For 600K this part might still need optimization if used
            for s in lang_dict[:args.count if args.count else len(lang_dict)]:
                yield s

    def data_generator():
        # Bucket size: 5000 images per subfolder
        bucket_size = 5000

        for i, text_line in enumerate(get_string_generator()):
            # Logic for sub-directories (Buckets)
            bucket_num = i // bucket_size
            bucket_path = os.path.join(base_dir, f"{bucket_num:04d}")
            os.makedirs(bucket_path, exist_ok=True)

            # Checkpoint: Skip if file already exists
            file_name = f"{i}.{args.extension}"
            target_file = os.path.join(bucket_path, file_name)
            if os.path.exists(target_file):
                continue

            # Process Arabic if needed
            if args.language == "ar":
                from arabic_reshaper import ArabicReshaper
                from bidi.algorithm import get_display
                arabic_reshaper = ArabicReshaper()
                text_line = " ".join([get_display(arabic_reshaper.reshape(w)) for w in text_line.split(" ")[::-1]])

            # Case conversion
            if args.case == "upper":
                text_line = text_line.upper()
            elif args.case == "lower":
                text_line = text_line.lower()

            yield (
                i,
                text_line,
                fonts[rnd.randrange(0, len(fonts))],
                bucket_path,  # Save directly to the bucket
                args.format,
                args.extension,
                args.skew_angle,
                args.random_skew,
                args.blur,
                args.random_blur,
                args.background,
                args.distorsion,
                args.distorsion_orientation,
                args.handwritten,
                args.name_format,
                args.width,
                args.alignment,
                args.text_color,
                args.orientation,
                args.space_width,
                args.character_spacing,
                args.margins,
                args.fit,
                args.output_mask,
                args.word_split,
                args.image_dir,
                args.stroke_width,
                args.stroke_fill,
                args.image_mode,
                args.output_bboxes,
                args.output_coco,
            )

    print(f"Generating {string_count} images into {base_dir}")

    p = Pool(args.thread_count)
    for _ in tqdm(
            p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                data_generator(),
            ),
            total=string_count,
    ):
        pass
    p.terminate()

    # Phase 2: COCO Conversion
    if args.output_coco:
        print("\n" + "=" * 50)
        print("Phase 2: Converting to COCO format (Streaming)...")
        print("=" * 50)

        from trdg.coco_generator import convert_metadata_to_coco

        # Important: convert_metadata_to_coco needs to be modified to scan buckets
        convert_metadata_to_coco(
            metadata_dir=base_dir,
            output_dir=coco_dir,
            train_ratio=args.train_ratio
        )

        print("\n" + "=" * 50)
        print("COCO dataset ready!")
        print(f"Images & Annotations at: {coco_dir}")
        print("=" * 50)


if __name__ == "__main__":
    main()