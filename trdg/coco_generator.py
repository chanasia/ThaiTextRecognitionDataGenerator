import json
import os
import random
from typing import List, Tuple, Dict
from datetime import datetime


def char_boxes_to_word_polygon(char_bboxes: List[Tuple[int, int, int, int]]) -> List[float]:
    """
    Convert character-level bounding boxes to word-level polygon.
    Handles Thai upper/lower vowels by creating convex hull around all characters.
    """
    if not char_bboxes:
        return []

    # Find min/max coordinates across all character boxes
    all_x1 = [bbox[0] for bbox in char_bboxes]
    all_y1 = [bbox[1] for bbox in char_bboxes]
    all_x2 = [bbox[2] for bbox in char_bboxes]
    all_y2 = [bbox[3] for bbox in char_bboxes]

    min_x = min(all_x1)
    min_y = min(all_y1)
    max_x = max(all_x2)
    max_y = max(all_y2)

    # Create polygon (clockwise from top-left)
    polygon = [
        min_x, min_y,  # top-left
        max_x, min_y,  # top-right
        max_x, max_y,  # bottom-right
        min_x, max_y  # bottom-left
    ]

    return polygon


def create_coco_annotation(
        metadata: Dict,
        annotation_id: int,
        category_id: int = 1
) -> Tuple[Dict, List[Dict]]:
    image_info = {
        "id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "width": metadata["width"],
        "height": metadata["height"]
    }

    annotations = []
    current_ann_id = annotation_id

    # 1. Word-level annotation
    word_polygon = char_boxes_to_word_polygon(metadata["char_bboxes"])

    if word_polygon:
        min_x = min(word_polygon[0:: 2])
        min_y = min(word_polygon[1:: 2])
        max_x = max(word_polygon[0:: 2])
        max_y = max(word_polygon[1:: 2])
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        area = bbox_width * bbox_height

        annotations.append({
            "id": current_ann_id,
            "image_id": metadata["image_id"],
            "category_id": 1,
            "segmentation": [word_polygon],
            "bbox": [min_x, min_y, bbox_width, bbox_height],
            "area": area,
            "iscrowd": 0
        })
        current_ann_id += 1

    if "char_positions" in metadata:
        for char_pos in metadata["char_positions"]:
            component_keys = [
                "base_bbox", "leading_bbox", "upper_vowel_bbox",
                "upper_tone_bbox", "upper_diacritic_bbox",
                "lower_bbox", "trailing_bbox"
            ]

            for key in component_keys:
                if char_pos.get(key):
                    bbox = char_pos[key]
                    char_polygon = [
                        bbox[0], bbox[1],
                        bbox[2], bbox[1],
                        bbox[2], bbox[3],
                        bbox[0], bbox[3]
                    ]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    if width > 0 and height > 0:
                        annotations.append({
                            "id": current_ann_id,
                            "image_id": metadata["image_id"],
                            "category_id": 2,
                            "segmentation": [char_polygon],
                            "bbox": [bbox[0], bbox[1], width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        current_ann_id += 1

    return image_info, annotations


def save_coco_json(
        output_path: str,
        images: List[Dict],
        annotations: List[Dict],
        categories: List[Dict] = None
):
    """Save COCO format JSON file."""
    if categories is None:
        categories = [
            {"id": 1, "name": "word"},
            {"id": 2, "name": "character"}
        ]

    coco_format = {
        "info": {
            "description": "Thai Text Detection Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=2)

    print(f"Saved COCO annotations to {output_path}")
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")


def convert_metadata_to_coco(
        metadata_dir: str,
        output_dir: str,
        train_ratio: float = 0.8
):
    """
    Convert all metadata files to COCO format by scanning buckets (sub-folders).
    Uses os.scandir for memory efficiency on large datasets.
    """

    # --- Step 1: Collect all metadata file paths using streaming scan ---
    # We need a list of paths to perform a random shuffle for train/val split.
    print(f"Scanning {metadata_dir} for metadata files...")
    all_metadata_paths = []

    # Iterate through bucket folders (0000, 0001, ...)
    with os.scandir(metadata_dir) as buckets:
        for bucket in buckets:
            if bucket.is_dir():
                with os.scandir(bucket.path) as files:
                    for f in files:
                        if f.is_file() and f.name.endswith("_metadata.json"):
                            all_metadata_paths.append(f.path)

    if not all_metadata_paths:
        print(f"No metadata files found in {metadata_dir}")
        return

    # Shuffle for random split
    random.shuffle(all_metadata_paths)

    # Split train/val
    split_idx = int(len(all_metadata_paths) * train_ratio)
    train_files = all_metadata_paths[:split_idx]
    val_files = all_metadata_paths[split_idx:]

    print(f"Total files found: {len(all_metadata_paths)}")
    print(f"Splitting: Train={len(train_files)}, Val={len(val_files)}")

    # --- Step 2: Process datasets ---
    def process_file_list(file_list, start_ann_id):
        images = []
        annotations = []
        current_ann_id = start_ann_id

        for meta_file in file_list:
            try:
                with open(meta_file, "r", encoding="utf8") as f:
                    metadata = json.load(f)

                # Update relative path for file_name if images are in subfolders
                # This ensures the COCO json points correctly to 'base/000x/img.jpg'
                rel_path = os.path.relpath(meta_file, metadata_dir)
                folder_name = os.path.dirname(rel_path)
                metadata["file_name"] = os.path.join("base", folder_name, metadata["file_name"])

                image_info, anns = create_coco_annotation(metadata, current_ann_id)

                if anns:
                    images.append(image_info)
                    annotations.extend(anns)
                    current_ann_id += len(anns)
            except Exception as e:
                print(f"Error processing {meta_file}: {e}")
                continue

        return images, annotations, current_ann_id

    # Process Train Set
    print("Processing Train set...")
    train_images, train_anns, next_id = process_file_list(train_files, 1)

    # Process Val Set
    print("Processing Val set...")
    val_images, val_anns, _ = process_file_list(val_files, next_id)

    # --- Step 3: Save results ---
    save_coco_json(
        os.path.join(output_dir, "annotations", "train.json"),
        train_images,
        train_anns
    )

    save_coco_json(
        os.path.join(output_dir, "annotations", "val.json"),
        val_images,
        val_anns
    )


if __name__ == "__main__":
    # Example usage
    convert_metadata_to_coco(
        metadata_dir="dataset/thai_text/base",
        output_dir="dataset/thai_text/coco-output",
        train_ratio=0.8
    )