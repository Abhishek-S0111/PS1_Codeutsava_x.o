"""
Dataset Converter — converts NEU-DET XML and Severstal RLE masks to YOLO format.

Usage:
    python training/dataset_converter.py \
        --neu-dir /path/to/NEU-DET \
        --severstal-csv /path/to/train.csv \
        --severstal-images /path/to/train_images \
        --output-dir data/training_dataset
"""
import os
import sys
import csv
import xml.etree.ElementTree as ET
import shutil
import hashlib
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rle_to_mask(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode Severstal run-length encoding to a binary mask."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not rle or rle.strip() == "":
        return mask.reshape(shape)
    tokens = list(map(int, rle.split()))
    for i in range(0, len(tokens), 2):
        start = tokens[i] - 1
        length = tokens[i + 1]
        mask[start: start + length] = 1
    # Severstal masks are column-major
    return mask.reshape(shape, order="F")


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert binary mask to bounding box (x_min, y_min, x_max, y_max)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max), int(y_max)


def bbox_to_yolo(bbox: Tuple, img_w: int, img_h: int, class_id: int = 0) -> str:
    """Convert (x1, y1, x2, y2) to YOLO format: class cx cy w h (normalised)."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_neu_det(neu_dir: str, output_images: Path, output_labels: Path, split: str):
    """Convert NEU-DET dataset (Pascal VOC XML annotations) to YOLO."""
    neu_path = Path(neu_dir)
    img_dir = neu_path / "IMAGES"
    ann_dir = neu_path / "ANNOTATIONS"

    if not img_dir.exists():
        # Try alternative directory structures
        for alt in ["images", "JPEGImages"]:
            if (neu_path / alt).exists():
                img_dir = neu_path / alt
                break
    if not ann_dir.exists():
        for alt in ["annotations", "Annotations"]:
            if (neu_path / alt).exists():
                ann_dir = neu_path / alt
                break

    if not img_dir.exists() or not ann_dir.exists():
        print(f"  [WARN] NEU-DET directories not found in {neu_dir}")
        return 0

    count = 0
    xml_files = list(ann_dir.glob("*.xml"))
    for xml_file in xml_files:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()

        filename_el = root.find("filename")
        if filename_el is None:
            continue
        filename = filename_el.text

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        labels = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            x1 = int(float(bndbox.find("xmin").text))
            y1 = int(float(bndbox.find("ymin").text))
            x2 = int(float(bndbox.find("xmax").text))
            y2 = int(float(bndbox.find("ymax").text))
            labels.append(bbox_to_yolo((x1, y1, x2, y2), img_w, img_h, class_id=0))

        if labels:
            # Copy image
            src_img = img_dir / filename
            if not src_img.exists():
                src_img = img_dir / filename.replace(".jpg", ".bmp").replace(".png", ".bmp")
            if src_img.exists():
                dest_name = f"neu_{src_img.stem}"
                dest_img = output_images / split / f"{dest_name}{src_img.suffix}"
                dest_lbl = output_labels / split / f"{dest_name}.txt"
                shutil.copy2(str(src_img), str(dest_img))
                with open(dest_lbl, "w") as f:
                    f.write("\n".join(labels) + "\n")
                count += 1

    return count


def convert_severstal(
    csv_path: str, img_dir: str, output_images: Path, output_labels: Path, split: str
):
    """Convert Severstal Steel Defect Detection dataset (RLE masks) to YOLO."""
    csv_file = Path(csv_path)
    images_path = Path(img_dir)

    if not csv_file.exists():
        print(f"  [WARN] Severstal CSV not found: {csv_path}")
        return 0

    # Parse CSV: ImageId_ClassId, EncodedPixels
    entries: Dict[str, List[str]] = {}
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 2:
                continue
            img_class = row[0]
            rle = row[1]
            img_id = img_class.rsplit("_", 1)[0]
            if rle.strip():
                if img_id not in entries:
                    entries[img_id] = []
                entries[img_id].append(rle)

    count = 0
    for img_id, rle_list in entries.items():
        img_file = images_path / img_id
        if not img_file.exists():
            continue

        if Image:
            with Image.open(img_file) as im:
                img_w, img_h = im.size
        else:
            img_w, img_h = 1600, 256  # Default Severstal dimensions

        labels = []
        for rle in rle_list:
            mask = rle_to_mask(rle, (img_h, img_w))
            if mask.sum() < 10:
                continue
            bbox = mask_to_bbox(mask)
            labels.append(bbox_to_yolo(bbox, img_w, img_h, class_id=0))

        if labels:
            dest_name = f"sev_{Path(img_id).stem}"
            dest_img = output_images / split / f"{dest_name}{Path(img_id).suffix}"
            dest_lbl = output_labels / split / f"{dest_name}.txt"
            shutil.copy2(str(img_file), str(dest_img))
            with open(dest_lbl, "w") as f:
                f.write("\n".join(labels) + "\n")
            count += 1

    return count


def create_split(output_dir: Path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Re-split images into train/val/test if all were dumped into train/."""
    train_imgs = output_dir / "images" / "train"
    train_lbls = output_dir / "labels" / "train"

    files = sorted(train_imgs.glob("*.*"))
    random.seed(42)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for f in files[n_train: n_train + n_val]:
        stem = f.stem
        suffix = f.suffix
        shutil.move(str(f), str(output_dir / "images" / "val" / f.name))
        lbl = train_lbls / f"{stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), str(output_dir / "labels" / "val" / f"{stem}.txt"))

    for f in files[n_train + n_val:]:
        stem = f.stem
        suffix = f.suffix
        shutil.move(str(f), str(output_dir / "images" / "test" / f.name))
        lbl = train_lbls / f"{stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), str(output_dir / "labels" / "test" / f"{stem}.txt"))


def check_duplicates(output_dir: Path):
    """Check for near-duplicate images using file hash."""
    hashes = {}
    dups = []
    for img in (output_dir / "images").rglob("*.*"):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            h = hashlib.md5(img.read_bytes()).hexdigest()
            if h in hashes:
                dups.append((str(img), hashes[h]))
            else:
                hashes[h] = str(img)
    return dups


def main():
    parser = argparse.ArgumentParser(description="Convert defect datasets to YOLO format")
    parser.add_argument("--neu-dir", type=str, default="", help="Path to NEU-DET dataset")
    parser.add_argument("--severstal-csv", type=str, default="", help="Path to Severstal train.csv")
    parser.add_argument("--severstal-images", type=str, default="", help="Path to Severstal images")
    parser.add_argument("--output-dir", type=str, default="data/training_dataset")
    args = parser.parse_args()

    output = Path(args.output_dir)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = 0

    if args.neu_dir:
        print(f"Converting NEU-DET from {args.neu_dir} ...")
        n = convert_neu_det(args.neu_dir, output / "images", output / "labels", "train")
        print(f"  Converted {n} NEU-DET images")
        total += n

    if args.severstal_csv:
        print(f"Converting Severstal from {args.severstal_csv} ...")
        n = convert_severstal(
            args.severstal_csv, args.severstal_images,
            output / "images", output / "labels", "train",
        )
        print(f"  Converted {n} Severstal images")
        total += n

    if total == 0:
        print("No datasets provided. Use --neu-dir and/or --severstal-csv.")
        return

    print("Splitting into train/val/test ...")
    create_split(output)

    print("Checking for duplicates ...")
    dups = check_duplicates(output)
    if dups:
        print(f"  Found {len(dups)} duplicate(s):")
        for d in dups[:5]:
            print(f"    {d[0]}  ==  {d[1]}")
    else:
        print("  No duplicates found.")

    # Write dataset.yaml for YOLO training
    yaml_content = f"""# Steel Defect Detection Dataset
# Generated by dataset_converter.py
path: {output.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['defect']
"""
    with open(output / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    for split in ["train", "val", "test"]:
        n_imgs = len(list((output / "images" / split).glob("*.*")))
        n_lbls = len(list((output / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")

    print(f"\nDone! Total: {total} images. Dataset YAML: {output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
