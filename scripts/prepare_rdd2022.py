from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalise_class(raw: str) -> int | None:
    token = (raw or "").strip().upper()
    for code, idx in CLASS_MAP.items():
        if token == code or token.startswith(code + "_") or token.startswith(code + " "):
            return idx
    return None


def get_image_dimensions(xml_root: ET.Element, image_path: Path) -> tuple[int, int]:
    size = xml_root.find("size")
    if size is not None:
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))
        if width > 0 and height > 0:
            return width, height
    with Image.open(image_path) as im:
        return im.size


def voc_to_yolo(xml_path: Path, image_path: Path) -> tuple[list[str], Counter[str]]:
    root = ET.parse(xml_path).getroot()
    width, height = get_image_dimensions(root, image_path)
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for obj in root.findall("object"):
        class_idx = normalise_class(obj.findtext("name", ""))
        if class_idx is None:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
        except ValueError:
            continue
        xmin = min(max(xmin, 0.0), float(width))
        xmax = min(max(xmax, 0.0), float(width))
        ymin = min(max(ymin, 0.0), float(height))
        ymax = min(max(ymax, 0.0), float(height))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        counts[CLASS_NAMES[class_idx]] += 1
    return lines, counts


def file_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem.split("_")[-1] if ch.isdigit())
    return int(digits) if digits else 0


def group_key(path: Path, position: int, block_size: int) -> str:
    # Filename series prefix + adjacent sorted blocks are a temporal-proximity proxy.
    prefix = path.stem.rsplit("_", 1)[0] if "_" in path.stem else "series"
    return f"{prefix}:{position // block_size}"


def assign_groups(groups: list[str], seed: int) -> dict[str, str]:
    """Deterministic approximately 80/10/10 group split with non-empty partitions."""
    if not groups:
        return {}
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    count = len(ordered)
    if count < 3:
        return {group: ("train" if i == 0 else "val") for i, group in enumerate(ordered)}
    n_train = max(1, int(round(0.80 * count)))
    n_val = max(1, int(round(0.10 * count)))
    if n_train + n_val >= count:
        n_train, n_val = count - 2, 1
    return {
        group: (
            "train" if i < n_train
            else "val" if i < n_train + n_val
            else "test"
        )
        for i, group in enumerate(ordered)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert RDD2022 Pascal VOC XML labels to YOLO format with grouped splits."
    )
    parser.add_argument("--images", required=True, type=Path, help="Directory containing labelled training images")
    parser.add_argument("--annotations", required=True, type=Path, help="Directory containing Pascal VOC XML files")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset directory")
    parser.add_argument("--block-size", type=int, default=25, help="Adjacent sorted frames per split group")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.images.is_dir() or not args.annotations.is_dir():
        parser.error("--images and --annotations must both be existing directories")
    if args.block_size < 1:
        parser.error("--block-size must be positive")

    images = sorted(
        [p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: (p.stem.rsplit("_", 1)[0], file_number(p), p.name),
    )
    if not images:
        parser.error("No image files found")

    paired = [(image, args.annotations / f"{image.stem}.xml") for image in images]
    missing = [str(image) for image, xml in paired if not xml.exists()]
    if missing:
        parser.error(
            f"Missing XML annotations for {len(missing)} images. First missing image: {missing[0]}"
        )

    group_for_image = {
        image: group_key(image, i, args.block_size)
        for i, (image, _) in enumerate(paired)
    }
    split_map = assign_groups(sorted(set(group_for_image.values())), args.seed)
    split_images: Counter[str] = Counter()
    total_classes: Counter[str] = Counter()
    manifest = []

    for image_path, xml_path in paired:
        split = split_map[group_for_image[image_path]]
        label_lines, counts = voc_to_yolo(xml_path, image_path)
        total_classes.update(counts)
        split_images[split] += 1

        image_out = args.output / "images" / split / image_path.name
        label_out = args.output / "labels" / split / f"{image_path.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_out)
        label_out.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        manifest.append({
            "image": image_path.name,
            "annotation": str(xml_path),
            "split": split,
            "group": group_for_image[image_path],
            "objects": len(label_lines),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in CLASS_NAMES.items()},
    }
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False),
        encoding="utf-8",
    )
    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "annotation", "split", "group", "objects"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("Image split counts:", dict(split_images))
    print("Object counts:", dict(total_classes))
    print("Wrote:", args.output / "data.yaml")
    print("Review split_manifest.csv for adjacent-frame grouping assumptions before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
