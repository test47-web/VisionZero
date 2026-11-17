from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image, folder or video.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/predictions"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.weights.exists():
        parser.error(&"Weights not found: {args.weights}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    results = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, device=args.device, save=True, project=str(args.output), name="annotated", exist_ok=True, stream=False)
    rows = []
    for result in results:
        filename = Path(result.path).name if result.path else "frame"
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            rows.append({
                "source": filename,
                "class_id": cls_id,
                "class_name": result.names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf.item()),
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
            })
    csv_path = args.output / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} detections to {csv_path}")


if __name__ == "__main__":
    main()
