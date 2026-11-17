from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on a labelled dataset split.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.weights.exists() or not args.data.exists():
        parser.error("Both --weights and --data must exist")
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=args.device, plots=True)
    print("Evaluation results (record the actual output; do not pre-fill metrics):")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
