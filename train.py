from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact YOLO road-damage detector.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="0 for first CUDA GPU, cpu for CPU")
    parser.add_argument("--project", default="runs/road_damage")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Dataset YAML not found: {args.data}")
    if args.epochs < 1 or args.imgsz < 320 or args.batch < 1:
        parser.error("epochs, batch and image size must be positive (imgsz >= 320)")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=True,
        cache=False,
        project=args.project,
        name=args.name,
        patience=20,
        plots=True,
        save=True,
        exist_ok=True,
    )
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    print(f"Training complete. Inspect {save_dir / 'weights' / 'best.pt'} and all generated plots/logs.")


if __name__ == "__main__":
    main()
