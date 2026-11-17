from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CLASS_NAMES, MODEL_PATH


class ModelUnavailableError(RuntimeError):
    pass


def load_yolo(model_path: str | Path | None = None):
    """Load an Ultralytics checkpoint lazily so OpenCV mode works without it."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise ModelUnavailableError(
            f"YOLO weights not found at {path}. Prepare RDD2022 and train the model first."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelUnavailableError("Install Ultralytics with `pip install -r requirements.txt`.") from exc
    return YOLO(str(path))


def predict_image(image_bgr: np.ndarray, model: Any | None = None, conf: float = 0.25):
    if model is None:
        model = load_yolo()
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()
    rows: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            rows.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": score,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return annotated, rows
