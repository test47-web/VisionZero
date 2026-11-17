from __future__ import annotations

import io
from typing import Any

import pandas as pd


def detections_to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    columns = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    frame = pd.DataFrame(rows, columns=columns)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def summarize_detections(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["class_name", "count", "mean_confidence"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("class_name", as_index=False)
        .agg(count=("class_name", "size"), mean_confidence=("confidence", "mean"))
        .sort_values("count", ascending=False)
    )
