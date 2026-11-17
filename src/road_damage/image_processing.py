from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Candidate:
    x1: int
    y1: int
    x2: int
    y2: int
    heuristic_score: float
    area: int


def preprocess_road_image(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Return intermediate OpenCV views for an interpretable classical baseline."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(denoised, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return {
        "gray": gray,
        "clahe": enhanced,
        "denoised": denoised,
        "edges": edges,
        "closed_edges": closed,
    }


def find_damage_candidates(
    image_bgr: np.ndarray,
    min_area: int = 120,
    max_area_fraction: float = 0.18,
    max_candidates: int = 30,
) -> tuple[np.ndarray, list[Candidate], dict[str, np.ndarray]]:
    """Highlight edge-rich regions; scores are heuristics, not model probabilities."""
    views = preprocess_road_image(image_bgr)
    h, w = image_bgr.shape[:2]
    max_area = int(h * w * max_area_fraction)
    contours, _ = cv2.findContours(
        views["closed_edges"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[Candidate] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 8 or bh < 4:
            continue
        roi = views["edges"][y : y + bh, x : x + bw]
        edge_density = float(np.count_nonzero(roi)) / max(1, roi.size)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        # A transparent visual score only; it is deliberately not named confidence.
        score = float(np.clip(0.65 * edge_density + 0.35 * (1.0 - min(circularity, 1.0)), 0, 1))
        candidates.append(Candidate(x, y, x + bw, y + bh, score, area))
    candidates.sort(key=lambda item: (item.heuristic_score, item.area), reverse=True)
    candidates = candidates[:max_candidates]
    annotated = image_bgr.copy()
    for idx, item in enumerate(candidates, start=1):
        cv2.rectangle(annotated, (item.x1, item.y1), (item.x2, item.y2), (0, 190, 255), 2)
        label = f"Candidate {idx} | heuristic {item.heuristic_score:.2f}"
        cv2.putText(
            annotated,
            label,
            (item.x1, max(18, item.y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 190, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated, candidates, views
