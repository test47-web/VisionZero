# Experiment protocol

## Research questions
1. Can a compact object detector localise and classify the four RDD2022 categories on held-out Indian road images?
2. How does a classical OpenCV edge/contour pipeline behave compared with a trained detector?
3. Does a consistent CLAHE-enhanced input experiment alter per-class detector performance?
4. Which classes and visual conditions generate the most false positives and false negatives?

## Dataset protocol
Use labelled India training images. Convert Pascal VOC XML boxes to YOLO labels. Keep the official image-only challenge test data untouched. Create an internal grouped split from labelled images by sorting filenames and keeping adjacent blocks together. Retain the manifest. This is a temporal-proximity proxy, not a substitute for verified drive/session metadata. Count per-class boxes and empty images after conversion.

## Model protocol
Primary baseline: Ultralytics YOLO26n, pretrained initialisation, 640 image size, 60 epochs, batch 8, workers 4, AMP enabled and early-stopping patience 20. Record actual library/model versions. If batch 8 exhausts GPU memory, reduce to 4; if needed, reduce image size to 512.

## Metrics
Report aggregate and per-class precision, recall, F1 at a stated IoU/confidence setting, mAP@50, mAP@50-95, confusion matrix, PR curves, latency and checkpoint size. State hardware and timing method. Do not pre-fill results before a completed run.

## Ablations
- OpenCV baseline: CLAHE, denoising, Canny and morphology with contour proposals.
- YOLO baseline on original RGB images.
- Optional CLAHE-input experiment, applied consistently to train, validation and internal test.
- Optional 512 versus 640 image-size experiment with the same split.

## Reporting
The initial repository contains no trained-model benchmark results. Populate result tables only from actual logs. Include failure cases and discuss class imbalance, small cracks, visibility, camera angle and domain shift.
