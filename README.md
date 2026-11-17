# AI-Based Road Damage and Pothole Detection Using Image Processing and Deep Learning

A student research prototype for analysing road-surface images using OpenCV and a YOLO26n object-detection workflow.

> **Implementation status:** the OpenCV candidate-analysis mode is runnable after dependency installation. The repository includes the RDD2022 XML-to-YOLO converter and YOLO training, evaluation and inference scripts. It does **not** include the RDD2022 dataset or trained custom weights. Four-class model detections become available after you prepare the data and train a checkpoint. No benchmark metrics are claimed before evaluation.

## Selected RDD2022 classes

| Class ID | Dataset code | Category |
|---:|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

The CRDDC 2022 India subset lists approximately 9,000 training images and 1,000 test images. Training annotations are in Pascal VOC XML format; the official challenge test set is image-only. See the official source links below.

## Features

- OpenCV baseline: grayscale, CLAHE, Gaussian denoising, Canny edges, morphological closing and contour-based candidate regions.
- Pascal VOC XML to YOLO text-label conversion.
- Grouped split manifest to reduce the risk of near-neighbour leakage.
- YOLO26n fine-tuning, validation and inference scripts.
- Streamlit single-image interface with annotated output and CSV export.
- Image, folder and video prediction utility.

## Environment setup (Windows / PowerShell)

Python 3.10 or 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the official selector for your OS, Python and CUDA setup: https://pytorch.org/get-started/locally/

Then install project dependencies:

```powershell
pip install -r requirements.txt
```

Verify CUDA availability:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run the application

```powershell
streamlit run app.py
```

Choose **OpenCV candidate analysis** to use the image-processing path without a trained checkpoint. The candidate boxes and heuristic scores are not four-class predictions or calibrated probabilities. The YOLO option requires a custom-trained checkpoint at `weights/best.pt` or the path stored in `ROAD_DAMAGE_MODEL`.

## Prepare the RDD2022 India subset

1. Download the India subset from the official CRDDC data page.
2. Extract the labelled training images and Pascal VOC XML annotations locally.
3. Use the actual extracted directory paths in the command below:

```powershell
python scripts/prepare_rdd2022.py --images data/raw/RDD2022_India/train/images --annotations data/raw/RDD2022_India/train/annotations --output data/processed/india --block-size 25 --seed 42
```

The converter writes YOLO labels, `data.yaml` and `split_manifest.csv`. Adjacent filename blocks are kept in one partition as a practical temporal-proximity proxy. This does not prove capture-session separation; inspect the manifest and use verified drive/session metadata when available. Keep the official image-only challenge test set untouched.

## Train and evaluate

```powershell
python train.py --data data/processed/india/data.yaml --model yolo26n.pt --epochs 60 --imgsz 640 --batch 8 --workers 4
python evaluate.py --weights runs/road_damage/train/weights/best.pt --data data/processed/india/data.yaml --split test
```

For an 8 GB laptop GPU, batch 8 is an initial setting, not a guarantee. Reduce to batch 4 or image size 512 if out of memory. Use the official PyTorch selector to configure CUDA; driver and framework compatibility determine whether the GPU is available.

## Predict

```powershell
python predict.py --weights runs/road_damage/train/weights/best.pt --source path\to\road.jpg --output runs\predictions
python predict.py --weights runs/road_damage/train/weights/best.pt --source path\to\clip.mp4 --output runs\video
```

The prediction script writes annotated media and `detections.csv`. Report actual per-class and aggregate precision, recall, F1, mAP@50, mAP@50-95 and inference latency from completed runs only.

## Interpretation and limitations

- OpenCV contours are heuristic candidates, not trained D00/D10/D20/D40 classifications.
- A no-detection result does not prove that a road is damage-free.
- RDD2022 is not a Bengaluru-specific road inventory; local roads, monsoon conditions, camera viewpoint and surface textures may differ.
- Bounding boxes do not provide physical depth, real-world damage area, structural severity, repair urgency or GPS coordinates.
- This is a decision-support prototype for human review, not a certified road-safety inspection.

## Reproducibility

Record the dataset version/source/download date/licence, class counts, generated split manifest, random seed, package versions, GPU/driver, training arguments and evaluation logs. Do not commit the dataset archives, processed images, model weights or training caches. Review the dataset and Ultralytics licence terms before redistribution or deployment.

## Official references

- RDD2022 / CRDDC dataset repository: https://github.com/sekilab/RoadDamageDetector
- CRDDC 2022 data page: https://crddc2022.sekilab.global/data/
- RDD2022 Figshare record: https://doi.org/10.6084/m9.figshare.21431547.v1
- Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/
- Ultralytics licence: https://www.ultralytics.com/license
- OpenCV image-processing documentation: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
- Streamlit file uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit download button: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- PyTorch installation selector: https://pytorch.org/get-started/locally/

## Licence

This university prototype is released under GNU AGPL-3.0, subject to the Ultralytics licensing requirements for the training and model components. Read the full licence text at https://www.gnu.org/licenses/agpl-3.0.txt and the current Ultralytics licensing guidance at https://www.ultralytics.com/license. RDD2022 is separate third-party dataset material and remains subject to its own licence and attribution terms.
