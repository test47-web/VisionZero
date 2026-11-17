# AI-Based Road Damage and Pothole Detection Using Image Processing and Deep Learning

A reproducible university project prototype for road-surface analysis. It combines an interpretable OpenCV baseline with a YOLO26n training/inference pipeline for four RDD2022 categories.

> **Model status:** this repository contains source code and training/evaluation workflows, not a trained checkpoint or the RDD2022 dataset. OpenCV candidate analysis runs immediately. Four-class YOLO detections require the dataset to be prepared and a model to be trained. No benchmark metrics are claimed before a real evaluation run.

## Damage classes

| ID | Category | Meaning |
|---|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

The official RDD2022 annotations use Pascal VOC XML. `scripts/prepare_rdd2022.py` converts the four selected classes to YOLO labels.

## Features

- OpenCV pipeline: grayscale conversion, CLAHE, Gaussian denoising, Canny edges, morphological closing and contour candidates.
- YOLO26n training and inference scripts.
- Pascal VOC XML to YOLO conversion.
- Grouped split manifest to reduce near-neighbour leakage.
- Streamlit interface for single-image analysis and CSV export.
- Image, folder and video prediction script.
- Evaluation protocol with no fabricated results.

## Setup (Windows / PowerShell)

Python 3.10 or 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the official selector for your Windows/Python/CUDA environment: https://pytorch.org/get-started/locally/

Then install the project packages:

```powershell
pip install -r requirements.txt
```

Check CUDA availability:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run the app

```powershell
streamlit run app.py
```

Select **OpenCV candidate analysis** to run without a trained model. The highlighted regions are heuristic edge/contour candidates, not semantic classifications or calibrated confidence values.

Select **YOLO detector** after a trained checkpoint is available at `weights/best.pt` or via the `ROAD_DAMAGE_MODEL` environment variable.

## Prepare RDD2022 (India)

1. Obtain the dataset from the official CRDDC links: https://crddc2022.sekilab.global/data/
2. Download and extract the India subset locally. Dataset files are not redistributed in this repository.
3. Locate the labelled training images and matching Pascal VOC XML annotation folder.
4. Run the converter (adjust the paths to match your extracted archive):

```powershell
python scripts/prepare_rdd2022.py --images data/raw/RDD2022_India/train/images --annotations data/raw/RDD2022_India/train/annotations --output data/processed/india --block-size 25 --seed 42
```

The script creates `data.yaml`, YOLO labels and `split_manifest.csv`. It groups adjacent sorted filenames in blocks of 25 as a temporal-proximity proxy. This reduces likely leakage from neighbouring frames but is not equivalent to verified drive/session metadata; inspect the manifest and document the limitation. Keep the official image-only challenge test set untouched.

## Train and evaluate

```powershell
python train.py --data data/processed/india/data.yaml --model yolo26n.pt --epochs 60 --imgsz 640 --batch 8 --workers 4
python evaluate.py --weights runs/road_damage/train/weights/best.pt --data data/processed/india/data.yaml --split test
```

If an 8 GB laptop GPU runs out of memory, reduce `--batch` to 4, then consider `--imgsz 512`. Training time and CUDA compatibility depend on the installed driver, framework build, thermals and power settings. Follow the official PyTorch selector instead of copying a random CUDA command.

## Predict

```powershell
python predict.py --weights runs/road_damage/train/weights/best.pt --source path\to\road.jpg --output runs\predictions
python predict.py --weights runs/road_damage/train/weights/best.pt --source path\to\clip.mp4 --output runs\video
```

Outputs include annotated media and `detections.csv`. Record actual aggregate and per-class precision, recall, F1, mAP@50, mAP@50-95 and inference latency from completed evaluation runs.

## Recommended experiments

1. OpenCV candidate baseline (qualitative; not a four-class detector).
2. YOLO26n on original RGB images.
3. Optional CLAHE-input ablation (apply consistently to train, validation and internal test).
4. Optional image-size comparison (512 vs 640).
5. Error analysis by class, lighting, texture and viewpoint.

## Limitations

- The application is not a Bengaluru-specific road-condition survey.
- The dataset does not provide physical depth, damage area in real-world units or engineering severity.
- GPS is not inferred from pixels; geotagging requires trusted image metadata or an external capture device.
- OpenCV candidates can be triggered by shadows, markings and surface texture.
- The system is decision support for human review, not an official road-safety inspection.

## Reproducibility and responsible use

Record dataset version/source/date/licence, class counts, split manifest, seed, model/package versions, driver/GPU, training configuration and logs. Do not commit raw dataset files, run caches or large weights. Review the current RDD2022 dataset terms before redistributing any material. Review the Ultralytics licence before publishing model weights or distributing a derivative application.

## Official references

- RDD2022 source repository: https://github.com/sekilab/RoadDamageDetector
- CRDDC 2022 data page: https://crddc2022.sekilab.global/data/
- RDD2022 data record: https://doi.org/10.6084/m9.figshare.21431547.v1
- Ultralytics documentation: https://docs.ultralytics.com/
- Ultralytics licence: https://www.ultralytics.com/license
- OpenCV image processing: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
- Streamlit file uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit download button: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- PyTorch installation selector: https://pytorch.org/get-started/locally/

## Repository status

A trained checkpoint and measured experimental results must be added only after the dataset preparation and local training/evaluation steps have been completed.
