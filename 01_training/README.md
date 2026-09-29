# 01_training

Trains the four YOLO models that make up the detection cascade.

| Script | Output model | Task | Dataset (`config.yaml` key) |
|---|---|---|---|
| `train_face.py` | `models/face_detector.pt` | detect a face in the frame | `paths.dataset_face` |
| `train_eyes_widerface.py` | `models/eyes_detector.pt` | locate eye regions inside a face crop | `paths.dataset_eyes_widerface` |
| `train_eyes.py` | `models/eyes_classifier.pt` | classify an eye crop as open / closed | `paths.dataset_eyes` |
| `train_mouth.py` | `models/mouth_detector.pt` | detect `open_mouth` / `closed_mouth` | `paths.dataset_mouth` |

All four models are already committed under `models/`, so you only need to run this stage if you want to retrain.

## Setup

### 1. Install requirements

```bash
make install
```

### 2. Download datasets

**Face detection — WiderFace:**

Download from <https://universe.roboflow.com/mohamedgobara/widerface-yolo> in **YOLOv8** format and extract it to `paths.dataset_face` (default `datasets/widerface`).

**Eye region localization — derived from WiderFace:**

Generated, not downloaded. After WiderFace is in place, run:

```bash
python create_eyes_widerface.py
```

This writes an eye-region detection dataset to `paths.dataset_eyes_widerface` (default `datasets/eyes_widerface`).

**Eye state classification — open/closed eyes:**

A YOLO *classification* dataset with `open` and `closed` class folders, placed at `paths.dataset_eyes` (default `datasets/eyes_yolo`). `prepare_eyes_dataset.py` converts a flat open/closed image collection into the expected layout.

**Mouth state — open/closed mouth:**

A YOLO detection dataset with `open_mouth` / `closed_mouth` classes, placed at `paths.dataset_mouth` (default `datasets/mouth_dataset`).

### 3. Configure parameters

Edit `config.yaml`. Key settings:

- `training.device` — `0` for GPU, `cpu` for CPU, `null` to auto-detect
- `training.epochs_face` / `epochs_eyes` / `epochs_mouth` — per-model epoch counts
- `training.imgsz`, `training.batch`, `training.patience`, `training.seed`
- `augmentation.enabled` — set to `false` to skip offline augmentation

### 4. Run training

```bash
make all
```

This runs augmentation followed by all four trainings. Individual steps:

```bash
make augment              # offline augmentation of the training split
make train-face           # face detector
make train-eyes-regions   # eye region localization
make train-eyes           # eye state classifier
make train-mouth          # mouth state detector
```

## Output

- Trained weights are copied to `models/` (`paths.models_output_dir`)
- Full Ultralytics run directories — loss curves, PR/F1 curves, confusion matrices — land in `runs/detect/`, and per-model copies are written to `results/`
- All plots are saved at 150 DPI

Both `runs/` and `results/` are gitignored; representative training curves are committed under [`../docs/`](../docs) instead.

## Legacy scripts

`train_states.py` and `validate.py` belong to an earlier two-model design (a single `states_detector.pt` covering eye and mouth state) that was replaced by the four-model cascade. They are kept for reference and are not part of `make all`; `validate.py` still expects `states_detector.pt` and will not run against the current model set.

## Third-party model

`face_landmarker.task` (used by the MediaPipe pipeline in stages 2 and 3) is not redistributed here. Download it into `models/`:

```bash
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
