# 02_testing

Evaluate and compare the YOLO pipeline and MediaPipe pipeline on the UTA-RLDD video dataset.

## Setup

### 1. Install requirements

```bash
make install
```

### 2. Download UTA-RLDD dataset

Download from: https://github.com/altaga/RLDD

Place the dataset folder at the path specified in `config.yaml` under `paths.uta_rldd_dir` (default: `datasets/uta_rldd`).

Video files must follow the naming convention used by UTA-RLDD: filenames containing `drowsiness0` (alert), `drowsiness10` (drowsy), or `drowsiness5` (intermediate, excluded).

### 3. Ensure trained models exist

The trained models are committed in the repository at the path specified in `config.yaml` under `paths.models_dir` (default: `../01_training/models`):
- `face_detector.pt`
- `eyes_detector.pt`
- `eyes_classifier.pt`
- `mouth_detector.pt`

The MediaPipe pipeline additionally needs `face_landmarker.task` in the same directory — it is not redistributed here, see `../01_training/README.md` for the download command.

### 4. Configure parameters

Edit `config.yaml` to adjust evaluation settings:
- `sampling.sample_interval_seconds`: how many seconds between sampled frames (default 2.0)
- `window.duration` and `window.drowsy_fraction`: sliding window parameters
- `evaluation.min_drowsy_duration_seconds`: minimum alarm episode duration in video time
- `mediapipe`: EAR/MAR thresholds

### 5. Run evaluation

```bash
make all
```

This runs the full pipeline:
1. Process all videos with both pipelines and save raw results
2. Compute precision, recall, F1, specificity, and timing metrics
3. Generate all visualizations

Individual steps:

```bash
make evaluate     # run YOLO and MediaPipe on all videos
make metrics      # compute metrics from raw results
make visualize    # generate plots
```

## Output

All outputs are saved to `results/` (configurable via `paths.results_dir`):

**Figures and tables:**
- `results/plots/` — all figures saved at 150 DPI:
  - `metrics_comparison.png` — precision/recall/F1/specificity bar chart
  - `confusion_matrix_yolo.png` and `confusion_matrix_mediapipe.png`
  - `alarm_time_comparison.png` — time to first alarm box plot
  - `timeline_example.png` — alarm timeline for an example video
  - `inference_time_comparison.png` — mean inference time with error bars
- `results/metrics_summary.txt` — formatted metrics table with a methodology note
- `results/sample_frames/` — annotated frame examples from both pipelines (gitignored: derived from the source videos)

**Data files:**
- `results/raw_results.csv` — per-video results for both pipelines (episode-based alarms)
- `results/metrics.csv` — aggregated metrics
- `results/per_video_results.csv` — side-by-side per-video comparison
