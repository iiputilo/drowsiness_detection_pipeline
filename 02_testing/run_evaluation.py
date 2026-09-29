import csv
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def get_ground_truth(video_path: Path, config: dict) -> str | None:
    stem = video_path.stem.lower()
    eval_cfg = config["evaluation"]
    skip_label = str(eval_cfg["skip_label"])
    alert_label = str(eval_cfg["alert_label"])
    drowsy_label = str(eval_cfg["drowsy_label"])

    # UTA-RLDD structure: uta_rldd/{subject_id}/{label}.mp4
    # stem is "0", "5", "10", "10_1", "10_2", etc.
    level = stem.split("_")[0]
    if level == skip_label:
        return "skip"
    if level == drowsy_label:
        return "drowsy"
    if level == alert_label:
        return "alert"

    # fallback: drowsinessN pattern in filename
    if f"drowsiness{skip_label}" in stem:
        return "skip"
    if f"drowsiness{drowsy_label}" in stem:
        return "drowsy"
    if f"drowsiness{alert_label}" in stem:
        return "alert"

    return None


def draw_yolo_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    annotated = frame.copy()
    if result.get("face_box"):
        x1, y1, x2, y2 = result["face_box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        state = result.get("state", "")
        color = (0, 0, 255) if state == "drowsy" else (0, 255, 0)
        cv2.putText(annotated, state.upper(), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def draw_mediapipe_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    annotated = frame.copy()
    lms = result.get("landmarks")
    if lms:
        for pt in lms:
            cv2.circle(annotated, tuple(pt), 1, (0, 255, 0), -1)
    ear = result.get("ear")
    mar = result.get("mar")
    state = result.get("state", "")
    color = (0, 0, 255) if state == "drowsy" else (0, 255, 0)
    y = 20
    if ear is not None:
        cv2.putText(annotated, f"EAR: {ear:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    if mar is not None:
        cv2.putText(annotated, f"MAR: {mar:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    cv2.putText(annotated, f"STATE: {state.upper()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


def save_sample_frames(
    frames: list,
    video_name: str,
    pipeline_name: str,
    num_frames: int,
    results_dir: Path,
) -> None:
    if not frames:
        return
    out_dir = results_dir / "sample_frames" / pipeline_name
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(frames) - 1, min(num_frames, len(frames)), dtype=int)
    for idx in indices:
        frame, frame_idx = frames[idx]
        out_path = out_dir / f"{video_name}_{pipeline_name}_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), frame)


def get_pipeline_window_cfg(config: dict, pipeline_name: str) -> dict:
    base = config["window"]
    overrides = base.get("pipelines", {}).get(pipeline_name, {})
    return {
        "duration": overrides.get("duration", base["duration"]),
        "drowsy_fraction": overrides.get("drowsy_fraction", base["drowsy_fraction"]),
    }


def process_video(
    video_path: Path,
    pipelines: dict,
    config: dict,
    results_dir: Path,
) -> list[dict]:
    sampling_cfg = config["sampling"]
    vis_cfg = config["visualization"]
    progress_cfg = config.get("progress", {})
    progress_enabled = progress_cfg.get("enabled", True)
    per_model_progress = progress_cfg.get("per_model", True)
    model_bar_width = progress_cfg.get("bar_width", None)
    model_bar_leave = progress_cfg.get("leave_model_bar", False)
    update_every = max(1, int(progress_cfg.get("update_every_samples", 10)))

    eval_cfg = config.get("evaluation", {})
    min_drowsy_duration = float(eval_cfg.get("min_drowsy_duration_seconds", 2.0))

    from sliding_window import SlidingWindow

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sampling_cfg["sample_interval_seconds"]))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_samples_est = (total_frames + frame_step - 1) // frame_step if total_frames > 0 else None

    ground_truth = get_ground_truth(video_path, config)
    if ground_truth is None:
        logger.warning(f"Cannot determine ground truth for: {video_path.name}, skipping")
        cap.release()
        return []

    video_records = []
    save_frames = vis_cfg.get("save_sample_frames", True)
    num_sample_frames = vis_cfg.get("num_sample_frames", 10)

    for pipeline_name, pipeline in pipelines.items():
        pw_cfg = get_pipeline_window_cfg(config, pipeline_name)
        sw = SlidingWindow(pw_cfg["duration"], pw_cfg["drowsy_fraction"])
        logger.debug(
            f"{pipeline_name}: window={pw_cfg['duration']}s, "
            f"drowsy_fraction={pw_cfg['drowsy_fraction']}"
        )
        inference_times = []
        window_predictions = []
        timestamps = []
        first_alarm_time = None
        alarm_count = 0
        sample_frames = []

        # Track alarm episodes in video time
        alarm_episode_start = None
        last_alarm_ts = None
        first_alarm_time_raw = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        num_windows = 0
        total_samples = 0

        frame_pbar = None
        if progress_enabled and per_model_progress:
            frame_pbar = tqdm(
                total=total_samples_est,
                desc=f"{video_path.name} | {pipeline_name}",
                leave=model_bar_leave,
                ncols=model_bar_width,
                unit="sample",
                disable=False,
            )

        try:
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps
                try:
                    result = pipeline.predict(frame)
                except Exception as e:
                    logger.warning(f"Prediction failed at frame {frame_idx} in {video_path.name}: {e}")
                    frame_idx += frame_step
                    total_samples += 1
                    if frame_pbar is not None:
                        frame_pbar.update(1)
                    continue

                total_samples += 1
                inference_times.append(result.get("inference_ms", 0.0))
                is_drowsy = result.get("state") == "drowsy"
                alarm = sw.update(is_drowsy, timestamp)

                if alarm:
                    num_windows += 1
                    if first_alarm_time_raw is None:
                        first_alarm_time_raw = timestamp
                    if alarm_episode_start is None:
                        alarm_episode_start = timestamp
                    last_alarm_ts = timestamp
                else:
                    if alarm_episode_start is not None and last_alarm_ts is not None:
                        episode_duration = last_alarm_ts - alarm_episode_start
                        if episode_duration >= min_drowsy_duration:
                            alarm_count += 1
                            if first_alarm_time is None:
                                first_alarm_time = alarm_episode_start
                        alarm_episode_start = None
                        last_alarm_ts = None

                window_predictions.append(1 if alarm else 0)
                timestamps.append(timestamp)

                if save_frames:
                    if pipeline_name == "yolo":
                        annotated = draw_yolo_overlay(frame, result)
                    else:
                        annotated = draw_mediapipe_overlay(frame, result)
                    sample_frames.append((annotated, frame_idx))

                if frame_pbar is not None:
                    if total_samples % update_every == 0:
                        frame_pbar.set_postfix(
                            t=f"{timestamp:.1f}s",
                            alarm=alarm_count,
                            alarm_state=bool(alarm),
                            state=result.get("state", ""),
                        )
                    frame_pbar.update(1)

                frame_idx += frame_step
        finally:
            if frame_pbar is not None:
                frame_pbar.close()

        if alarm_episode_start is not None and last_alarm_ts is not None:
            episode_duration = last_alarm_ts - alarm_episode_start
            if episode_duration >= min_drowsy_duration:
                alarm_count += 1
                if first_alarm_time is None:
                    first_alarm_time = alarm_episode_start

        if save_frames and sample_frames:
            save_sample_frames(
                sample_frames, video_path.stem, pipeline_name,
                num_sample_frames, results_dir
            )

        mean_inf = float(np.mean(inference_times)) if inference_times else 0.0
        alarm_raised = alarm_count > 0
        first_alarm_time_out = first_alarm_time if alarm_raised else ""

        alarm_rate = num_windows / total_samples if total_samples > 0 else 0.0
        video_records.append({
            "video_file": str(video_path),
            "pipeline": pipeline_name,
            "ground_truth": ground_truth,
            "alarm_raised": alarm_raised,
            "first_alarm_time_seconds": first_alarm_time_out,
            "first_alarm_time_seconds_raw": first_alarm_time_raw if first_alarm_time_raw is not None else "",
            "min_drowsy_duration_seconds": min_drowsy_duration,
            "alarm_count": alarm_count,
            "mean_inference_ms": mean_inf,
            "num_windows": num_windows,
            "total_samples": total_samples,
            "alarm_rate": alarm_rate,
        })

    cap.release()
    return video_records


def main() -> None:
    config = load_config()
    paths = config["paths"]
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    progress_cfg = config.get("progress", {})
    progress_enabled = progress_cfg.get("enabled", True)
    per_video_progress = progress_cfg.get("per_video", True)
    video_bar_width = progress_cfg.get("bar_width", None)

    seed = config["reproducibility"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    uta_dir = Path(paths["uta_rldd_dir"])
    if not uta_dir.exists():
        logger.error(f"UTA-RLDD dataset directory not found: {uta_dir}")
        logger.error("Please download the dataset from github.com/altaga/RLDD and update config.yaml")
        sys.exit(1)

    models_dir = Path(paths["models_dir"])
    face_model = models_dir / "face_detector.pt"
    eyes_model = models_dir / "eyes_detector.pt"
    eyes_classifier_model = models_dir / "eyes_classifier.pt"
    mouth_model = models_dir / "mouth_detector.pt"

    for m in [face_model, eyes_model, eyes_classifier_model, mouth_model]:
        if not m.exists():
            logger.error(f"Model not found: {m}. Run 01_training first.")
            sys.exit(1)

    mp_cfg = config["mediapipe"]

    from pipelines.yolo_pipeline import YOLOPipeline
    from pipelines.mediapipe_pipeline import MediaPipePipeline

    logger.info("Initializing pipelines...")
    try:
        yolo_pipeline = YOLOPipeline(str(face_model), str(eyes_model), str(eyes_classifier_model), str(mouth_model))
    except Exception as e:
        logger.error(f"Failed to initialize YOLO pipeline: {e}")
        sys.exit(1)

    try:
        mp_pipeline = MediaPipePipeline(
            ear_threshold=mp_cfg["ear_threshold"],
            mar_threshold=mp_cfg["mar_threshold"],
            min_detection_confidence=mp_cfg["min_detection_confidence"],
            min_tracking_confidence=mp_cfg["min_tracking_confidence"],
            model_path=paths["face_landmarker_model"],
        )
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe pipeline: {e}")
        sys.exit(1)

    pipelines = {"yolo": yolo_pipeline, "mediapipe": mp_pipeline}

    video_files = []
    video_files_map = {}
    for ext in VIDEO_EXTENSIONS:
        for p in uta_dir.rglob(f"*{ext}"):
            video_files_map[str(p).lower()] = p
        for p in uta_dir.rglob(f"*{ext.upper()}"):
            video_files_map.setdefault(str(p).lower(), p)
    video_files = list(video_files_map.values())

    if not video_files:
        logger.error(f"No video files found in: {uta_dir}")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} video files")

    all_records = []
    skipped = 0
    processed = 0
    total_frames_yolo = 0
    total_frames_mp = 0

    video_pbar = tqdm(
        video_files,
        desc="Videos",
        ncols=video_bar_width,
        disable=not (progress_enabled and per_video_progress),
        unit="video",
    )
    for video_path in video_pbar:
        gt = get_ground_truth(video_path, config)
        if progress_enabled and per_video_progress:
            video_pbar.set_postfix(video=video_path.name, label=gt or "unknown")
        if gt == "skip":
            logger.info(f"Skipping video with drowsiness5 label: {video_path.name}")
            skipped += 1
            continue
        if gt is None:
            logger.warning(f"Unknown label pattern, skipping: {video_path.name}")
            skipped += 1
            continue

        records = process_video(video_path, pipelines, config, results_dir)
        all_records.extend(records)
        processed += 1

    csv_path = results_dir / "raw_results.csv"
    fieldnames = [
        "video_file", "pipeline", "ground_truth", "alarm_raised",
        "first_alarm_time_seconds", "first_alarm_time_seconds_raw",
        "min_drowsy_duration_seconds", "alarm_count", "mean_inference_ms",
        "num_windows", "total_samples", "alarm_rate",
    ]
    try:
        with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(f"Saved raw results: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save results CSV: {e}")

    logger.info(f"Total videos processed: {processed}")
    logger.info(f"Total videos skipped: {skipped}")
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
