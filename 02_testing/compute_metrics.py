import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def compute_pipeline_metrics(df: pd.DataFrame, pipeline: str) -> dict:
    sub = df[df["pipeline"] == pipeline].copy()

    if "alarm_count" in sub.columns:
        sub["alarm_count"] = pd.to_numeric(sub["alarm_count"], errors="coerce").fillna(0)
        pred_alarm = sub["alarm_count"] > 0
    else:
        pred_alarm = sub["alarm_raised"] == True

    tp = int(((sub["ground_truth"] == "drowsy") & (pred_alarm == True)).sum())
    fp = int(((sub["ground_truth"] == "alert") & (pred_alarm == True)).sum())
    tn = int(((sub["ground_truth"] == "alert") & (pred_alarm == False)).sum())
    fn = int(((sub["ground_truth"] == "drowsy") & (pred_alarm == False)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    drowsy_rows = sub[(sub["ground_truth"] == "drowsy") & (pred_alarm == True)]
    alarm_times = pd.to_numeric(drowsy_rows["first_alarm_time_seconds"], errors="coerce").dropna()
    alarm_time_mean = float(alarm_times.mean()) if len(alarm_times) > 0 else float("nan")
    alarm_time_std = float(alarm_times.std()) if len(alarm_times) > 1 else 0.0

    inf_times = pd.to_numeric(sub["mean_inference_ms"], errors="coerce").dropna()
    inf_mean = float(inf_times.mean()) if len(inf_times) > 0 else 0.0
    inf_std = float(inf_times.std()) if len(inf_times) > 1 else 0.0

    drowsy_rates = pd.to_numeric(
        sub[sub["ground_truth"] == "drowsy"]["alarm_rate"], errors="coerce"
    ).dropna()
    alert_rates = pd.to_numeric(
        sub[sub["ground_truth"] == "alert"]["alarm_rate"], errors="coerce"
    ).dropna()
    mean_drowsy_alarm_rate = float(drowsy_rates.mean()) if len(drowsy_rates) > 0 else float("nan")
    mean_alert_alarm_rate = float(alert_rates.mean()) if len(alert_rates) > 0 else float("nan")
    drowsiness_detection_ratio = (
        mean_drowsy_alarm_rate - mean_alert_alarm_rate
        if not (np.isnan(mean_drowsy_alarm_rate) or np.isnan(mean_alert_alarm_rate))
        else float("nan")
    )

    return {
        "pipeline": pipeline,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "alarm_time_mean_s": alarm_time_mean,
        "alarm_time_std_s": alarm_time_std,
        "inference_mean_ms": inf_mean,
        "inference_std_ms": inf_std,
        "mean_drowsy_alarm_rate": mean_drowsy_alarm_rate,
        "mean_alert_alarm_rate": mean_alert_alarm_rate,
        "drowsiness_detection_ratio": drowsiness_detection_ratio,
    }


def build_per_video_table(df: pd.DataFrame) -> pd.DataFrame:
    yolo = df[df["pipeline"] == "yolo"].copy()
    mp = df[df["pipeline"] == "mediapipe"].copy()

    def _episode_prediction(sub: pd.DataFrame) -> pd.Series:
        if "alarm_count" not in sub.columns:
            return sub["alarm_raised"]
        counts = pd.to_numeric(sub["alarm_count"], errors="coerce").fillna(0)
        return counts > 0

    yolo_pred = _episode_prediction(yolo)
    mp_pred = _episode_prediction(mp)

    merge_cols = ["video_file", "ground_truth"]
    yolo = yolo.assign(alarm_raised=yolo_pred).rename(columns={
        "alarm_raised": "yolo_prediction",
        "first_alarm_time_seconds": "yolo_alarm_time",
        "mean_inference_ms": "yolo_inference_ms",
    })
    mp = mp.assign(alarm_raised=mp_pred).rename(columns={
        "alarm_raised": "mediapipe_prediction",
        "first_alarm_time_seconds": "mediapipe_alarm_time",
        "mean_inference_ms": "mediapipe_inference_ms",
    })

    keep_yolo = merge_cols + ["yolo_prediction", "yolo_alarm_time", "yolo_inference_ms"]
    keep_mp = merge_cols + ["mediapipe_prediction", "mediapipe_alarm_time", "mediapipe_inference_ms"]

    merged = pd.merge(
        yolo[keep_yolo], mp[keep_mp],
        on=merge_cols, how="outer"
    )
    return merged


def format_summary_table(metrics_list: list[dict], window_size: int, window_threshold: int) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("PIPELINE EVALUATION METRICS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("EVALUATION METHODOLOGY NOTE:")
    lines.append("  Ground truth labels are assigned at the video level (weak supervision).")
    lines.append("  Each video is labelled either 'alert' (drowsiness0) or 'drowsy' (drowsiness10).")
    lines.append("  Videos with label drowsiness5 (intermediate state) were excluded from evaluation.")
    lines.append(f"  Sliding window size: {window_size}, alarm threshold: {window_threshold}.")
    lines.append("  Video-level labelling is a known limitation of this evaluation protocol.")
    lines.append("")
    lines.append("-" * 80)

    header = f"{'Metric':<28} {'YOLO':>12} {'MediaPipe':>12}"
    lines.append(header)
    lines.append("-" * 80)

    m0 = {m["pipeline"]: m for m in metrics_list}
    yolo = m0.get("yolo", {})
    mp = m0.get("mediapipe", {})

    rows = [
        ("True Positives", "tp"),
        ("False Positives", "fp"),
        ("True Negatives", "tn"),
        ("False Negatives", "fn"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("Specificity", "specificity"),
        ("False Alarm Rate", "false_alarm_rate"),
        ("Alarm Time Mean (s)", "alarm_time_mean_s"),
        ("Alarm Time Std (s)", "alarm_time_std_s"),
        ("Inference Mean (ms)", "inference_mean_ms"),
        ("Inference Std (ms)", "inference_std_ms"),
        ("--- Continuous metrics ---", None),
        ("Drowsy Alarm Rate (↑)", "mean_drowsy_alarm_rate"),
        ("Alert Alarm Rate  (↓)", "mean_alert_alarm_rate"),
        ("Detection Ratio   (↑)", "drowsiness_detection_ratio"),
    ]

    for label, key in rows:
        if key is None:
            lines.append(f"{label}")
            continue
        yv = yolo.get(key, "N/A")
        mv = mp.get(key, "N/A")
        yv_str = f"{yv:.4f}" if isinstance(yv, float) else str(yv)
        mv_str = f"{mv:.4f}" if isinstance(mv, float) else str(mv)
        lines.append(f"{label:<28} {yv_str:>12} {mv_str:>12}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    paths = config["paths"]
    results_dir = Path(paths["results_dir"])
    window_cfg = config["window"]
    window_size = window_cfg.get("size", window_cfg.get("duration", "N/A"))
    window_threshold = window_cfg.get("threshold", window_cfg.get("drowsy_fraction", "N/A"))

    raw_csv = results_dir / "raw_results.csv"
    if not raw_csv.exists():
        logger.error(f"Raw results not found: {raw_csv}. Run run_evaluation.py first.")
        sys.exit(1)

    try:
        df = pd.read_csv(str(raw_csv))
        df["alarm_raised"] = df["alarm_raised"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    except Exception as e:
        logger.error(f"Failed to read raw results: {e}")
        sys.exit(1)

    metrics_list = []
    for pipeline_name in ["yolo", "mediapipe"]:
        if pipeline_name in df["pipeline"].values:
            m = compute_pipeline_metrics(df, pipeline_name)
            metrics_list.append(m)
        else:
            logger.warning(f"No results found for pipeline: {pipeline_name}")

    metrics_csv = results_dir / "metrics.csv"
    try:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(str(metrics_csv), index=False)
        logger.info(f"Saved metrics: {metrics_csv}")
    except Exception as e:
        logger.error(f"Failed to save metrics CSV: {e}")

    per_video_df = build_per_video_table(df)
    per_video_csv = results_dir / "per_video_results.csv"
    try:
        per_video_df.to_csv(str(per_video_csv), index=False)
        logger.info(f"Saved per-video results: {per_video_csv}")
    except Exception as e:
        logger.warning(f"Could not save per-video results: {e}")

    summary_text = format_summary_table(
        metrics_list,
        window_size,
        window_threshold,
    )

    print("\n" + summary_text)

    summary_txt = results_dir / "metrics_summary.txt"
    try:
        summary_txt.write_text(summary_text + "\n", encoding="utf-8")
        logger.info(f"Saved metrics summary: {summary_txt}")
    except Exception as e:
        logger.warning(f"Could not save metrics summary: {e}")


if __name__ == "__main__":
    main()
