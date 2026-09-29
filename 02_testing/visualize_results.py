import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def get_style() -> str:
    available = plt.style.available
    for candidate in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "seaborn"]:
        if candidate in available:
            return candidate
    return "default"


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


def plot_metrics_comparison(metrics_df: pd.DataFrame, plots_dir: Path, fig_w: float, fig_h: float, dpi: int) -> None:
    metric_cols = ["precision", "recall", "f1", "specificity"]
    labels = ["Precision", "Recall", "F1 Score", "Specificity"]
    pipelines = metrics_df["pipeline"].tolist()
    colors = ["#2196F3", "#FF5722"]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.style.use(get_style())

    for i, (pipeline, color) in enumerate(zip(pipelines, colors)):
        row = metrics_df[metrics_df["pipeline"] == pipeline].iloc[0]
        values = [row.get(col, 0) for col in metric_cols]
        bars = ax.bar(x + i * width - width / 2, values, width, label=pipeline.upper(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline Performance Comparison")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, plots_dir / "metrics_comparison.png", dpi)


def plot_confusion_matrix(metrics_df: pd.DataFrame, pipeline: str, plots_dir: Path, fig_w: float, fig_h: float, dpi: int) -> None:
    row = metrics_df[metrics_df["pipeline"] == pipeline]
    if row.empty:
        logger.warning(f"No metrics for pipeline {pipeline}, skipping confusion matrix")
        return

    row = row.iloc[0]
    tp = int(row.get("tp", 0))
    fp = int(row.get("fp", 0))
    tn = int(row.get("tn", 0))
    fn = int(row.get("fn", 0))

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(fig_w * 0.6, fig_h * 0.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Alert", "Drowsy"],
        yticklabels=["Alert", "Drowsy"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {pipeline.upper()}")
    fig.tight_layout()
    save_figure(fig, plots_dir / f"confusion_matrix_{pipeline}.png", dpi)


def plot_alarm_time(raw_df: pd.DataFrame, plots_dir: Path, fig_w: float, fig_h: float, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(fig_w * 0.6, fig_h))
    plt.style.use(get_style())

    data_to_plot = []
    for pipeline in ["yolo", "mediapipe"]:
        sub = raw_df[(raw_df["pipeline"] == pipeline) & (raw_df["ground_truth"] == "drowsy") & (raw_df["alarm_raised"] == True)]
        times = pd.to_numeric(sub["first_alarm_time_seconds"], errors="coerce").dropna().tolist()
        data_to_plot.append(times)

    ax.boxplot(data_to_plot, labels=["YOLO", "MediaPipe"], patch_artist=True,
               boxprops=dict(facecolor="#2196F3", alpha=0.6))

    colors = ["#2196F3", "#FF5722"]
    for i, (times, color) in enumerate(zip(data_to_plot, colors)):
        x_jitter = np.random.normal(i + 1, 0.05, len(times))
        ax.scatter(x_jitter, times, color=color, alpha=0.6, s=30, zorder=3)

    ax.set_ylabel("Seconds")
    ax.set_title("Time to First Alarm on Drowsy Videos")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, plots_dir / "alarm_time_comparison.png", dpi)


def plot_timeline_example(raw_df: pd.DataFrame, plots_dir: Path, fig_w: float, fig_h: float, dpi: int) -> None:
    drowsy_videos = raw_df[raw_df["ground_truth"] == "drowsy"]["video_file"].unique()
    if len(drowsy_videos) == 0:
        logger.warning("No drowsy video results found for timeline plot")
        return

    example_video = drowsy_videos[0]
    video_name = Path(example_video).stem

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.style.use(get_style())

    colors = {"yolo": "#2196F3", "mediapipe": "#FF5722"}
    styles = {"yolo": "-", "mediapipe": "--"}

    for pipeline in ["yolo", "mediapipe"]:
        sub = raw_df[(raw_df["video_file"] == example_video) & (raw_df["pipeline"] == pipeline)]
        if sub.empty:
            continue
        alarm_time = pd.to_numeric(sub.iloc[0]["first_alarm_time_seconds"], errors="coerce")
        if not np.isnan(alarm_time):
            ax.axvline(x=alarm_time, color=colors[pipeline], linestyle=styles[pipeline],
                       label=f"{pipeline.upper()} alarm @ {alarm_time:.1f}s", linewidth=2)

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Alarm Triggered")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Alarm", "Alarm"])
    ax.set_title(f"System Alarm Timeline - Example Drowsy Video")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, plots_dir / "timeline_example.png", dpi)


def plot_inference_time(metrics_df: pd.DataFrame, plots_dir: Path, fig_w: float, fig_h: float, dpi: int) -> None:
    pipelines = []
    means = []
    stds = []

    for pipeline in ["yolo", "mediapipe"]:
        row = metrics_df[metrics_df["pipeline"] == pipeline]
        if row.empty:
            continue
        pipelines.append(pipeline.upper())
        means.append(float(row.iloc[0].get("inference_mean_ms", 0)))
        stds.append(float(row.iloc[0].get("inference_std_ms", 0)))

    fig, ax = plt.subplots(figsize=(fig_w * 0.5, fig_h))
    plt.style.use(get_style())
    colors = ["#2196F3", "#FF5722"]
    x = np.arange(len(pipelines))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors[:len(pipelines)], alpha=0.85)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05 + 0.5,
            f"{mean:.1f}ms",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Mean Inference Time per Frame")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, plots_dir / "inference_time_comparison.png", dpi)


def main() -> None:
    config = load_config()
    paths = config["paths"]
    vis_cfg = config["visualization"]
    results_dir = Path(paths["results_dir"])
    plots_dir = results_dir / "plots"

    dpi = vis_cfg.get("figure_dpi", 150)
    fig_w = vis_cfg.get("figure_width", 12)
    fig_h = vis_cfg.get("figure_height", 7)

    raw_csv = results_dir / "raw_results.csv"
    metrics_csv = results_dir / "metrics.csv"

    for p in [raw_csv, metrics_csv]:
        if not p.exists():
            logger.error(f"Required file not found: {p}. Run evaluate and compute_metrics first.")
            sys.exit(1)

    try:
        raw_df = pd.read_csv(str(raw_csv))
        raw_df["alarm_raised"] = raw_df["alarm_raised"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        metrics_df = pd.read_csv(str(metrics_csv))
    except Exception as e:
        logger.error(f"Failed to read results files: {e}")
        sys.exit(1)

    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(metrics_df, plots_dir, fig_w, fig_h, dpi)
    plot_confusion_matrix(metrics_df, "yolo", plots_dir, fig_w, fig_h, dpi)
    plot_confusion_matrix(metrics_df, "mediapipe", plots_dir, fig_w, fig_h, dpi)
    plot_alarm_time(raw_df, plots_dir, fig_w, fig_h, dpi)
    plot_timeline_example(raw_df, plots_dir, fig_w, fig_h, dpi)
    plot_inference_time(metrics_df, plots_dir, fig_w, fig_h, dpi)

    logger.info("All plots saved to: " + str(plots_dir))


if __name__ == "__main__":
    main()
