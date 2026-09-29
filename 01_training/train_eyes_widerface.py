import logging
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent


def load_config(config_path: Path = ROOT / "config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def validate_dataset(dataset_path: Path) -> bool:
    required = [
        dataset_path / "data.yaml",
        dataset_path / "train" / "images",
        dataset_path / "valid" / "images",
    ]
    for p in required:
        if not p.exists():
            return False
    return True


def copy_results(run_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        "results.png",
        "confusion_matrix.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
        "BoxR_curve.png",
    ]
    for filename in files_to_copy:
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            logger.info(f"Saved result: {output_dir / filename}")
        else:
            logger.warning(f"Result file not found: {src}")


def main() -> None:
    from ultralytics import YOLO

    config = load_config()
    paths = config["paths"]
    training = config["training"]

    dataset_path = ROOT / paths["dataset_eyes_widerface"]
    models_output_dir = ROOT / paths["models_output_dir"]
    results_output_dir = ROOT / paths.get("results_output_dir", "results")
    runs_dir = ROOT / "runs"

    if not validate_dataset(dataset_path):
        logger.error(f"Dataset not found or incomplete at: {dataset_path}")
        logger.error("Run create_eyes_widerface.py first to generate the dataset.")
        sys.exit(1)

    models_output_dir.mkdir(parents=True, exist_ok=True)

    device = training.get("device") or None

    logger.info("Loading YOLO26n pretrained weights for eye locator training")
    try:
        model = YOLO(str(ROOT / "yolo26n.pt"))
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    data_yaml = str(dataset_path / "data.yaml")
    logger.info(f"Starting eye locator training with dataset: {data_yaml}")

    try:
        results = model.train(
            data=data_yaml,
            epochs=training["epochs_eyes"],
            imgsz=training["imgsz"],
            batch=training["batch"],
            patience=training["patience"],
            seed=training["seed"],
            device=device,
            project=str(runs_dir / "detect"),
            name="eyes_widerface_detector",
            mosaic=0.0,
            mixup=0.0,
            workers=4,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    try:
        metrics = results.results_dict
        logger.info("Training complete. Final metrics:")
        for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.4f}")
    except Exception as e:
        logger.warning(f"Could not extract detailed metrics: {e}")

    run_dir = runs_dir / "detect" / "eyes_widerface_detector"
    weights_path = run_dir / "weights" / "best.pt"
    if weights_path.exists():
        logger.info(f"Best weights → {weights_path}")
    else:
        logger.warning("best.pt not found — check run directory")

    copy_results(run_dir, results_output_dir / "eyes_widerface_detector")
    logger.info("Eye locator training finished")


if __name__ == "__main__":
    main()
