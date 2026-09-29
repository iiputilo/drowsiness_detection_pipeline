import logging
import shutil
import sys
from pathlib import Path

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
        "PR_curve.png",
        "F1_curve.png",
    ]
    for filename in files_to_copy:
        src = run_dir / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(str(src), str(dst))
            logger.info(f"Saved training result: {dst}")
        else:
            logger.warning(f"Result file not found: {src}")


def main() -> None:
    from ultralytics import YOLO

    config = load_config()
    paths = config["paths"]
    training = config["training"]

    dataset_path = Path(paths["dataset_mouth"])
    models_output_dir = Path(paths["models_output_dir"])

    if not validate_dataset(dataset_path):
        logger.error(f"Dataset not found or incomplete at: {dataset_path}")
        logger.error(
            "Please place the mouth_dataset (open/closed mouth, YOLO format) "
            "at the path specified in config.yaml under paths.dataset_mouth"
        )
        sys.exit(1)

    models_output_dir.mkdir(parents=True, exist_ok=True)

    device = training.get("device") or None

    logger.info("Loading YOLO26n pretrained weights for mouth state detector training")
    try:
        model = YOLO("yolo26n.pt")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    data_yaml = str(dataset_path / "data.yaml")
    logger.info(f"Starting mouth state detector training with dataset: {data_yaml}")

    try:
        results = model.train(
            data=data_yaml,
            epochs=training["epochs_mouth"],
            imgsz=training["imgsz"],
            batch=training["batch"],
            patience=training["patience"],
            seed=training["seed"],
            device=device,
            project="runs",
            name="mouth_detector",
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
        if hasattr(results, "names"):
            logger.info(f"  Classes: {results.names}")
    except Exception as e:
        logger.warning(f"Could not extract detailed metrics: {e}")

    run_dir = Path("runs") / "detect" / "mouth_detector"
    results_dir = Path(paths.get("results_output_dir", "results")) / "mouth_detector"
    copy_results(run_dir, results_dir)

    logger.info("Mouth state detector training finished")


if __name__ == "__main__":
    main()
