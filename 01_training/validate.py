import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATES_CLASSES = ["drowsy", "not_drowsy"]


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def run_validation(model_path: Path, data_yaml: Path) -> dict:
    from ultralytics import YOLO

    logger.info(f"Validating model: {model_path}")
    try:
        model = YOLO(str(model_path))
        results = model.val(data=str(data_yaml))
        return results
    except Exception as e:
        logger.error(f"Validation failed for {model_path}: {e}")
        sys.exit(1)


def save_confusion_matrix_plot(cm_data: np.ndarray, class_names: list, output_path: Path) -> None:
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True Label",
            xlabel="Predicted Label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        thresh = cm_data.max() / 2.0
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                ax.text(
                    j, i, f"{cm_data[i, j]:.0f}",
                    ha="center", va="center",
                    color="white" if cm_data[i, j] > thresh else "black",
                )

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion matrix: {output_path}")
    except Exception as e:
        logger.warning(f"Could not save confusion matrix plot: {e}")


def extract_metrics(results, class_names: list) -> list[dict]:
    rows = []
    try:
        box = results.box
        for i, name in enumerate(class_names):
            try:
                row = {
                    "class": name,
                    "precision": float(box.p[i]) if hasattr(box, "p") and i < len(box.p) else 0.0,
                    "recall": float(box.r[i]) if hasattr(box, "r") and i < len(box.r) else 0.0,
                    "map50": float(box.ap50[i]) if hasattr(box, "ap50") and i < len(box.ap50) else 0.0,
                    "map50_95": float(box.ap[i]) if hasattr(box, "ap") and i < len(box.ap) else 0.0,
                }
            except Exception:
                row = {"class": name, "precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}
            rows.append(row)
    except Exception as e:
        logger.warning(f"Could not extract per-class metrics: {e}")
        for name in class_names:
            rows.append({"class": name, "precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0})
    return rows


def save_summary(summary_rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    txt_path = results_dir / "validation_summary.txt"
    header = f"{'Model':<25} {'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
    separator = "-" * len(header)
    lines = [header, separator]
    for row in summary_rows:
        lines.append(
            f"{row['model']:<25} {row['class']:<15} "
            f"{row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['map50']:>10.4f} {row['map50_95']:>10.4f}"
        )
    content = "\n".join(lines) + "\n"
    txt_path.write_text(content)
    logger.info(f"Saved validation summary: {txt_path}")

    csv_path = results_dir / "validation_summary.csv"
    try:
        with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["model", "class", "precision", "recall", "mAP50", "mAP50-95"])
            for row in summary_rows:
                writer.writerow([
                    row["model"], row["class"],
                    f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                    f"{row['map50']:.4f}", f"{row['map50_95']:.4f}",
                ])
        logger.info(f"Saved validation CSV (semicolon-delimited): {csv_path}")
    except Exception as e:
        logger.warning(f"Could not save CSV: {e}")

    print("\n" + header)
    print(separator)
    for row in summary_rows:
        print(
            f"{row['model']:<25} {row['class']:<15} "
            f"{row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['map50']:>10.4f} {row['map50_95']:>10.4f}"
        )


def main() -> None:
    config = load_config()
    paths = config["paths"]
    models_dir = Path(paths["models_output_dir"])
    results_dir = Path("results")

    states_model_path = models_dir / "states_detector.pt"
    face_model_path = models_dir / "face_detector.pt"

    for p in [states_model_path, face_model_path]:
        if not p.exists():
            logger.error(f"Model not found: {p}. Run training first.")
            sys.exit(1)

    dataset_states = Path(paths["dataset_states"])
    dataset_face = Path(paths["dataset_face"])

    states_results = run_validation(states_model_path, dataset_states / "data.yaml")
    face_results = run_validation(face_model_path, dataset_face / "data.yaml")

    try:
        if hasattr(states_results, "confusion_matrix") and states_results.confusion_matrix is not None:
            cm = states_results.confusion_matrix.matrix
            save_confusion_matrix_plot(
                cm, STATES_CLASSES,
                results_dir / "validation_states_confusion_matrix.png"
            )
        else:
            logger.warning("Confusion matrix data not available for states model")
    except Exception as e:
        logger.warning(f"Could not generate states confusion matrix: {e}")

    try:
        if hasattr(face_results, "confusion_matrix") and face_results.confusion_matrix is not None:
            cm = face_results.confusion_matrix.matrix
            save_confusion_matrix_plot(
                cm, ["face"],
                results_dir / "validation_face_confusion_matrix.png"
            )
        else:
            logger.warning("Confusion matrix data not available for face model")
    except Exception as e:
        logger.warning(f"Could not generate face confusion matrix: {e}")

    summary_rows = []

    states_metrics = extract_metrics(states_results, STATES_CLASSES)
    for row in states_metrics:
        row["model"] = "states_detector"
        summary_rows.append(row)

    face_metrics = extract_metrics(face_results, ["face"])
    for row in face_metrics:
        row["model"] = "face_detector"
        summary_rows.append(row)

    save_summary(summary_rows, results_dir)
    logger.info("Validation complete")


if __name__ == "__main__":
    main()
