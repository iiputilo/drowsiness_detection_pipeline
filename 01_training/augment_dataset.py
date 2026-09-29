import logging
import os
import shutil
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def load_yolo_labels(label_path: str) -> tuple[list, list]:
    class_labels = []
    bboxes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
    except Exception as e:
        logger.warning(f"Could not read label file {label_path}: {e}")
    return class_labels, bboxes


def save_yolo_labels(label_path: str, class_labels: list, bboxes: list) -> None:
    try:
        with open(label_path, "w") as f:
            for cls, bbox in zip(class_labels, bboxes):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")
    except Exception as e:
        logger.warning(f"Could not save label file {label_path}: {e}")


def build_transform(aug_cfg: dict) -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg["brightness_limit"],
                contrast_limit=aug_cfg["contrast_limit"],
                p=0.8,
            ),
            A.Blur(blur_limit=aug_cfg["blur_limit"], p=0.5),
            A.HorizontalFlip(p=aug_cfg["horizontal_flip_probability"]),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def augment_dataset(config: dict) -> None:
    aug_cfg = config["augmentation"]
    if not aug_cfg.get("enabled", True):
        logger.info("Augmentation is disabled in config. Skipping.")
        return

    dataset_path = Path(config["paths"]["dataset_states"])
    dataset_face_path = Path(config["paths"].get("dataset_face", ""))

    train_images_dir = dataset_path / "train" / "images"
    train_labels_dir = dataset_path / "train" / "labels"

    if not train_images_dir.exists():
        logger.error(f"Train images directory not found: {train_images_dir}")
        logger.error("Please download the driver-no-yawn dataset and place it at the path specified in config.yaml")
        sys.exit(1)

    if not train_labels_dir.exists():
        logger.error(f"Train labels directory not found: {train_labels_dir}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in train_images_dir.iterdir()
        if f.suffix.lower() in image_extensions and "_aug" not in f.stem
    ]

    if not image_files:
        logger.error(f"No original images found in {train_images_dir}")
        sys.exit(1)

    num_copies = aug_cfg["num_augmented_copies"]
    transform = build_transform(aug_cfg)

    logger.info(f"Found {len(image_files)} original training images")
    logger.info(f"Creating {num_copies} augmented copies per image")

    total_augmented = 0
    for img_path in tqdm(image_files, desc="Augmenting images"):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label_path = train_labels_dir / (img_path.stem + ".txt")
            class_labels, bboxes = load_yolo_labels(str(label_path))

            for i in range(1, num_copies + 1):
                try:
                    transformed = transform(
                        image=image_rgb,
                        bboxes=bboxes,
                        class_labels=class_labels,
                    )
                    aug_image = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
                    aug_bboxes = transformed["bboxes"]
                    aug_labels = transformed["class_labels"]

                    aug_stem = f"{img_path.stem}_aug{i}"
                    aug_image_path = train_images_dir / (aug_stem + img_path.suffix)
                    aug_label_path = train_labels_dir / (aug_stem + ".txt")

                    cv2.imwrite(str(aug_image_path), aug_image)
                    save_yolo_labels(str(aug_label_path), aug_labels, aug_bboxes)
                    total_augmented += 1
                except Exception as e:
                    logger.warning(f"Augmentation copy {i} failed for {img_path.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed processing {img_path.name}: {e}")

    logger.info(f"Original images: {len(image_files)}")
    logger.info(f"Total augmented images created: {total_augmented}")


def main() -> None:
    config = load_config()
    augment_dataset(config)


if __name__ == "__main__":
    main()
