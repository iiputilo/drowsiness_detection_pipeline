import logging
import random
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SRC_ROOT = Path("datasets/open_closed_eyes")
DST_ROOT = Path("datasets/eyes_yolo")

# class index: 0=closed, 1=open
CLASSES = {"closed": 0, "open": 1}

SPLIT_SIZES = {
    "train": 10_000,
    "val": 1_000,
    "test": 1_000,
}

# maps source split folder name -> destination split folder name
SPLIT_MAP = {
    "train": "train",
    "val": "val",
    "test": "test",
}

LABEL_TEMPLATE = "{cls_id} 0.5 0.5 1.0 1.0\n"

DATA_YAML = {
    "path": str(DST_ROOT.resolve()),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 2,
    "names": ["closed", "open"],
}


def sample_images(src_split_dir: Path, cls_name: str, n: int, seed: int) -> list[Path]:
    images = sorted(src_split_dir.glob("*.png")) + sorted(src_split_dir.glob("*.jpg"))
    if len(images) < n:
        logger.warning(
            f"{src_split_dir}/{cls_name}: only {len(images)} images available, using all of them (requested {n})"
        )
        return images
    random.seed(seed)
    return random.sample(images, n)


def convert_split(src_split: str, dst_split: str, seed: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    img_dst = DST_ROOT / dst_split / "images"
    lbl_dst = DST_ROOT / dst_split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for cls_name, cls_id in CLASSES.items():
        src_dir = SRC_ROOT / src_split / cls_name
        if not src_dir.exists():
            logger.error(f"Source directory not found: {src_dir}")
            sys.exit(1)

        n = SPLIT_SIZES[dst_split]
        selected = sample_images(src_dir, cls_name, n, seed + cls_id)

        for img_path in selected:
            dst_name = f"{cls_name}_{img_path.name}"
            shutil.copy2(img_path, img_dst / dst_name)
            label_path = lbl_dst / (dst_name.rsplit(".", 1)[0] + ".txt")
            label_path.write_text(LABEL_TEMPLATE.format(cls_id=cls_id))

        counts[cls_name] = len(selected)
        logger.info(f"  {dst_split}/{cls_name}: copied {len(selected)} images")

    return counts


def write_data_yaml() -> None:
    yaml_path = DST_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(DATA_YAML, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote {yaml_path}")


def main() -> None:
    if DST_ROOT.exists():
        logger.info(f"Removing existing dataset at {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    seed = 42
    for src_split, dst_split in SPLIT_MAP.items():
        logger.info(f"Processing split: {src_split} -> {dst_split}")
        convert_split(src_split, dst_split, seed)

    write_data_yaml()
    logger.info("Dataset preparation complete.")
    logger.info(f"Dataset saved to: {DST_ROOT.resolve()}")


if __name__ == "__main__":
    main()
