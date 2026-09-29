"""
Generates a YOLO eye-detection dataset from WiderFace images.

Pipeline per image:
  1. Read existing WiderFace label  →  face bbox (no model needed)
  2. Crop the largest face from the image
  3. MediaPipe Face Mesh  →  locate left & right eye landmarks
  4. Compute eye bboxes from landmarks  →  normalised YOLO labels (class 0 = eye)
  5. Save crop + label to  datasets/eyes_widerface/{split}/

Stops after TARGET_TOTAL valid face-crops with eyes are collected.
Finishes with a matplotlib preview of 10 random samples.
"""

import random
import shutil
from pathlib import Path

import cv2
import urllib.request

import mediapipe as mp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
WIDERFACE_DIR = ROOT / "datasets" / "widerface"
OUT_DIR       = ROOT / "datasets" / "eyes_widerface"

TARGET_TOTAL  = 2_000           # stop after this many saved face-crops
MIN_FACE_PX   = 60              # ignore face crops smaller than this (px)
EYE_PAD       = 0.25            # padding around eye landmarks (fraction of eye size)
SPLIT_RATIO   = {"train": 0.80, "valid": 0.10, "test": 0.10}
SEED          = 42

# MediaPipe face_landmarker.task model (auto-downloaded if missing)
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_PATH = ROOT / "face_landmarker.task"

# Face Mesh landmark indices for each eye contour (478-point model)
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# ───────────────────────────────────────────────────────────────────────────


def setup_dirs() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in SPLIT_RATIO:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def write_yaml() -> None:
    data = {
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["eye"],
    }
    with open(OUT_DIR / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def collect_image_paths() -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs where both files exist."""
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "valid", "test"):
        img_dir = WIDERFACE_DIR / split / "images"
        lbl_dir = WIDERFACE_DIR / split / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


def largest_face_crop(img_bgr: np.ndarray, lbl_path: Path) -> np.ndarray | None:
    """Read WiderFace YOLO label, pick the largest face bbox, return the crop."""
    H, W = img_bgr.shape[:2]
    lines = lbl_path.read_text().strip().splitlines()
    if not lines:
        return None

    best_area, best_crop = 0, None
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area and (x2 - x1) >= MIN_FACE_PX and (y2 - y1) >= MIN_FACE_PX:
            best_area = area
            best_crop = img_bgr[y1:y2, x1:x2]

    return best_crop


def assign_split(rng: random.Random) -> str:
    r = rng.random()
    if r < SPLIT_RATIO["train"]:
        return "train"
    if r < SPLIT_RATIO["train"] + SPLIT_RATIO["valid"]:
        return "valid"
    return "test"


def landmarks_to_bbox(
    lm_indices: list[int],
    landmarks,
    fw: int,
    fh: int,
) -> tuple[float, float, float, float] | None:
    """Convert a set of landmark indices to a padded (cx, cy, w, h) normalised bbox."""
    xs = [landmarks[i].x * fw for i in lm_indices]
    ys = [landmarks[i].y * fh for i in lm_indices]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * EYE_PAD
    pad_y = (y2 - y1) * EYE_PAD
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(fw, x2 + pad_x)
    y2 = min(fh, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, (x2 - x1) / fw, (y2 - y1) / fh


def detect_eyes(
    face_crop: np.ndarray,
    landmarker: mp.tasks.vision.FaceLandmarker,
) -> list[tuple[float, float, float, float]]:
    """Run MediaPipe FaceLandmarker on a face crop, return normalised (cx, cy, w, h) eye bboxes."""
    fh, fw = face_crop.shape[:2]
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return []

    landmarks = result.face_landmarks[0]
    bboxes = []
    for indices in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        bbox = landmarks_to_bbox(indices, landmarks, fw, fh)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def visualise_samples(
    records: list[tuple[Path, list[tuple[float, float, float, float]]]],
    n: int = 10,
) -> None:
    samples = random.sample(records, min(n, len(records)))
    cols = 5
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, (img_path, bboxes) in zip(axes, samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fh, fw = img.shape[:2]
        ax.imshow(img)
        for cx, cy, bw, bh in bboxes:
            rx = (cx - bw / 2) * fw
            ry = (cy - bh / 2) * fh
            rect = mpatches.FancyBboxPatch(
                (rx, ry), bw * fw, bh * fh,
                linewidth=2, edgecolor="lime", facecolor="none",
                boxstyle="round,pad=1",
            )
            ax.add_patch(rect)
        ax.set_title(img_path.stem, fontsize=7)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.suptitle("10 random samples — eyes_widerface", fontsize=13, y=1.01)
    plt.tight_layout()
    preview_path = OUT_DIR / "samples_preview.png"
    plt.savefig(str(preview_path), dpi=120, bbox_inches="tight")
    print(f"\nPreview saved → {preview_path}")
    plt.show()


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    setup_dirs()
    write_yaml()

    if not LANDMARKER_PATH.exists():
        print(f"Downloading face_landmarker.task …")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print(f"Saved → {LANDMARKER_PATH}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    all_pairs = collect_image_paths()
    random.shuffle(all_pairs)
    print(f"WiderFace pool: {len(all_pairs)} images  |  target: {TARGET_TOTAL}")

    saved = 0
    counters = {"train": 0, "valid": 0, "test": 0}
    records: list[tuple[Path, list]] = []

    pbar = tqdm(all_pairs, desc="Processing", unit="img")
    for img_path, lbl_path in pbar:
        if saved >= TARGET_TOTAL:
            break

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        face_crop = largest_face_crop(img_bgr, lbl_path)
        if face_crop is None:
            continue

        bboxes = detect_eyes(face_crop, landmarker)
        if not bboxes:
            continue

        # ── save face crop + YOLO label ────────────────────────────────────
        split = assign_split(rng)
        stem  = f"wf_{saved:05d}"
        out_img = OUT_DIR / split / "images" / f"{stem}.jpg"
        out_lbl = OUT_DIR / split / "labels" / f"{stem}.txt"

        cv2.imwrite(str(out_img), face_crop)
        lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in bboxes]
        out_lbl.write_text("\n".join(lines))

        records.append((out_img, bboxes))
        counters[split] += 1
        saved += 1
        pbar.set_postfix(saved=saved, **counters)

    print(f"\nFinished. Saved {saved} samples.")
    print(f"  train: {counters['train']}  valid: {counters['valid']}  test: {counters['test']}")
    print(f"  Dataset → {OUT_DIR.resolve()}")

    landmarker.close()  # type: ignore[union-attr]

    if records:
        visualise_samples(records, n=10)


if __name__ == "__main__":
    main()
