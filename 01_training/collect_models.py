"""
Finds best.pt weights and result images in the runs/ directory
and copies them to models/ and results/.

Run after training (or while face training is still in progress):
    python collect_models.py
"""

import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

RESULT_IMAGES = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
]

TARGETS = [
    {
        "run_name": "states_detector",
        "model_dst": "states_detector.pt",
        "results_dst": "training_states",
    },
    {
        "run_name": "face_detector",
        "model_dst": "face_detector.pt",
        "results_dst": "training_face",
    },
]


def find_run_dir(runs_root: Path, run_name: str) -> Path | None:
    """Find the most recent run directory matching run_name anywhere under runs_root."""
    candidates = sorted(
        runs_root.rglob(f"{run_name}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent.parent
    return None


def collect(target: dict) -> None:
    run_name = target["run_name"]
    run_dir = find_run_dir(RUNS_DIR, run_name)

    if run_dir is None:
        logger.warning(f"[{run_name}] No run directory found — skipping")
        return

    best_pt = run_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst_model = MODELS_DIR / target["model_dst"]
    shutil.copy2(best_pt, dst_model)
    logger.info(f"[{run_name}] Copied model  {best_pt} -> {dst_model}")

    results_out = RESULTS_DIR / target["results_dst"]
    results_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for filename in RESULT_IMAGES:
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, results_out / filename)
            copied += 1
    logger.info(f"[{run_name}] Copied {copied} result image(s) -> {results_out}")


def main() -> None:
    if not RUNS_DIR.exists():
        logger.error(f"runs/ directory not found. Run training first.")
        return

    for target in TARGETS:
        collect(target)


if __name__ == "__main__":
    main()
