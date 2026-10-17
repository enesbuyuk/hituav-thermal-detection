"""Thermal preprocessing pipeline: Median(3x3) -> CLAHE -> Unsharp Masking(k=0.5).

Mirrors Section 3.3 of notebooks/hit_uav_yolov8.ipynb. Reads the raw HIT-UAV dataset,
writes processed images plus labels (with DontCare boxes removed) to a new directory.

Usage:
    python scripts/preprocess.py [--src dataset/hit-uav] [--dst dataset/hit-uav-processed]
"""

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

SPLITS = ("train", "val", "test")
NUM_CLASSES = 4  # 0 Person, 1 Car, 2 Motorcycle, 3 Other Vehicle (4 = DontCare is dropped)


def preprocess_thermal(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)  # (1+k)*img - k*blur, k=0.5
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def filter_labels(src: Path) -> tuple[list[str], int]:
    kept, dropped = [], 0
    if src.exists():
        for line in src.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) < NUM_CLASSES:
                kept.append(line.strip())
            elif parts:
                dropped += 1
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("dataset/hit-uav"))
    parser.add_argument("--dst", type=Path, default=Path("dataset/hit-uav-processed"))
    args = parser.parse_args()

    if not (args.src / "images").is_dir():
        parser.error(f"{args.src}/images not found — download the dataset first (./setup.sh)")
    if args.dst.exists():
        shutil.rmtree(args.dst)

    total, dropped = 0, 0
    for split in SPLITS:
        img_out = args.dst / "images" / split
        lbl_out = args.dst / "labels" / split
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)

        for img_path in tqdm(sorted((args.src / "images" / split).glob("*.jpg")), desc=split):
            processed = preprocess_thermal(cv2.imread(str(img_path)))
            cv2.imwrite(str(img_out / img_path.name), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

            kept, n_dropped = filter_labels(args.src / "labels" / split / f"{img_path.stem}.txt")
            (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
            dropped += n_dropped
            total += 1

    print(f"Processed {total} images -> {args.dst} (removed {dropped} DontCare boxes)")


if __name__ == "__main__":
    main()
