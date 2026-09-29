"""Measure Detect Now at different decision thresholds.

Usage (from the project folder, with .venv active):

    python evaluate_threshold.py "path\\to\\real_images" "path\\to\\fake_images"

Optional:
    --thresholds 0.26 0.45 0.81     thresholds to highlight (default shown)
    --csv scores.csv                save every image's score
    --already-cropped               images are ready-made face crops (skip YuNet)

It runs the SAME steps as the website: YuNet face detection -> crop -> pad ->
resize -> model. Images with no detectable face are skipped (and counted),
exactly as the website would refuse them.

HONESTY WARNING: these numbers are only trustworthy if the images were never
used for training AND you did not pick the threshold by looking at these same
images. Choose the threshold on the validation set, then measure once on a
separate test set.
"""

import argparse
import csv
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from preprocessing import BASE_FOLDER, crop_face, detect_human_face, preprocess_image

MODEL_FILE = os.environ.get(
    "DETECT_NOW_MODEL_FILE",
    "detect_now_efficientnetb0_faceswap_crops.keras",
)
MODEL_PATH = os.path.join(BASE_FOLDER, "model", MODEL_FILE)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 32


def score_folder(model, folder, label, already_cropped=False):
    """Return ([(file name, label, fake score)], [(file name, reason)]).

    already_cropped=True treats every image as a ready-made face crop (the kind
    used for training) and skips face detection. That measures the model itself,
    NOT the full website pipeline."""
    names, batches, skipped = [], [], []

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            skipped.append((path.name, "unreadable"))
            continue

        if already_cropped:
            names.append(path.name)
            batches.append(preprocess_image(image)[0])
            continue

        _, face_box, _ = detect_human_face(image)
        if face_box is None:
            skipped.append((path.name, "no face found"))
            continue

        names.append(path.name)
        batches.append(preprocess_image(crop_face(image, face_box))[0])

    scores = []
    for start in range(0, len(batches), BATCH_SIZE):
        chunk = np.stack(batches[start:start + BATCH_SIZE])
        output = model.predict(chunk, verbose=0)
        scores.extend(float(v) for v in np.asarray(output).reshape(-1))

    rows = [(name, label, score) for name, score in zip(names, scores)]
    return rows, skipped


def metrics_at(real_scores, fake_scores, threshold):
    false_positives = int(np.sum(real_scores >= threshold))
    true_negatives = len(real_scores) - false_positives
    true_positives = int(np.sum(fake_scores >= threshold))
    false_negatives = len(fake_scores) - true_positives

    real_correct = true_negatives / max(len(real_scores), 1)
    fakes_caught = true_positives / max(len(fake_scores), 1)
    total = len(real_scores) + len(fake_scores)
    return {
        "threshold": threshold,
        "real_correct": real_correct,
        "fakes_caught": fakes_caught,
        "accuracy": (true_negatives + true_positives) / max(total, 1),
        "balanced": (real_correct + fakes_caught) / 2,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
    }


def auc_score(real_scores, fake_scores):
    """Chance that a random fake scores higher than a random real image.
    0.5 = no better than guessing, 1.0 = perfect. Needs no threshold."""
    fake = fake_scores[:, None]
    real = real_scores[None, :]
    return float((fake > real).mean() + 0.5 * (fake == real).mean())


def print_table(title, real_scores, fake_scores, thresholds):
    print(f"\n{title}")
    print(
        f"{'threshold':>9} | {'real correct':>16} | {'fakes caught':>16} | "
        f"{'accuracy':>8} | {'balanced':>8}"
    )
    print("-" * 72)
    for threshold in thresholds:
        m = metrics_at(real_scores, fake_scores, threshold)
        print(
            f"{threshold:>9.2f} | "
            f"{m['true_negatives']:>5}/{len(real_scores):<5} {m['real_correct']:>5.1%} | "
            f"{m['true_positives']:>5}/{len(fake_scores):<5} {m['fakes_caught']:>5.1%} | "
            f"{m['accuracy']:>8.1%} | {m['balanced']:>8.1%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("real_folder", type=Path)
    parser.add_argument("fake_folder", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.26, 0.45, 0.81])
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--already-cropped",
        action="store_true",
        help="images are already face crops (e.g. your train/val/test library): "
        "skip face detection",
    )
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Loading model: {MODEL_FILE}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Scoring real images...")
    real_rows, real_skipped = score_folder(
        model, args.real_folder, "real", args.already_cropped
    )
    print("Scoring fake images...")
    fake_rows, fake_skipped = score_folder(
        model, args.fake_folder, "fake", args.already_cropped
    )

    if not real_rows or not fake_rows:
        raise SystemExit("Need at least one scorable image in BOTH folders.")

    print(
        f"\nScored {len(real_rows)} real and {len(fake_rows)} fake images. "
        f"Skipped: {len(real_skipped)} real, {len(fake_skipped)} fake "
        "(no face found or unreadable)."
    )
    for name, reason in (real_skipped + fake_skipped)[:15]:
        print(f"  skipped {name}: {reason}")

    real_scores = np.array([row[2] for row in real_rows])
    fake_scores = np.array([row[2] for row in fake_rows])

    print_table("Your thresholds:", real_scores, fake_scores, args.thresholds)
    print_table(
        "Sweep:", real_scores, fake_scores,
        [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)],
    )

    print(
        f"\nAverage fake score on real images: {real_scores.mean():.3f}   "
        f"on fake images: {fake_scores.mean():.3f}"
    )
    print(
        f"AUC (threshold-free, 0.5 = guessing, 1.0 = perfect): "
        f"{auc_score(real_scores, fake_scores):.3f}"
    )
    print(
        "Remember: only quote these numbers if the images were never used for "
        "training and the threshold was not chosen on these same images."
    )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["file", "true_label", "fake_score"])
            writer.writerows(real_rows + fake_rows)
        print(f"Saved per-image scores to {args.csv}")


if __name__ == "__main__":
    main()