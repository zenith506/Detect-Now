"""Run Detect Now over the team's shared test images and write a results file.

    python score_shared_test.py shared_test_images\\images results_detectnow_v2.csv --threshold 0.40

Choose the model with DETECT_NOW_MODEL_FILE (file name inside the model folder),
exactly as for the backend. The threshold is required so it is never guessed.

Uses the SAME steps as the website: YuNet face detection -> crop -> pad -> resize.
An image where no face is found gets an empty verdict (the website would refuse it too).
Then score it with:   python score_results.py answer_key.csv results_detectnow_v2.csv
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

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--threshold", type=float, required=True, help="fake if score >= this (0.05-0.95)")
    args = parser.parse_args()

    model_file = os.environ.get("DETECT_NOW_MODEL_FILE", "detect_now_efficientnetb0_faceswap_crops.keras")
    model_path = os.path.join(BASE_FOLDER, "model", model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Model: {model_file} | threshold: {args.threshold}")
    model = tf.keras.models.load_model(model_path, compile=False)

    files = sorted(p for p in args.image_folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in {args.image_folder}")

    rows, refused = [], 0
    for path in files:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        _, face_box, _ = detect_human_face(image)
        if face_box is None:
            rows.append((path.name, "", ""))
            refused += 1
            continue
        batch = preprocess_image(crop_face(image, face_box))
        score = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
        rows.append((path.name, "fake" if score >= args.threshold else "real", round(score, 4)))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "verdict", "score"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv} ({refused} images had no face found -> blank verdict)")


if __name__ == "__main__":
    main()
