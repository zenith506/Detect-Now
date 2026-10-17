"""Turn raw images into model-ready 224x224 face crops using the SAME steps as backend.py:
YuNet face -> crop with 35% margin -> pad to square -> Lanczos resize to 224."""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

WORK = Path("/content/work")
OUT = WORK / "data_ready"
YUNET_MODEL_PATH = str(WORK / "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THRESHOLD = 0.80
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
YUNET_MAX_DETECTION_SIZE = 640
MODEL_INPUT_SIZE = (224, 224)
FACE_CROP_MARGIN = 0.35
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Images that belong to the team's shared test set. Anything made from them
# (as the picture OR as the donor face) is kept out of training.
SHARED_TEST_STEMS = {
    "real_9940", "real_9941", "real_9942", "real_9943", "real_9944", "real_9945", "real_9946",
    "real_9947", "real_9948", "real_9949", "real_995", "real_9950", "real_9951", "real_9952",
    "real_9953", "real_9954", "real_9955", "real_9956", "real_9957", "real_9958", "real_9959",
    "real_996", "real_9960", "real_9961", "real_9962", "real_9963", "real_9964", "real_9965",
    "real_9966", "real_9967", "real_9968", "real_9969", "real_997", "real_9970", "real_9971",
    "real_9972", "real_9973", "real_9974", "real_9975", "real_9976", "real_9977", "real_9978",
    "real_9979", "real_998", "real_9980", "real_9981", "real_9982", "real_9983", "real_9984",
    "real_999",
}

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320),
            YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K,
        )
    return _detector


def detect_face(image):
    """Same logic as backend.py detect_human_face. Returns a face box dict or None."""
    detector = get_detector()
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    original_height, original_width = bgr.shape[:2]
    scale = min(1.0, YUNET_MAX_DETECTION_SIZE / float(max(original_width, original_height)))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = bgr.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: float(f[2] * f[3] * f[14]))
    inverse = 1.0 / scale
    x = max(0, min(int(round(float(face[0]) * inverse)), original_width - 1))
    y = max(0, min(int(round(float(face[1]) * inverse)), original_height - 1))
    w = max(1, min(int(round(float(face[2]) * inverse)), original_width - x))
    h = max(1, min(int(round(float(face[3]) * inverse)), original_height - y))
    return {"x": x, "y": y, "width": w, "height": h}


def crop_face(image, box, margin=FACE_CROP_MARGIN):
    width, height = image.size
    mx, my = int(box["width"] * margin), int(box["height"] * margin)
    return image.crop((
        max(0, box["x"] - mx), max(0, box["y"] - my),
        min(width, box["x"] + box["width"] + mx), min(height, box["y"] + box["height"] + my),
    ))


def to_model_image(image):
    width, height = image.size
    size = max(width, height)
    square = Image.new("RGB", (size, size), (0, 0, 0))
    square.paste(image, ((size - width) // 2, (size - height) // 2))
    return square.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)


def clean_stem(file_name):
    """train_real_9940_swap.png -> real_9940"""
    stem = Path(file_name).stem
    for prefix in ("train_", "val_", "test_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if stem.endswith("_swap"):
        stem = stem[:-5]
    return stem


def is_shared_test(name):
    return Path(name).stem.lower() in SHARED_TEST_STEMS


def convert(files, out_dir, prefix):
    made, skipped = 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            skipped += 1
            continue
        box = detect_face(image)
        if box is None:
            skipped += 1
            continue
        to_model_image(crop_face(image, box)).save(out_dir / f"{prefix}{path.stem}.png")
        made += 1
    return made, skipped


def list_images(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def paired_sources(split):
    """(real_files, fake_files) for one split, dropping pairs that touch the shared test set."""
    base = WORK / "paired_data"
    manifest = {}
    manifest_path = base / "manifest.csv"
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("split") == split:
                    manifest[row["fake_file"]] = row
    real_files, fake_files, dropped = [], [], 0
    for fake in list_images(base / split / "fake"):
        row = manifest.get(fake.name)
        target = Path(row["target"]).stem.lower() if row else clean_stem(fake.name).lower()
        donor = Path(row["donor"]).stem.lower() if row else ""
        if target in SHARED_TEST_STEMS or donor in SHARED_TEST_STEMS:
            dropped += 1
            continue
        real = base / split / "real" / (fake.name[: -len("_swap.png")] + ".png")
        if real.exists():
            real_files.append(real)
            fake_files.append(fake)
    return real_files, fake_files, dropped


def main():
    if not Path(YUNET_MODEL_PATH).exists():
        raise SystemExit(f"YuNet model not found: {YUNET_MODEL_PATH}")

    jobs = []  # (split, class, files, prefix)
    for split in ("train", "val", "test"):
        real_files, fake_files, dropped = paired_sources(split)
        print(f"paired_data/{split}: {len(real_files)} pairs kept, {dropped} dropped (touch shared test set)")
        jobs.append((split, "real", real_files, "pd_"))
        jobs.append((split, "fake", fake_files, "pd_"))

    kaggle = WORK / "kaggle_subset"
    for split in ("train", "val"):
        for source_class, label in (("original", "real"), ("altered", "fake")):
            files = list_images(kaggle / split / source_class)
            print(f"kaggle_subset/{split}/{source_class}: found {len(files)} files")
            jobs.append((split, label, files, "kg_"))

    if not any(files for _, _, files, _ in jobs):
        raise SystemExit("No images found anywhere. Check the copy/unzip step.")

    total_skipped = 0
    for split, label, files, prefix in jobs:
        made, skipped = convert(files, OUT / split / label, prefix)
        total_skipped += skipped
        print(f"  -> {split}/{label} from {prefix[:-1]}: {made} saved, {skipped} skipped")

    print("\nFINAL COUNTS in data_ready:")
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            print(f"  {split}/{label}: {len(list_images(OUT / split / label))}")
    print(f"skipped in total: {total_skipped}")


if __name__ == "__main__":
    main()