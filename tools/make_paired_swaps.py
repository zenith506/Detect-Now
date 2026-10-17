"""Build a PAIRED real/fake dataset (real image + same image with another face swapped in)."""
import argparse
import csv
import random
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def upscale(image):
    h, w = image.shape[:2]
    pad = int(0.5 * max(h, w))
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    ph, pw = padded.shape[:2]
    scale = max(1.0, 640.0 / min(ph, pw))
    if scale > 1.0:
        padded = cv2.resize(padded, (int(round(pw * scale)), int(round(ph * scale))), interpolation=cv2.INTER_CUBIC)
    return padded, pad, (h, w, ph, pw)


def restore(big, pad, dims):
    h, w, ph, pw = dims
    if big.shape[:2] != (ph, pw):
        big = cv2.resize(big, (pw, ph), interpolation=cv2.INTER_AREA)
    return big[pad:pad + h, pad:pad + w]


def assign_split(relative_name, test_share, val_share):
    bucket = zlib.crc32(relative_name.encode("utf-8")) % 1000
    if bucket < test_share * 1000:
        return "test"
    if bucket < (test_share + val_share) * 1000:
        return "val"
    return "train"


def largest_face(app, image):
    faces = app.get(image)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("real_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--model", default="inswapper_128.onnx")
    parser.add_argument("--val-share", type=float, default=0.15)
    parser.add_argument("--test-share", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    import insightface
    from insightface.app import FaceAnalysis

    paths = sorted(p for p in args.real_folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {args.real_folder}")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: args.count]

    splits = {"test": [], "val": [], "train": []}
    for path in paths:
        name = str(path.relative_to(args.real_folder))
        splits[assign_split(name, args.test_share, args.val_share)].append(path)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if args.gpu else -1, det_size=(640, 640))
    try:
        swapper = insightface.model_zoo.get_model(args.model, providers=providers)
    except TypeError:
        swapper = insightface.model_zoo.get_model(args.model)

    manifest_path = args.output_folder / "manifest.csv"
    args.output_folder.mkdir(parents=True, exist_ok=True)
    manifest = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(manifest)
    if manifest_path.stat().st_size == 0:
        writer.writerow(["split", "real_file", "fake_file", "target", "donor"])

    started = time.time()
    total_made = 0
    total_skipped = 0

    for split, split_paths in splits.items():
        print(f"\n[{split}] finding faces in {len(split_paths)} images...")
        faces = {}
        for path in split_paths:
            image = read_image(path)
            if image is None:
                continue
            big, _, _ = upscale(image)
            face = largest_face(app, big)
            if face is not None:
                faces[path] = face
        valid = sorted(faces.keys())
        total_skipped += len(split_paths) - len(valid)
        print(f"[{split}] {len(valid)} usable, {len(split_paths) - len(valid)} skipped (no face)")

        if len(valid) < 2:
            print(f"[{split}] not enough images, skipping this split")
            continue

        for target_path in valid:
            stem = f"{split}_{target_path.stem}"
            real_out = args.output_folder / split / "real" / f"{stem}.png"
            fake_out = args.output_folder / split / "fake" / f"{stem}_swap.png"
            if real_out.exists() and fake_out.exists():
                continue

            donor_path = rng.choice([p for p in valid if p != target_path])
            image = read_image(target_path)
            big, pad, dims = upscale(image)
            swapped = swapper.get(big, faces[target_path], faces[donor_path], paste_back=True)

            write_png(real_out, restore(big, pad, dims))
            write_png(fake_out, restore(swapped, pad, dims))
            writer.writerow([split, real_out.name, fake_out.name, target_path.name, donor_path.name])
            manifest.flush()
            total_made += 1

            if total_made % 25 == 0:
                print(f"  made {total_made} pairs ({(time.time() - started) / 60:.1f} min)")

    manifest.close()
    print(f"\nFinished: {total_made} new pairs, {total_skipped} skipped. Output: {args.output_folder}")


if __name__ == "__main__":
    main()