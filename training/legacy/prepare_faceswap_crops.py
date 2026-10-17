from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import backend


SOURCE = Path(r"C:\Users\david\Downloads\detect_now_faceswap_data")
OUTPUT = Path(r"C:\Users\david\Downloads\detect_now_faceswap_crops")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    total_saved = 0
    total_no_face = 0

    for split in ("train", "validation"):
        for label in ("real", "fake"):
            source_folder = SOURCE / split / label
            output_folder = OUTPUT / split / label
            output_folder.mkdir(parents=True, exist_ok=True)

            paths = sorted(
                path
                for path in source_folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in EXTENSIONS
            )

            saved = 0
            no_face = 0

            for path in paths:
                try:
                    with Image.open(path) as source_image:
                        image = ImageOps.exif_transpose(
                            source_image
                        ).convert("RGB")

                    _, box, _ = backend.detect_human_face(image)
                    if box is None:
                        no_face += 1
                        continue

                    processed = backend.preprocess_image(
                        backend.crop_face(image, box)
                    )[0]

                    pixels = np.clip(
                        processed, 0, 255
                    ).astype(np.uint8)

                    Image.fromarray(
                        pixels, mode="RGB"
                    ).save(output_folder / f"{path.stem}.png")

                    saved += 1

                except Exception as error:
                    print(f"Could not process {path}: {error}")
                    no_face += 1

                if (saved + no_face) % 1000 == 0:
                    print(
                        f"{split}/{label}: "
                        f"{saved + no_face}/{len(paths)} checked",
                        flush=True,
                    )

            print(
                f"{split}/{label}: saved {saved}, "
                f"no face or error {no_face}",
                flush=True,
            )

            total_saved += saved
            total_no_face += no_face

    print(f"Finished. Saved: {total_saved}")
    print(f"No face or error: {total_no_face}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()