"""Experimental Detect Now training on face swaps and paired portrait edits.

Run from the project folder with its virtual-environment Python.
The current website model and threshold are never overwritten.
"""

import json
import math
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import backend


PROJECT = Path(__file__).resolve().parent
FACE_DATA = Path(r"C:\Users\david\Downloads\detect_now_faceswap_crops")
PAIRS = PROJECT / "retrain_data" / "train"
BASE_MODEL = PROJECT / "model" / "detect_now_efficientnetb0_faceswap_crops.keras"
OUTPUT_MODEL = PROJECT / "model" / "detect_now_efficientnetb0_mixed.keras"
OUTPUT_THRESHOLD = PROJECT / "model" / "decision_threshold_mixed.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 2
EDIT_FRACTION = 0.05
SEED = 42
CLASSES = ["real", "fake"]


def paired_edit_dataset():
    """Include only matched edits, excluding the ten development pairs."""
    real_dir = PAIRS / "real"
    fake_dir = PAIRS / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(f"Paired edit folders missing under {PAIRS}")

    images = []
    labels = []
    pair_count = 0
    for fake_path in sorted(fake_dir.glob("manipulated_*.png")):
        number = fake_path.stem.removeprefix("manipulated_")
        if not number.isdecimal() or 9955 <= int(number) <= 9964:
            continue
        real_path = real_dir / f"real_{number}.png"
        if not real_path.exists():
            print(f"Skipping {fake_path.name}: missing {real_path.name}")
            continue
        prepared = []
        for path in (real_path, fake_path):
            with Image.open(path) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
            _, box, _ = backend.detect_human_face(rgb)
            if box is None:
                print(f"Skipping pair {number}: no face in {path.name}")
                break
            prepared.append(backend.preprocess_image(backend.crop_face(rgb, box))[0])
        if len(prepared) == 2:
            images.extend(prepared)
            labels.extend((0, 1))
            pair_count += 1

    if pair_count < 20:
        raise ValueError(f"Expected at least 20 matched edit pairs; found {pair_count}.")
    print(f"Matched portrait edit pairs: {pair_count}", flush=True)
    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1))
    )
    return ds.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True).repeat()


def face_swap_datasets():
    for split in ("train", "validation"):
        for label in CLASSES:
            if not (FACE_DATA / split / label).is_dir():
                raise FileNotFoundError(f"Missing {FACE_DATA / split / label}")

    settings = dict(
        labels="inferred",
        label_mode="binary",
        class_names=CLASSES,
        image_size=IMAGE_SIZE,
        crop_to_aspect_ratio=True,
        batch_size=BATCH_SIZE,
    )
    train = tf.keras.utils.image_dataset_from_directory(
        FACE_DATA / "train", shuffle=True, seed=SEED, **settings
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        FACE_DATA / "validation", shuffle=False, **settings
    )
    file_count = len(train.file_paths)
    return train.unbatch().repeat(), validation, file_count


def configure_model():
    if not BASE_MODEL.exists():
        raise FileNotFoundError(BASE_MODEL)
    model = tf.keras.models.load_model(BASE_MODEL, compile=False)
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:-50]:
        layer.trainable = False
    for layer in backbone.layers[-50:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def select_threshold(model, validation):
    labels = np.concatenate([y.numpy().ravel() for _, y in validation]).astype(int)
    scores = model.predict(validation, verbose=0).ravel()
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in np.arange(0.20, 0.96, 0.01):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        tn = int(np.sum(~predicted & (labels == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best[0]:
            best = (float(f1), float(threshold), (tp + tn) / len(labels), fp / max(fp + tn, 1))
    return best


def main():
    tf.keras.utils.set_random_seed(SEED)
    face_train, validation, face_count = face_swap_datasets()
    edit_train = paired_edit_dataset()
    mixed = tf.data.Dataset.sample_from_datasets(
        [face_train, edit_train],
        weights=[1 - EDIT_FRACTION, EDIT_FRACTION],
        seed=SEED,
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"Face swap training files: {face_count}", flush=True)
    print(f"Paired edits sampled in training: {EDIT_FRACTION:.0%}", flush=True)
    model = configure_model()
    model.fit(
        mixed,
        steps_per_epoch=math.ceil(face_count / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=validation,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL, monitor="val_auc", mode="max", save_best_only=True
        )],
    )

    best_model = tf.keras.models.load_model(OUTPUT_MODEL, compile=False)
    f1, threshold, accuracy, false_positive_rate = select_threshold(best_model, validation)
    OUTPUT_THRESHOLD.write_text(json.dumps({
        "fake_threshold": round(threshold, 2),
        "validation_f1": round(f1, 4),
        "validation_accuracy": round(accuracy, 4),
        "validation_false_positive_rate": round(false_positive_rate, 4),
        "label_order": CLASSES,
        "note": "Exploratory mixed training; validate separately on other edit types.",
    }, indent=2), encoding="utf-8")
    print(f"Mixed model: {OUTPUT_MODEL}")
    print(f"Mixed threshold: {threshold:.2f}")
    print(f"Face swap validation F1: {f1:.4f}")
    print(f"Face swap validation accuracy: {accuracy:.2%}")
    print(f"Face swap validation real false positive rate: {false_positive_rate:.2%}")
    print(f"Threshold information: {OUTPUT_THRESHOLD}")


if __name__ == "__main__":
    main()
