import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf


SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 8
HEAD_LEARNING_RATE = 3e-4
FINE_TUNE_LEARNING_RATE = 1e-5

PROJECT_FOLDER = Path(__file__).resolve().parent
TRAIN_FOLDER = PROJECT_FOLDER / "retrain_data" / "train"
VALIDATION_FOLDER = PROJECT_FOLDER / "retrain_data" / "validation"
CURRENT_MODEL = PROJECT_FOLDER / "model" / "detect_now_efficientnetb0.keras"
OUTPUT_MODEL = PROJECT_FOLDER / "model" / "detect_now_efficientnetb0_improved.keras"
HEAD_MODEL = PROJECT_FOLDER / "model" / "detect_now_efficientnetb0_head.keras"
THRESHOLD_FILE = PROJECT_FOLDER / "model" / "decision_threshold.json"

# This order is intentional: sigmoid output 0 = Real and 1 = Fake.
CLASS_NAMES = ["real", "fake"]


def require_project_files():
    required = [
        CURRENT_MODEL,
        TRAIN_FOLDER / "real",
        TRAIN_FOLDER / "fake",
        VALIDATION_FOLDER / "real",
        VALIDATION_FOLDER / "fake",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Create the required model and dataset folders first:\n- "
            + "\n- ".join(missing)
        )


def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_datasets():
    common_arguments = {
        "labels": "inferred",
        "label_mode": "binary",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "crop_to_aspect_ratio": True,
    }

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_FOLDER,
        shuffle=True,
        seed=SEED,
        **common_arguments,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_FOLDER,
        shuffle=False,
        **common_arguments,
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_dataset.prefetch(autotune),
        validation_dataset.prefetch(autotune),
    )


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def prepare_head_training_model():
    model = tf.keras.models.load_model(CURRENT_MODEL, compile=False)
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = False
    compile_model(model, HEAD_LEARNING_RATE)
    return model


def prepare_fine_tuning_model():
    model = tf.keras.models.load_model(HEAD_MODEL, compile=False)
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    # Fine-tune more of the final feature extractor after the head adapts.
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )

    compile_model(model, FINE_TUNE_LEARNING_RATE)
    return model


def calculate_class_weights():
    real_count = count_images(TRAIN_FOLDER / "real")
    fake_count = count_images(TRAIN_FOLDER / "fake")

    if real_count == 0 or fake_count == 0:
        raise ValueError("Both training classes must contain images.")

    total = real_count + fake_count
    return {
        0: total / (2.0 * real_count),
        1: total / (2.0 * fake_count),
    }


def choose_best_threshold(model, validation_dataset):
    labels = np.concatenate([
        batch_labels.numpy().reshape(-1)
        for _, batch_labels in validation_dataset
    ]).astype(int)

    probabilities = model.predict(
        validation_dataset,
        verbose=0,
    ).reshape(-1)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.20, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        true_positive = np.sum((predictions == 1) & (labels == 1))
        false_positive = np.sum((predictions == 1) & (labels == 0))
        false_negative = np.sum((predictions == 0) & (labels == 1))

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return round(best_threshold, 2), round(best_f1, 4)


def main():
    tf.keras.utils.set_random_seed(SEED)
    require_project_files()

    train_dataset, validation_dataset = build_datasets()
    class_weights = calculate_class_weights()

    print("\nStage 1: adapting the classification head...\n")
    head_model = prepare_head_training_model()
    head_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            HEAD_MODEL,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    head_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=head_callbacks,
        shuffle=False,
    )

    print("\nStage 2: fine-tuning the final EfficientNetB0 layers...\n")
    model = prepare_fine_tuning_model()
    fine_tune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=fine_tune_callbacks,
        shuffle=False,
    )

    best_model = tf.keras.models.load_model(OUTPUT_MODEL, compile=False)
    best_threshold, best_f1 = choose_best_threshold(
        best_model,
        validation_dataset,
    )

    THRESHOLD_FILE.write_text(
        json.dumps(
            {
                "fake_threshold": best_threshold,
                "validation_f1": best_f1,
                "label_order": CLASS_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Improved model saved to: {OUTPUT_MODEL}")
    print(f"Recommended fake threshold: {best_threshold}")
    print(f"Validation F1 at this threshold: {best_f1}")
    print(f"Threshold information saved to: {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()
