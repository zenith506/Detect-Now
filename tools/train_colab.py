"""Fine-tune Detect Now on Kaggle + inswapper data (run in Colab).

Reads data_ready/{train,val,test}/{real,fake} made by prepare_data.py.
File names starting kg_ come from Kaggle, pd_ from the inswapper pairs.
The threshold is chosen on VAL only. TEST (inswapper pairs the model never saw)
is used once, at the end, for the honest number.
"""
import json
import os
import shutil
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

WORK = Path("/content/work")
DATA = WORK / "data_ready"
BASE_MODEL = WORK / "detect_now_efficientnetb0_faceswap_crops.keras"
STEM = "detect_now_efficientnetb0_faceswap_crops_v2"
OUT_MODEL = WORK / f"{STEM}.keras"
OUT_THRESHOLD = WORK / f"decision_threshold_{STEM}.json"
DRIVE_OUT = Path("/content/drive/MyDrive/detect_now_v2")

SEED = 42
BATCH = 32
EPOCHS = 12
LEARNING_RATE = 2e-5
UNFREEZE_LAST = 100          # backbone layers that may change (batch-norm stays frozen)
MAX_FALSE_POSITIVE_RATE = 0.05
CLASS_NAMES = ["real", "fake"]   # sigmoid output: 0 = real, 1 = fake


def auc_score(real_scores, fake_scores):
    fake, real = fake_scores[:, None], real_scores[None, :]
    return float((fake > real).mean() + 0.5 * (fake == real).mean())


def metrics_at(labels, probs, threshold):
    pred = probs >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    tn = int(np.sum(~pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": round(float(threshold), 2),
        "real_correct": tn / max(tn + fp, 1),
        "fakes_caught": recall,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
    }


def choose_threshold(labels, probs):
    """Best F1 while real images wrongly flagged stay under the cap."""
    best_any, best_capped = None, None
    for t in np.arange(0.05, 0.96, 0.01):
        m = metrics_at(labels, probs, t)
        if best_any is None or m["f1"] > best_any["f1"]:
            best_any = m
        if m["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE:
            if best_capped is None or m["f1"] > best_capped["f1"]:
                best_capped = m
    return best_capped or best_any


def subset_auc(labels, probs, mask):
    labels, probs = labels[mask], probs[mask]
    if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
        return float("nan")
    return auc_score(probs[labels == 0], probs[labels == 1])


def find_backbone(tf, model):
    try:
        return model.get_layer("efficientnetb0")
    except ValueError:
        pass
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Could not find the EfficientNet backbone: " + ", ".join(l.name for l in model.layers))


def load_split(tf, split, **kwargs):
    return tf.keras.utils.image_dataset_from_directory(
        str(DATA / split), labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
        image_size=(224, 224), seed=SEED, **kwargs,
    )


def score_split(tf, model, split):
    ds = load_split(tf, split, batch_size=64, shuffle=False)
    paths = list(ds.file_paths)
    labels = np.concatenate([y.numpy().reshape(-1) for _, y in ds]).astype(int)
    probs = np.asarray(model.predict(ds, verbose=0)).reshape(-1)
    return paths, labels, probs


def augment_fn(tf):
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)

        def jpeg():
            small = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
            small = tf.image.random_jpeg_quality(small, 45, 95)
            return tf.cast(small, tf.float32)

        image = tf.cond(tf.random.uniform([]) < 0.5, jpeg, lambda: image)
        image = tf.image.random_brightness(image, 20.0)
        image = tf.image.random_contrast(image, 0.85, 1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
        image.set_shape([224, 224, 3])
        return image, label
    return augment


def report(title, paths, labels, probs, threshold=None):
    is_kaggle = np.array([Path(p).name.startswith("kg_") for p in paths])
    print(f"\n{title}")
    for name, mask in (("Kaggle ", is_kaggle), ("inswapper", ~is_kaggle)):
        if mask.sum() == 0:
            continue
        line = f"  {name:9s} n={int(mask.sum()):4d}  AUC={subset_auc(labels, probs, mask):.3f}"
        if threshold is not None:
            m = metrics_at(labels[mask], probs[mask], threshold)
            line += (f"  @ {threshold:.2f}: real correct {m['real_correct']:.1%}, "
                     f"fakes caught {m['fakes_caught']:.1%}, accuracy {m['accuracy']:.1%}")
        print(line)


def main():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    print("TensorFlow", tf.__version__, "| Keras", tf.keras.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # ---- data ----
    train = (load_split(tf, "train", batch_size=None, shuffle=False)
             .cache()
             .shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
             .map(augment_fn(tf), num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    val = load_split(tf, "val", batch_size=BATCH, shuffle=False).cache().prefetch(tf.data.AUTOTUNE)

    # ---- "before" numbers for the model as it is today ----
    old_model = tf.keras.models.load_model(str(BASE_MODEL), compile=False)
    v_paths, v_labels, old_val = score_split(tf, old_model, "val")
    t_paths, t_labels, old_test = score_split(tf, old_model, "test")
    report("BEFORE (current model) - validation", v_paths, v_labels, old_val, 0.45)
    report("BEFORE (current model) - test", t_paths, t_labels, old_test, 0.45)
    del old_model

    # ---- fine-tune ----
    model = tf.keras.models.load_model(str(BASE_MODEL), compile=False)
    model.trainable = True
    backbone = find_backbone(tf, model)
    backbone.trainable = True
    for layer in backbone.layers[:-UNFREEZE_LAST]:
        layer.trainable = False
    for layer in backbone.layers[-UNFREEZE_LAST:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    print("trainable weights:", sum(int(np.prod(w.shape)) for w in model.trainable_weights))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(
        train, validation_data=val, epochs=EPOCHS, verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(OUT_MODEL), monitor="val_auc", mode="max", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7),
        ],
    )

    # ---- "after" numbers ----
    best = tf.keras.models.load_model(str(OUT_MODEL), compile=False)
    v_paths, v_labels, new_val = score_split(tf, best, "val")
    choice = choose_threshold(v_labels, new_val)
    threshold = choice["threshold"]
    print(f"\nThreshold chosen on validation (false positives <= {MAX_FALSE_POSITIVE_RATE:.0%}): {threshold}")
    report("AFTER (v2 model) - validation (optimistic: threshold picked here)", v_paths, v_labels, new_val, threshold)
    t_paths, t_labels, new_test = score_split(tf, best, "test")
    report("AFTER (v2 model) - TEST, never seen (use this one)", t_paths, t_labels, new_test, threshold)

    OUT_THRESHOLD.write_text(json.dumps({
        "fake_threshold": threshold,
        "model_file": OUT_MODEL.name,
        "label_order": CLASS_NAMES,
        "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "validation": choice,
    }, indent=2), encoding="utf-8")

    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT_MODEL, DRIVE_OUT / OUT_MODEL.name)
    shutil.copy(OUT_THRESHOLD, DRIVE_OUT / OUT_THRESHOLD.name)
    print("\nSaved to Drive folder:", DRIVE_OUT)
    print(" ", OUT_MODEL.name)
    print(" ", OUT_THRESHOLD.name)


if __name__ == "__main__":
    main()