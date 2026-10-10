import os

import cv2
import numpy as np
from PIL import Image

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

MODEL_INPUT_SIZE = (224, 224)
FACE_CROP_MARGIN = 0.35

YUNET_MODEL_PATH = os.path.join(
    BASE_FOLDER,
    "model",
    "face_detection_yunet_2023mar.onnx",
)
YUNET_SCORE_THRESHOLD = 0.80
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
YUNET_MAX_DETECTION_SIZE = 640

yunet_detector = None


def load_yunet_detector():
    global yunet_detector

    if yunet_detector is not None:
        return yunet_detector

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "This OpenCV installation does not include FaceDetectorYN. "
            "Install or upgrade opencv-python."
        )

    if not os.path.exists(YUNET_MODEL_PATH):
        raise FileNotFoundError(
            "YuNet face detector model was not found. Expected: "
            f"{YUNET_MODEL_PATH}"
        )

    yunet_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K,
    )
    return yunet_detector


def detect_human_face(image):
    """Return (face_count, face_box, method). face_box is None if no face."""
    detector = load_yunet_detector()

    rgb_image = np.asarray(image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    original_height, original_width = bgr_image.shape[:2]
    longest_side = max(original_width, original_height)
    scale = min(1.0, YUNET_MAX_DETECTION_SIZE / float(longest_side))

    if scale < 1.0:
        detection_width = max(1, int(original_width * scale))
        detection_height = max(1, int(original_height * scale))
        detection_image = cv2.resize(
            bgr_image,
            (detection_width, detection_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_image = bgr_image

    detection_height, detection_width = detection_image.shape[:2]
    detector.setInputSize((detection_width, detection_height))
    _, faces = detector.detect(detection_image)

    if faces is None or len(faces) == 0:
        return 0, None, "none"

    selected_face = max(
        faces,
        key=lambda face: float(face[2] * face[3] * face[14]),
    )

    inverse_scale = 1.0 / scale
    x = int(round(float(selected_face[0]) * inverse_scale))
    y = int(round(float(selected_face[1]) * inverse_scale))
    width = int(round(float(selected_face[2]) * inverse_scale))
    height = int(round(float(selected_face[3]) * inverse_scale))
    face_score = float(selected_face[14])

    x = max(0, min(x, original_width - 1))
    y = max(0, min(y, original_height - 1))
    width = max(1, min(width, original_width - x))
    height = max(1, min(height, original_height - y))

    face_box = {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "score": round(face_score, 6),
    }
    return len(faces), face_box, "yunet"


def crop_face(image, face_box, margin=FACE_CROP_MARGIN):
    if not face_box:
        return image

    width, height = image.size
    x = face_box["x"]
    y = face_box["y"]
    w = face_box["width"]
    h = face_box["height"]

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    crop_x1 = max(0, x - margin_x)
    crop_y1 = max(0, y - margin_y)
    crop_x2 = min(width, x + w + margin_x)
    crop_y2 = min(height, y + h + margin_y)

    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def pad_to_square(image):
    """Pad without stretching so a partial face at an edge is preserved."""
    width, height = image.size
    square_size = max(width, height)

    padded_image = Image.new("RGB", (square_size, square_size), (0, 0, 0))
    left = (square_size - width) // 2
    top = (square_size - height) // 2
    padded_image.paste(image, (left, top))
    return padded_image


def to_model_image(image):
    """Face crop -> square -> 224x224 PIL image. This is the model's input."""
    return pad_to_square(image).resize(
        MODEL_INPUT_SIZE,
        Image.Resampling.LANCZOS,
    )


def preprocess_image(image):
    """Face crop -> float32 batch of shape (1, 224, 224, 3), values 0-255."""
    image_array = np.asarray(to_model_image(image), dtype=np.float32)
    return np.expand_dims(image_array, axis=0)
