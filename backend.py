import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datetime import datetime, timezone
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError


app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_FOLDER,
    "model",
    "detect_now_efficientnetb0.keras",
)

MODEL_NAME = "Detect Now EfficientNetB0"
MODEL_INPUT_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
}

model = None

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml",
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_detector.empty():
    raise RuntimeError(
        "OpenCV face detector could not be loaded."
    )


def load_detection_model():
    global model

    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    print(
        "Loading Detect Now EfficientNetB0 model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(
        "Detect Now EfficientNetB0 model is ready."
    )

    return model


def allowed_file(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1,
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def read_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()

    if not image_bytes:
        raise ValueError(
            "The uploaded file is empty."
        )

    try:
        image = Image.open(
            BytesIO(image_bytes)
        )

        image.verify()

        image = Image.open(
            BytesIO(image_bytes)
        )

        image = ImageOps.exif_transpose(image)

        return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from error


def detect_human_face(image):
    image_array = np.asarray(image)

    gray_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY,
    )

    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return 0, None

    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3],
    )

    face_box = {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }

    return len(faces), face_box


def preprocess_image(image):
    resized_image = image.resize(
        MODEL_INPUT_SIZE,
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    return np.expand_dims(
        image_array,
        axis=0,
    )


def predict_image(image_batch):
    detection_model = load_detection_model()

    output = detection_model.predict(
        image_batch,
        verbose=0,
    )

    fake_probability = float(
        np.asarray(output).reshape(-1)[0]
    )

    fake_probability = float(
        np.clip(
            fake_probability,
            0.0,
            1.0,
        )
    )

    real_probability = (
        1.0 - fake_probability
    )

    if fake_probability >= 0.5:
        prediction = "DEEPFAKE"
        confidence = fake_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
    }


def create_explanation(
    prediction,
    confidence,
):
    percentage = round(
        confidence * 100,
        2,
    )

    if prediction == "DEEPFAKE":
        return (
            "The analysed content is likely to be "
            f"Deepfake. The model confidence is "
            f"{percentage}%."
        )

    return (
        "The analysed content is likely to be "
        f"Real. The model confidence is "
        f"{percentage}%."
    )


@app.get("/")
def home():
    return jsonify({
        "application": "Detect Now",
        "status": (
            "ready"
            if model is not None
            else "loading"
        ),
        "model": MODEL_NAME,
        "architecture": "EfficientNetB0",
        "framework": "TensorFlow",
        "supported_media": ["image"],
        "message": (
            "Detect Now backend is running."
        ),
    })


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({
        "status": (
            "ready"
            if model is not None
            else "loading"
        ),
        "model": MODEL_NAME,
        "model_file": os.path.basename(
            MODEL_PATH
        ),
        "architecture": "EfficientNetB0",
        "framework": "TensorFlow",
        "tensorflow_version": tf.__version__,
    })


@app.post("/predict")
@app.post("/api/predict")
@app.post("/analyze")
@app.post("/api/analyze")
@app.post("/detect")
@app.post("/api/detect")
def predict():
    uploaded_file = (
        request.files.get("file")
        or request.files.get("image")
        or request.files.get("media")
    )

    if uploaded_file is None:
        return jsonify({
            "success": False,
            "error": "No image was uploaded.",
            "message": "Please select an image.",
        }), 400

    filename = (
        uploaded_file.filename or ""
    )

    if not filename:
        return jsonify({
            "success": False,
            "error": "No image was selected.",
            "message": "Please select an image.",
        }), 400

    if not allowed_file(filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format.",
            "message": (
                "Upload a JPG, JPEG, PNG, "
                "WEBP or BMP image."
            ),
        }), 400

    try:
        original_image = read_uploaded_image(
            uploaded_file
        )

        face_count, face_box = (
            detect_human_face(
                original_image
            )
        )

        if face_count == 0:
            return jsonify({
                "success": False,
                "error": (
                    "No human face was detected."
                ),
                "message": (
                    "Unable to analyse: no clear "
                    "human face was detected."
                ),
                "face_detected": False,
                "faces_detected": 0,
            }), 422

        image_batch = preprocess_image(
            original_image
        )

        result = predict_image(
            image_batch
        )

        prediction = result["prediction"]
        confidence = result["confidence"]

        real_probability = result[
            "real_probability"
        ]

        fake_probability = result[
            "fake_probability"
        ]

        confidence_percentage = round(
            confidence * 100,
            2,
        )

        real_percentage = round(
            real_probability * 100,
            2,
        )

        fake_percentage = round(
            fake_probability * 100,
            2,
        )

        explanation = create_explanation(
            prediction,
            confidence,
        )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        return jsonify({
            "success": True,
            "prediction": prediction,
            "result": prediction,
            "classification": prediction,
            "label": prediction,
            "confidence": (
                confidence_percentage
            ),
            "confidence_percentage": (
                confidence_percentage
            ),
            "real_probability": (
                real_percentage
            ),
            "fake_probability": (
                fake_percentage
            ),
            "real_score": real_percentage,
            "fake_score": fake_percentage,
            "scores": {
                "real": round(
                    real_probability,
                    6,
                ),
                "fake": round(
                    fake_probability,
                    6,
                ),
            },
            "face_detected": True,
            "faces_detected": int(
                face_count
            ),
            "face_box": face_box,
            "filename": filename,
            "image_width": (
                original_image.width
            ),
            "image_height": (
                original_image.height
            ),
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "model_name": MODEL_NAME,
            "architecture": (
                "EfficientNetB0"
            ),
            "framework": "TensorFlow",
            "input_size": "224 x 224",
            "explanation": explanation,
            "summary": explanation,
            "performance": {
                "accuracy": 76.67,
                "precision": 82.56,
                "recall": 67.62,
                "f1_score": 74.35,
                "auc": 87.09,
            },
            "warning": (
                "This is an academic prototype. "
                "The result may be incorrect and "
                "should not be treated as "
                "forensic proof."
            ),
        }), 200

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error),
            "message": str(error),
        }), 400

    except Exception as error:
        print(
            f"Prediction error: {error}"
        )

        return jsonify({
            "success": False,
            "error": (
                "The image could not be analysed."
            ),
            "message": str(error),
        }), 500


@app.post("/analyze-video")
@app.post("/api/analyze-video")
def analyze_video():
    return jsonify({
        "success": False,
        "error": (
            "Video detection is not available yet."
        ),
        "message": (
            "Video detection will be added "
            "in the next development stage."
        ),
    }), 501


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": (
            "The uploaded image is too large."
        ),
        "message": (
            "Upload an image smaller than 10 MB."
        ),
    }), 413


if __name__ == "__main__":
    try:
        load_detection_model()

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
        )

    except Exception as error:
        print(
            "The Detect Now backend "
            "could not start."
        )

        print(f"Reason: {error}")