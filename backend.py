import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

_THREADS = os.environ.get("DETECT_NOW_THREADS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_name, _THREADS)

from datetime import datetime, timezone
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError

from preprocessing import (
    MODEL_INPUT_SIZE,
    YUNET_MAX_DETECTION_SIZE,
    YUNET_MODEL_PATH,
    YUNET_NMS_THRESHOLD,
    YUNET_SCORE_THRESHOLD,
    YUNET_TOP_K,
    crop_face,
    detect_human_face,
    load_yunet_detector,
    pad_to_square,
    preprocess_image,
)

HAS_TF = True  

cv2.setNumThreads(int(_THREADS))
try:
    tf.config.threading.set_intra_op_parallelism_threads(int(_THREADS))
    tf.config.threading.set_inter_op_parallelism_threads(int(_THREADS))
except RuntimeError:
    pass  

def log(message):
    """Print right away so the Render log shows where a request is."""
    print(f"[detect-now] {message}", flush=True)


app = Flask(__name__)

_custom_origins = os.environ.get("DETECT_NOW_CORS_ORIGINS", "").strip()
if _custom_origins:
    ALLOWED_ORIGINS = [o.strip() for o in _custom_origins.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        r"https://zenith506\.github\.io$",
        r"http://localhost(:\d+)?$",
        r"http://127\.0\.0\.1(:\d+)?$",
    ]
CORS(app, origins=ALLOWED_ORIGINS)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))


MODEL_FILE = os.environ.get(
    "DETECT_NOW_MODEL_FILE",
    "detect_now_efficientnetb0_faceswap_crops_v2.keras",
)
THRESHOLD_FILE = os.environ.get(
    "DETECT_NOW_THRESHOLD_FILE",
    "decision_threshold_detect_now_efficientnetb0_faceswap_crops_v2.json",
)

MODEL_PATH = os.path.join(BASE_FOLDER, "model", MODEL_FILE)
THRESHOLD_PATH = os.path.join(BASE_FOLDER, "model", THRESHOLD_FILE)

DEFAULT_FAKE_THRESHOLD = 0.26


BAND_REAL_BELOW = float(os.environ.get("DETECT_NOW_BAND_REAL_BELOW", "0.10"))
BAND_UNCERTAIN_BELOW = float(os.environ.get("DETECT_NOW_BAND_UNCERTAIN_BELOW", "0.25"))
BAND_DEEPFAKE_FROM = float(os.environ.get("DETECT_NOW_BAND_DEEPFAKE_FROM", "0.50"))
if not 0.0 < BAND_REAL_BELOW < BAND_UNCERTAIN_BELOW < BAND_DEEPFAKE_FROM < 1.0:
    raise SystemExit(
        "Verdict bands must satisfy 0 < REAL_BELOW < UNCERTAIN_BELOW < DEEPFAKE_FROM < 1."
    )

VERDICT_TEXT = {
    "LIKELY_REAL": "likely real",
    "UNCERTAIN": "uncertain",
    "LIKELY_DEEPFAKE": "likely deepfake",
    "DEEPFAKE": "deepfake",
}


def verdict_for(fake_probability):
    if fake_probability < BAND_REAL_BELOW:
        return "LIKELY_REAL"
    if fake_probability < BAND_UNCERTAIN_BELOW:
        return "UNCERTAIN"
    if fake_probability < BAND_DEEPFAKE_FROM:
        return "LIKELY_DEEPFAKE"
    return "DEEPFAKE"


MODEL_NAME = "Detect Now EfficientNetB0"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

MAX_IMAGE_PIXELS = 40_000_000

model = None
fake_threshold = None


def load_fake_threshold():
    """Read the decision threshold and make sure it belongs to this model."""
    if not os.path.exists(THRESHOLD_PATH):
        print(
            f"WARNING: threshold file not found ({THRESHOLD_PATH}). "
            f"Using default {DEFAULT_FAKE_THRESHOLD}, which may not suit this model."
        )
        return DEFAULT_FAKE_THRESHOLD

    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
            threshold_data = json.load(file)
        threshold = float(
            threshold_data.get("fake_threshold", DEFAULT_FAKE_THRESHOLD)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print(
            f"WARNING: could not read {THRESHOLD_PATH}. "
            f"Using default {DEFAULT_FAKE_THRESHOLD}."
        )
        return DEFAULT_FAKE_THRESHOLD

    
    owner = threshold_data.get("model_file")
    if owner and owner != MODEL_FILE:
        raise RuntimeError(
            f"Threshold file {THRESHOLD_FILE} belongs to model '{owner}', "
            f"but the server is loading '{MODEL_FILE}'. "
            "Set DETECT_NOW_MODEL_FILE and DETECT_NOW_THRESHOLD_FILE together."
        )

    return float(np.clip(threshold, 0.05, 0.95))


def load_detection_model():
    global model

    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    print("Loading Detect Now EfficientNetB0 model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Detect Now EfficientNetB0 model is ready.")
    return model


def allowed_file(filename):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def read_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()

    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(BytesIO(image_bytes))
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "The image is too large. Please upload a smaller image."
            )

        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
    ) as error:
        raise ValueError("The uploaded file is not a valid image.") from error


def initialise():
    """Load everything at start-up so problems show immediately, not on the
    first user's upload. If a file is missing the server refuses to start."""
    global fake_threshold
    load_detection_model()
    load_yunet_detector()
    fake_threshold = load_fake_threshold()
    print(f"Decision threshold: {fake_threshold}")


initialise()


def predict_image(image_batch):
    detection_model = load_detection_model()
    threshold = fake_threshold if fake_threshold is not None else load_fake_threshold()

    output = detection_model.predict(image_batch, verbose=0)
    fake_probability = float(
        np.clip(np.asarray(output).reshape(-1)[0], 0.0, 1.0)
    )
    real_probability = 1.0 - fake_probability

    if fake_probability >= threshold:
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
        "fake_threshold": threshold,
    }


def create_explanation(verdict, fake_probability):
    return (
        f"Verdict: {VERDICT_TEXT[verdict]}. Deepfake score: {fake_probability * 100:.2f}%. "
        f"Bands: below {BAND_REAL_BELOW * 100:.0f}% likely real, "
        f"{BAND_REAL_BELOW * 100:.0f}% to under {BAND_UNCERTAIN_BELOW * 100:.0f}% uncertain, "
        f"{BAND_UNCERTAIN_BELOW * 100:.0f}% to under {BAND_DEEPFAKE_FROM * 100:.0f}% likely deepfake, "
        f"{BAND_DEEPFAKE_FROM * 100:.0f}% and above deepfake. "
        "This score is not a calibrated probability or forensic proof."
    )


@app.after_request
def allow_private_network(response):
    """Lets an https page (GitHub Pages) call this local server in browsers
    that ask for Private Network Access permission."""
    if request.headers.get("Access-Control-Request-Private-Network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.get("/")
def home():
    return jsonify({
        "application": "Detect Now",
        "status": "ready" if model is not None else "loading",
        "model": MODEL_NAME,
        "architecture": "EfficientNetB0",
        "framework": "TensorFlow",
        "face_detector": "OpenCV YuNet",
        "supported_media": ["image"],
        "message": "Detect Now backend is running.",
    })


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({
        "status": "ready" if model is not None else "loading",
        "model": MODEL_NAME,
        "model_file": os.path.basename(MODEL_PATH),
        "architecture": "EfficientNetB0",
        "framework": "TensorFlow",
        "face_detector": "OpenCV YuNet",
        "face_detector_model": os.path.basename(YUNET_MODEL_PATH),
        "tensorflow_version": tf.__version__,
        "decision_threshold": (
            round(fake_threshold * 100, 2) if fake_threshold is not None else None
        ),
        "verdict_bands": {
            "likely_real_below": round(BAND_REAL_BELOW * 100, 2),
            "uncertain_below": round(BAND_UNCERTAIN_BELOW * 100, 2),
            "deepfake_from": round(BAND_DEEPFAKE_FROM * 100, 2),
        },
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

    filename = uploaded_file.filename or ""

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
            "message": "Upload a JPG, JPEG, PNG, WEBP or BMP image.",
        }), 400

    try:
        log(f"request received: {filename}")
        original_image = read_uploaded_image(uploaded_file)
        log(f"image read: {original_image.width}x{original_image.height}")

        face_count, face_box, detection_method = detect_human_face(original_image)
        log(f"face detection done: {face_count} face(s)")

        
        if face_box is None:
            return jsonify({
                "success": False,
                "error": "No human face was detected.",
                "message": (
                    "Please upload an image containing a visible "
                    "frontal or side-profile human face."
                ),
                "face_detected": False,
                "partial_face_mode": False,
                "detection_method": "none",
                "faces_detected": 0,
            }), 422

        cropped_image = crop_face(original_image, face_box)
        image_batch = preprocess_image(cropped_image)
        log("running model")
        result = predict_image(image_batch)
        log("model done")

        prediction = result["prediction"]
        confidence = result["confidence"]
        real_probability = result["real_probability"]
        fake_probability = result["fake_probability"]

        confidence_percentage = round(confidence * 100, 2)
        real_percentage = round(real_probability * 100, 2)
        fake_percentage = round(fake_probability * 100, 2)

        verdict = verdict_for(fake_probability)
        explanation = create_explanation(verdict, fake_probability)

        timestamp = datetime.now(timezone.utc).isoformat()

        return jsonify({
            "success": True,
            "verdict": verdict,
            "band_real_below": round(BAND_REAL_BELOW * 100, 2),
            "band_uncertain_below": round(BAND_UNCERTAIN_BELOW * 100, 2),
            "band_deepfake_from": round(BAND_DEEPFAKE_FROM * 100, 2),
            "prediction": prediction,
            "result": prediction,
            "classification": prediction,
            "label": prediction,
            "confidence": confidence_percentage,
            "confidence_percentage": confidence_percentage,
            "real_probability": real_percentage,
            "fake_probability": fake_percentage,
            "real_score": real_percentage,
            "fake_score": fake_percentage,
            "fake_threshold": round(result["fake_threshold"] * 100, 2),
            "scores": {
                "real": round(real_probability, 6),
                "fake": round(fake_probability, 6),
            },
            "face_detected": True,
            "partial_face_mode": False,
            "detection_method": detection_method,
            "face_detection_confidence": round(face_box["score"] * 100, 2),
            "faces_detected": int(face_count),
            "face_box": face_box,
            "filename": filename,
            "image_width": original_image.width,
            "image_height": original_image.height,
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "model_name": MODEL_NAME,
            "model_file": os.path.basename(MODEL_PATH),
            "architecture": "EfficientNetB0",
            "framework": "TensorFlow",
            "input_size": "224 x 224",
            "explanation": explanation,
            "summary": explanation,
            "warning": (
                "This is an academic prototype. The result may be "
                "incorrect and should not be treated as forensic proof."
            ),
        }), 200

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error),
            "message": str(error),
        }), 400

    except Exception as error:
        print(f"Prediction error: {error}")
        return jsonify({
            "success": False,
            "error": "The image could not be analysed.",
            "message": str(error),
        }), 500


@app.post("/analyze-video")
@app.post("/api/analyze-video")
def analyze_video():
    return jsonify({
        "success": False,
        "error": "Video detection is not available yet.",
        "message": (
            "Video detection will be added in the next development stage."
        ),
    }), 501


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "The uploaded image is too large.",
        "message": "Upload an image smaller than 10 MB.",
    }), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)