from io import BytesIO

import cv2
import numpy as np
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


MODEL_NAME = (
    "prithivMLmods/deepfake-detector-model-v1"
)

MAXIMUM_FILE_SIZE = 8 * 1024 * 1024

ALLOWED_FORMATS = {
    "JPEG",
    "PNG"
}


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAXIMUM_FILE_SIZE
)


CORS(
    app,
    resources={
        r"/predict": {
            "origins": [
                "http://127.0.0.1:5500",
                "http://localhost:5500",
                "https://zenith506.github.io"
            ]
        }
    }
)


processor = None
model = None


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


def load_detection_model():
    global processor
    global model

    print(
        "Loading the deepfake detection model..."
    )

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForImageClassification.from_pretrained(
            MODEL_NAME
        )
    )

    model.eval()

    print(
        "The deepfake detection model is ready."
    )


def find_largest_face(image):
    image_array = np.array(image)

    grey_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_detector.detectMultiScale(
        grey_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda face:
        face[2] * face[3]
    )

    x, y, width, height = largest_face

    horizontal_margin = int(
        width * 0.25
    )

    vertical_margin = int(
        height * 0.25
    )

    left = max(
        0,
        x - horizontal_margin
    )

    top = max(
        0,
        y - vertical_margin
    )

    right = min(
        image.width,
        x + width + horizontal_margin
    )

    bottom = min(
        image.height,
        y + height + vertical_margin
    )

    return image.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ready",
        "model": MODEL_NAME,
        "face_detection": "enabled"
    })


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "Please select an image."
        }), 400

    try:
        file_bytes = uploaded_file.read()

        if not file_bytes:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        image = Image.open(
            BytesIO(file_bytes)
        )

        if image.format not in ALLOWED_FORMATS:
            return jsonify({
                "error": (
                    "Only JPG, JPEG and PNG "
                    "images are accepted."
                )
            }), 400

        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert("RGB")

        detected_face = find_largest_face(
            image
        )

        if detected_face is None:
            return jsonify({
                "error": (
                    "No human face was detected. "
                    "Please upload a clear facial image."
                ),
                "code": "NO_FACE"
            }), 422

        inputs = processor(
            images=detected_face,
            return_tensors="pt"
        )

        with torch.inference_mode():
            output = model(**inputs)

            probabilities = torch.softmax(
                output.logits,
                dim=1
            )[0]

        fake_probability = (
            probabilities[0].item()
        )

        real_probability = (
            probabilities[1].item()
        )

        confidence_difference = abs(
            fake_probability
            - real_probability
        )

        if confidence_difference < 0.20:
            prediction = "Uncertain"

            confidence = max(
                fake_probability,
                real_probability
            )

        elif (
            fake_probability
            >= real_probability
        ):
            prediction = (
                "Likely Deepfake"
            )

            confidence = fake_probability

        else:
            prediction = "Likely Real"

            confidence = real_probability

        return jsonify({
            "prediction": prediction,

            "confidence": round(
                confidence * 100,
                2
            ),

            "fake_probability": round(
                fake_probability * 100,
                2
            ),

            "real_probability": round(
                real_probability * 100,
                2
            ),

            "filename": (
                uploaded_file.filename
            ),

            "face_detected": True,

            "model": MODEL_NAME,

            "warning": (
                "This prediction is experimental "
                "and should not be treated as "
                "forensic proof."
            )
        })

    except UnidentifiedImageError:
        return jsonify({
            "error": (
                "The selected file is not "
                "a valid image."
            )
        }), 400

    except Exception as error:
        print(
            f"Prediction error: {error}"
        )

        return jsonify({
            "error": (
                "The image could not be analysed."
            )
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": (
            "The image is larger than 8 MB."
        )
    }), 413


if __name__ == "__main__":
    load_detection_model()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )