# Detect Now

Detect Now is an academic deepfake-detection prototype developed by Group 20 at Charles Darwin University.

The tool allows users to upload a facial image and receive a prediction showing whether the image is likely to be Real or Deepfake.

## Live Website

Frontend:

https://zenith506.github.io/Detect-Now/

The GitHub Pages link hosts the frontend only. Image detection requires the Flask backend to run locally.

## Current Features

- Image upload
- JPG, JPEG and PNG support
- Image preview
- Human-face validation
- Real or Deepfake prediction
- Confidence percentage
- Real and Deepfake probability values
- Downloadable detection report
- SHA-256 file checksum
- Upload history
- Clear-history option
- About Us page
- Responsive website design

Video detection is planned for the next development stage.

## Detection Workflow

```text
User Upload
    ↓
File Validation
    ↓
Human-Face Validation
    ↓
Image Resizing to 224 × 224
    ↓
EfficientNetB0 Model
    ↓
Real or Deepfake Prediction
    ↓
Result and Detection Report