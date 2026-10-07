# Detect Now

Detect Now is an academic deepfake-detection prototype developed by Group 20 at
Charles Darwin University. Upload a facial image and receive a verdict: **likely real**,
**uncertain**, **likely deepfake** or **deepfake** (face swap).

**It is a prototype. Results can be wrong and are not forensic proof.**

## Live website

Frontend: <https://zenith506.github.io/Detect-Now/>

Backend API: <https://detect-now-backend.onrender.com> (hosted on Render).

When the page is opened from any address other than `localhost` or `127.0.0.1`, it sends the image
to the hosted backend, so uploaded images do travel to that server for analysis (they are analysed
in memory and not saved by the server). When the page is opened from `localhost`, it uses a backend
running on your own computer instead (`python backend.py`, at `http://127.0.0.1:5000`).

On the hosted backend, set the start command to
`gunicorn backend:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT` and, if the
frontend is served from an address other than GitHub Pages or localhost, set
`DETECT_NOW_CORS_ORIGINS` to a comma-separated list of the allowed addresses.

## Features

- Image upload (JPG, JPEG, PNG, WEBP, BMP, up to 10 MB), preview and face box overlay
- Human-face validation (YuNet); only the largest face in a photo is analysed
- Four-band verdict (likely real / uncertain / likely deepfake / deepfake) from the deepfake score
- Downloadable report with SHA-256 checksum
- Upload history (stored only in your browser) and CSV export
- Video detection: planned, not available yet

## Detection workflow

```
Upload -> file checks -> YuNet face detection -> crop with 35% margin
      -> pad to square -> resize to 224 x 224 -> EfficientNetB0 -> score
      -> score < 10% ? LIKELY REAL : < 25% ? UNCERTAIN : < 50% ? LIKELY DEEPFAKE : DEEPFAKE
```

The scores are raw classifier outputs, **not calibrated probabilities**.

## Run it locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python backend.py                # starts http://127.0.0.1:5000
```

Then open `index.html` through a local server or the GitHub Pages site.

The server **refuses to start** if the model, the face detector file or its threshold
file cannot be loaded, so a problem shows up immediately.

Files required in `model/`:

| File | Purpose |
|---|---|
| `detect_now_efficientnetb0_faceswap_crops_v2.keras` | classifier (default, v2) |
| `decision_threshold_detect_now_efficientnetb0_faceswap_crops_v2.json` | its decision threshold (0.40) |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (OpenCV Zoo) |

The original model (`detect_now_efficientnetb0_faceswap_crops.keras`, threshold file
`decision_threshold_faceswap_crops.json`) can still be used. Set both before starting the server
(file names inside `model/`):

```
$env:DETECT_NOW_MODEL_FILE = "detect_now_efficientnetb0_faceswap_crops.keras"
$env:DETECT_NOW_THRESHOLD_FILE = "decision_threshold_faceswap_crops.json"
python backend.py
```

The threshold file records which model it belongs to, and the server stops if they do not match.
Other websites can be allowed with `DETECT_NOW_CORS_ORIGINS`.

## Training and evaluation

How the v2 model was made (scripts are in `tools/`; the two Colab ones were run on a free Colab GPU):

1. `tools/make_paired_swaps.py` - for each real image, makes a fake with a different person's
   face swapped in (InsightFace inswapper). Real and fake go through identical processing, and donor
   faces stay inside the same split.
2. `tools/prepare_data_colab.py` - builds 224 x 224 face crops from the Kaggle subset and the
   inswapper pairs using the same steps as the backend (YuNet, 35% margin, pad, resize). Pairs that
   touch the team test set are dropped.
3. `tools/train_colab.py` - fine-tunes the original model with augmentation (JPEG quality, flips,
   brightness, contrast), reports before/after AUC per source, and picks a threshold on validation.
   Note: a model saved by a newer Keras may need its config cleaned of unknown settings before an
   older Keras (such as Colab's) can load it; the weights are unchanged.
4. In the project folder (they import `preprocessing.py`): `evaluate_threshold.py` scores folders of
   real and fake images at many thresholds and prints AUC; `score_shared_test.py` and
   `score_results.py` produce the team test-set table.

### Verdict bands

The website shows one of four verdicts, based only on the deepfake score:

| Deepfake score | Verdict |
|---|---|
| below 10% | likely real |
| 10% to under 25% | uncertain |
| 25% to under 50% | likely deepfake |
| 50% and above | deepfake |

The cut-offs can be changed with `DETECT_NOW_BAND_REAL_BELOW`, `DETECT_NOW_BAND_UNCERTAIN_BELOW` and
`DETECT_NOW_BAND_DEEPFAKE_FROM` (fractions, default `0.10`, `0.25` and `0.50`). The API also still returns the
older two-way `prediction`, made with the single decision threshold in the threshold file (0.40 for v2); the
shared-test scoring scripts use that.

The cut-offs were chosen after looking at the 77-image live test, so they have not yet been confirmed on new
images. What each band contained:

| Band | Kaggle test: real | Kaggle test: swaps | Live test: real | Live test: edited | Live test: swaps |
|---|---|---|---|---|---|
| below 10% | 725 | 101 | 22 | 20 | 0 |
| 10% to under 25% | 74 | 65 | 2 | 3 | 5 |
| 25% to under 50% | 47 | 71 | 2 | 2 | 8 |
| 50% and above | 101 | 710 | 0 | 0 | 13 |

Compared with the earlier bands (15% and 50%), on the Kaggle test set the share of real photos given a
likely deepfake or deepfake label rose from 10.7% to 15.6%, and the share of swaps given one rose from 75.0%
to 82.5%. About 1 in 8 of the images in the top band on the Kaggle set were real (101 of 811), so
"deepfake" is a strong label for a score that is not a calibrated probability.

### Results

The results tables below use a single threshold each (0.45 for the original model, 0.40 for v2),
not the four bands.

Two models were compared on the same held-out test images. "Real correct" is the share of real
images called real; "swaps caught" is the share of face swaps called fake.

| Test set | Measure | Original model (threshold 0.45) | v2 model (threshold 0.40) |
|---|---|---|---|
| Kaggle test (947 real + 947 swaps) | real correct / swaps caught | 89.4% / 92.9% | 88.0% / 78.2% |
| | accuracy / AUC | 91.2% / 0.973 | 83.1% / 0.926 |
| Inswapper test (322 images) | real correct / swaps caught | 78.3% / 41.0% | 94% / 72% |
| | accuracy / AUC | 59.6% / 0.652 | about 83% / 0.934 |
| Team test set (50 real + 50 swaps) | real correct / swaps caught | 80% / 34% | 92% / 64% |
| Team test set (39 AI expression edits) | edits caught | 23% | 5% |

- **Original model:** best on the Kaggle swaps it was trained on, but it did not generalise to
  swaps made with a different tool (inswapper).
- **v2 model:** fine-tuned on about 1,400 Kaggle frames plus 677 inswapper pairs. Much better on
  inswapper swaps, but it lost some accuracy on the Kaggle test set.
- Neither model detects expression edits made by AI tools (both are near the false-alarm rate).
- The inswapper test images come from the same swap tool used to train v2, so they do **not**
  show how well v2 handles swap tools it has never seen. That has not been measured.
- Thresholds were chosen by hand from validation results. Test sets are small (50 to 322 images
  per class), so small differences are within noise.
- The Kaggle splits do not share target videos, but some face-donor IDs in the test set also appear
  in train, so the Kaggle figures may be somewhat optimistic. Pairs touching the team test set were
  removed from training.

**Data and licences:** _(fill in: Kaggle dataset name, link and licence; the Real faces library
used for the inswapper pairs; how the swaps were generated)_

## Known limitations

- Face swaps from tools other than the two in the training data may be missed.
- AI expression edits and other non-swap manipulations are not detected.
- On the Kaggle test set, 15.6% of real photos got a likely deepfake or deepfake verdict (10.7% got deepfake).
- Heavy compression, blur, low resolution, covered or very small faces reduce reliability.
- Faces are judged one at a time (largest face only).
- The models have not been tested on images recompressed by social media.