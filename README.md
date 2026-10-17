# Detect Now

Detect Now is an academic facial deepfake-detection prototype developed by Group 20 at Charles Darwin University.

The application uses YuNet face detection and an EfficientNetB0 classifier to analyse facial images and display one of four verdicts: likely real, uncertain, likely deepfake or deepfake.

**Results can be incorrect. Classifier scores are not calibrated probabilities or forensic proof.**

## Live website

- Frontend: https://zenith506.github.io/Detect-Now/
- Backend: https://detect-now-backend.onrender.com
- Health endpoint: https://detect-now-backend.onrender.com/health

The frontend is hosted on GitHub Pages and the Flask backend is deployed on Render.

Pages opened through localhost or 127.0.0.1 use the local backend. The GitHub Pages website uses the hosted backend.

Hosted uploads leave the user's device. The application processes images in memory without an explicit server-side image-save step. This does not establish the hosting provider's logging or retention behaviour.

## Online deployment limitation

Hosted image inference remains unresolved.

During the latest test with `real_999_swap.png`, Render recorded:

- Request received.
- Image decoded at 256 × 256 pixels.
- One face detected.
- “Running model”.

The website remained on “Analysing” without returning a result. No completion or further progress appeared after repeated log refreshes.

A local test of the same named image completed successfully:

- Verdict: **DEEPFAKE**
- Deepfake score: **70.54%**
- Real score: **29.46%**
- Terminal: **“model done”**
- Response: **HTTP 200**

The team reported local completion within approximately five seconds. This was an informal observation, not a measured latency benchmark.

The hosted health endpoint returned ready. Earlier requests with a missing upload or an invalid image returned HTTP 400, confirming basic request validation.

The issue persisted after a Render compute-plan upgrade. The supplied metrics capture shows a 2 GB memory limit but does not establish memory exhaustion. CPU usage was not visible in that capture.

**The cause remains unconfirmed. No file-size failure boundary has been established.** Matching model hashes and complete environment traces were not captured for the local and hosted comparison.

## Features

- JPG, JPEG, PNG, WEBP and BMP image uploads.
- Configured request limit of 10 MiB.
- Image preview and detected-face outline.
- Human-face validation using OpenCV YuNet.
- One face selected using bounding-box area and detector confidence.
- Four verdict bands based on the deepfake score.
- Downloadable report with a SHA-256 checksum.
- Browser-local history and CSV export.

Video detection is not implemented.

## Detection workflow

1. Validate the uploaded file.
2. Detect a human face using YuNet.
3. Crop the face with a 35% margin.
4. Pad the crop to a square.
5. Resize to 224 × 224 pixels using Lanczos interpolation.
6. Run the EfficientNetB0 classifier.
7. Display the score and corresponding verdict.

The classifier output is a raw score, not a calibrated probability.

## Run locally

### Install dependencies and start the backend

From the project folder, using Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend.py
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The backend starts at:

http://127.0.0.1:5000

### Start the frontend

In a second terminal, from the same project folder:

```powershell
python -m http.server 5500
```

Open:

http://127.0.0.1:5500/

Check the backend configuration at:

http://127.0.0.1:5000/health

### Required model files

These files must be present in `model/`:

| File | Purpose |
|---|---|
| `detect_now_efficientnetb0_faceswap_crops_v2.keras` | Default v2 classifier |
| `decision_threshold_detect_now_efficientnetb0_faceswap_crops_v2.json` | Matching binary threshold configuration |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector |

The backend refuses to start if required model files or matching threshold settings cannot be loaded.

## Render configuration

Start command:

```bash
gunicorn backend:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180
```

For additional frontend origins, configure `DETECT_NOW_CORS_ORIGINS` as a comma-separated list of permitted origins.

The 180-second Gunicorn timeout is a configuration setting. It does not establish the cause of the observed inference hang.

## Model and training

### Final v2 model

- Architecture: EfficientNetB0.
- Framework: TensorFlow/Keras.
- Face detector: OpenCV YuNet.
- Input size: 224 × 224 pixels.
- Training environment: Google Colab with a Tesla T4 GPU.
- Recorded training versions: TensorFlow 2.20.0 and Keras 3.13.2.
- Deployment TensorFlow version: 2.21.0.
- Random seed: 42.
- Batch size: 32.
- Optimiser: Adam.
- Initial learning rate: 0.00002.
- Recorded epochs: 12.
- Fine-tuning: last 100 backbone layers eligible for training, with batch-normalisation layers frozen.
- Augmentation: horizontal flips, JPEG quality changes, brightness and contrast changes.
- Best displayed validation AUC: 0.9355 at epoch 10.

### Data preparation counts

| Split | Kaggle real | Kaggle fake | Inswapper real | Inswapper fake | Total |
|---|---:|---:|---:|---:|---:|
| Training | 699 | 695 | 677 | 677 | 2,748 |
| Validation | 150 | 150 | 151 | 151 | 602 |
| Inswapper test | — | — | 161 | 161 | 322 |

Inswapper pairs were generated using InsightFace `inswapper_128`. Donor faces remained within the same split. Pairs touching the known shared-test image list were excluded during preparation.

### Notebook roles

| File | Role |
|---|---|
| `notebooks/Detect_now.ipynb` | Latest v2 preparation, fine-tuning, training logs and threshold exploration |
| `notebooks/Untitled0.ipynb` | Earlier DFDC Part 34 experiment, completed approximately 21 days before the final makeover according to the team |

The earlier DFDC experiment used different data and preprocessing. Its saved accuracy of 76.67% and AUC of 0.8709 are separate development results, not the deployed v2 model's performance.

### Training scripts

| File | Purpose |
|---|---|
| `tools/make_paired_swaps.py` | Generate paired real and face-swapped images |
| `tools/prepare_data_colab.py` | Prepare YuNet face crops |
| `tools/train_colab.py` | Fine-tune the v2 model |
| `training/legacy/` | Earlier local preparation and retraining scripts |

The latest notebook records a Keras configuration compatibility repair before successful training. The extracted training scripts retain their original Colab paths and require the referenced input data.

Detailed training setup, input requirements and execution steps are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Threshold and verdict bands

The v2 binary evaluation threshold is **0.40**.

The website uses separate display bands:

| Deepfake score | Displayed verdict |
|---|---|
| Below 10% | Likely real |
| 10% to below 25% | Uncertain |
| 25% to below 50% | Likely deepfake |
| 50% and above | Deepfake |

The API also returns a binary `prediction` using the model's threshold configuration.

Display boundaries can be configured using:

- `DETECT_NOW_BAND_REAL_BELOW`
- `DETECT_NOW_BAND_UNCERTAIN_BELOW`
- `DETECT_NOW_BAND_DEEPFAKE_FROM`

Their default fractional values are `0.10`, `0.25` and `0.50`.

The display bands were selected after inspecting the 77-image website test and have not been independently validated on a new test set.

The training script initially selected a threshold of 0.72 using validation data. Later notebook cells inspected validation and test metrics across thresholds before saving 0.40. The final-threshold evaluation is therefore exploratory rather than a fully blind assessment.

## Recorded evaluation results

These results use binary thresholds of 0.45 for the original model and 0.40 for v2, rather than the four website display bands.

“Real correct” means the proportion of real images classified as real. “Swaps caught” means the proportion of face-swapped images classified as fake.

| Test set | Measure | Original model | v2 model |
|---|---|---:|---:|
| Kaggle: 947 real + 947 swaps | Real correct / swaps caught | 89.4% / 92.9% | 88.0% / 78.2% |
| Kaggle | Accuracy / AUC | 91.2% / 0.973 | 83.1% / 0.926 |
| Inswapper: 322 images | Real correct / swaps caught | 78.3% / 41.0% | About 94% / 72% |
| Inswapper | Accuracy / AUC | 59.6% / 0.652 | About 83% / 0.934 |
| Team: 50 real + 50 swaps | Real correct / swaps caught | 80% / 34% | 92% / 64% |
| Team: 50 real + 50 swaps | Accuracy | 57% | 78% |
| Team: 39 AI expression edits | Edits detected | 9/39 | 2/39 |

V2 improved performance on the Inswapper and team face-swap sets but reduced performance on the Kaggle test set. Neither model demonstrated reliable detection of AI expression edits.

The Inswapper test uses the same swap generator included in v2 training. It does not establish generalisation to unseen swap tools.

Some Kaggle face-donor identities overlap between training and testing, which may make those results optimistic.

## Result files

| File | Contents |
|---|---|
| `results/test_scores.csv` | Original-model Kaggle predictions |
| `results/test_v2_scores.csv` | V2 Kaggle predictions |
| `results/answer_key.csv` | Labels and filename mapping for the 139-image shared test |
| `results/results_detectnow_old.csv` | Original-model shared-test outputs |
| `results/results_detectnow_v2.csv` | V2 shared-test outputs |
| `results/own_scores.csv` | Separate earlier local evaluation record |
| `results/val_scores.csv` | Validation scores with unconfirmed checkpoint/run mapping |

`own_scores.csv` gives 41/50 real images correct at threshold 0.45, compared with 40/50 in the shared-test results. These records are kept separate.

The 2,000 rows in `val_scores.csv` are not assigned to the final v2 validation split of 602 images.

### Verify saved results

From the project folder:

```powershell
python tools/verify_recorded_results.py
```

This recalculates metrics from the recorded CSVs. It does not rerun training or model inference.

### Evaluate new images

After providing authorised labelled image folders:

```powershell
$env:DETECT_NOW_MODEL_FILE = "detect_now_efficientnetb0_faceswap_crops_v2.keras"
python evaluate_threshold.py data/test/real data/test/fake --thresholds 0.40 --csv new_v2_scores.csv
python score_shared_test.py data/shared v2_shared.csv --threshold 0.40
```

The example data paths must match the local dataset layout.

Use `--already-cropped` only for the corresponding preprocessed classifier-only evaluation mode.

## Data sources and licences

| Material | Source status | Licence status |
|---|---|---|
| Kaggle face-swap collection used for v2 | Unconfirmed | Unconfirmed |
| Original real-face image library | Unconfirmed | Unconfirmed |
| Inswapper source images | Unconfirmed | Unconfirmed |

No training images are included in the experiment evidence folders.

## Known limitations

- Hosted image inference remains unresolved.
- The cause of the online inference hang is unconfirmed.
- No file-size failure boundary has been established.
- Generalisation to unseen swap generators has not been established.
- AI expression edits are not reliably detected.
- Scores are not calibrated probabilities.
- Only one selected face is classified per image.
- Covered, small, blurred or low-resolution faces may reduce reliability.
- Some source identities overlap across Kaggle splits.
- Threshold and display-band selection involved inspection of evaluation results.
- Recorded tests do not establish production readiness.

## Further development

- Diagnose the hosted inference hang using controlled tests and server instrumentation.
- Evaluate fixed thresholds and display bands on a new untouched test set.
- Test manipulation generators excluded from training.
- Compare alternative architectures using identical data and preprocessing.
- Evaluate compression, image quality, response time and concurrency.
- Extend to video only after image inference is sufficiently dependable.