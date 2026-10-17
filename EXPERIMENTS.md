# Experiment reproduction guide

## Run identity

| Material | Role |
|---|---|
| `notebooks/Detect_now.ipynb` | Latest final v2 run: preparation, compatibility repair, training logs and threshold exploration |
| `notebooks/Untitled0.ipynb` | Earlier DFDC Part 34 experiment, approximately 21 days before final makeover according to the team |
| `tools/make_paired_swaps.py` | Extracted from latest notebook cell 18 |
| `tools/prepare_data_colab.py` | Extracted from cell 23, originally named prepare_data.py |
| `tools/train_colab.py` | Extracted from cell 25, originally named train.py |
| `training/legacy/` | Earlier local preparation and retraining scripts; their presence does not establish execution or deployment |

The earlier DFDC notebook downloads `greatgamedota/dfdc-part-34`, version 1.
It reports accuracy 76.67% and AUC 0.8709 with different data and direct resizing.
It is separate development evidence, not a controlled baseline for deployed v2.

## Final v2 environment and parameters

- Saved successful training environment: TensorFlow 2.20.0, Keras 3.13.2, Colab Tesla T4 GPU.
- Deployment environment: root requirements.txt (TensorFlow 2.21.0); distinguish training from deployment.
- Input: YuNet, 35% crop margin, square padding, Lanczos resize to 224 x 224; float32 pixels 0–255.
- Seed 42; batch 32; Adam learning rate 2e-5; 12 recorded epochs.
- Last 100 backbone layers eligible for training, except frozen batch-normalisation layers.
- Augmentation: horizontal flips, JPEG quality 45–95, brightness and contrast changes.
- Checkpoint: maximum validation AUC; best displayed value 0.9355 at epoch 10.
- Training: 2,748 images, comprising 699 real + 695 fake Kaggle crops and 677 Inswapper pairs.
- Validation: 602 images, comprising 150 real + 150 fake Kaggle crops and 151 Inswapper pairs.
- Inswapper test: 161 real + 161 fake images.
- Final binary threshold 0.40; website display bands 0.10, 0.25 and 0.50 are separate.

The automatic validation threshold was 0.72. Later notebook cells show validation AND
test metrics across thresholds before saving 0.40. Although the saved JSON note says
validation-based, this sequence includes test inspection. Final-threshold results are
exploratory and require confirmation on a new untouched test set. The original training
script's one-time-test description does not describe the later notebook exploration.

## Inputs and training execution

Open Detect_now.ipynb in Colab, choose a GPU runtime, mount your own Drive and restore
the input ZIPs and original face-crop checkpoint referenced in the cells. Original
Kaggle and real-face provenance is still unconfirmed; no training images are included.
Inspect the raw folder constants in prepare_data_colab.py before running it.

The working directory is `/content/work`. It needs the YuNet ONNX file, original
`detect_now_efficientnetb0_faceswap_crops.keras`, the raw data folders and pair manifest
referenced in the notebook. Preparation writes:

```text
/content/work/data_ready/
  train/real/  train/fake/
  val/real/    val/fake/
  test/real/   test/fake/
```

Cell 22 is Windows PowerShell preparation, not Colab Python. Earlier error outputs are
retained. Cell 28 records a Keras configuration repair before the successful run;
the extracted cell-25 training script requires that repair when loading the affected
checkpoint. Do not assume the archived notebook runs top to bottom in a clean runtime
without input preparation and compatibility fixes. Use its recorded installation and
archive extraction steps; inference requirements alone do not install InsightFace.

After restoring the raw layout and applying the recorded checkpoint repair, run in Colab:

```bash
python tools/prepare_data_colab.py
python tools/train_colab.py
```

These scripts retain their original /content/work and Google Drive paths. Copy the
resulting model and matching JSON together into model/ when reproducing locally.
Keep the provided final configuration separate from newly generated experiments.

For website setup, follow README.md and root requirements.txt. Legacy local scripts
have machine-specific Windows paths and batch size 16; they do not define final v2.

## Recorded results

| File in results/ | Meaning |
|---|---|
| test_scores.csv | Original-model Kaggle scores, 947 real + 947 fake |
| test_v2_scores.csv | Final v2 scores on the same named Kaggle inputs |
| answer_key.csv | Shared-test map: 50 real, 50 face swaps, 39 AI edits |
| results_detectnow_old.csv | Original shared-test outputs |
| results_detectnow_v2.csv | v2 shared-test outputs |
| own_scores.csv | Separate older record; 41/50 real correct at 0.45 versus 40/50 in shared-test results |
| val_scores.csv | 2,000 validation rows; checkpoint/run mapping unconfirmed, not assigned to final 602-image validation |

Recalculate the main recorded results without dependencies or model inference:

```bash
python tools/verify_recorded_results.py
```

To evaluate new authorised image inputs with the existing inference scripts:

```powershell
$env:DETECT_NOW_MODEL_FILE = "detect_now_efficientnetb0_faceswap_crops_v2.keras"
python evaluate_threshold.py data/test/real data/test/fake --thresholds 0.40 --csv new_v2_scores.csv
python score_shared_test.py data/shared v2_shared.csv --threshold 0.40
```

Replace example data paths with your actual inputs. Preserve split membership, model
identity, versions and face-rejection records. Use --already-cropped only for the
matching classifier-only evaluation mode. CSV recalculation is verification of saved
outputs, not a fresh independent model evaluation.
