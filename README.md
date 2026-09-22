# AI-Generated Media & Deepfake Detector

**Status: trained and evaluated on real data.** Every stage below has actually
been run — not just written. See [Results](#results) for real numbers.

> **⚠️ Scope limitation — read before using:** this model was trained only on
> **StyleGAN-generated** faces (see [Dataset](#dataset)). It reliably detects
> that specific generation method and **does not generalize to diffusion-model
> images** (Midjourney, Stable Diffusion, DALL-E, Lensa-style avatars, etc.) —
> tested manually, it misclassifies those as REAL. This is not a bug, it's an
> unseen-distribution gap (see [Limitations](#limitations)). **Do not use this
> tool to make real decisions about whether a specific image is AI-generated**
> — treat it strictly as a portfolio demonstration of the ML workflow, not a
> reliable detector.

## Project Overview

A computer-vision system that takes an uploaded face image and predicts whether
it is **REAL** or **POTENTIALLY MANIPULATED / AI-GENERATED**, with a confidence
score. It's a portfolio project built to demonstrate the full ML workflow:
dataset selection, preprocessing, transfer learning, training, evaluation, error
analysis, and a working demo — not a production moderation tool.

## Motivation

AI-generated and manipulated media (deepfakes, GAN-generated faces) are
increasingly used to mislead people. This project explores, at a scale
appropriate for a student laptop, how far a simple transfer-learning approach
can go at flagging this kind of content — and just as importantly, where it
fails and why.

## Features (implemented)

- Face detection and cropping (OpenCV Haar Cascade)
- Transfer-learning classifier (MobileNetV2, fine-tuned)
- Training pipeline with early stopping, best-checkpoint saving, loss/accuracy plots
- Evaluation: accuracy, precision, recall, F1, confusion matrix on a held-out test set
- Empirically-derived UNCERTAIN confidence band (computed from validation data, not a guessed constant)
- Error analysis notebook (correct/incorrect examples by category)
- Grad-CAM visual explanation (optional, toggled in the UI)
- Streamlit web demo with disclaimer, no permanent storage of uploaded images

**Not implemented** (see [Future Scope](#future-scope)): video detection, any
form of face search/database, authentication, cloud deployment automation.

## Tech Stack

- Python, PyTorch, torchvision, OpenCV
- NumPy, pandas, scikit-learn, matplotlib
- Streamlit
- Jupyter Notebook (dataset prep + error analysis)

## Architecture

```
Dataset (rvf10k valid split, Kaggle -- built from the 140k Real and Fake Faces source)
   -> Face detection & crop (OpenCV Haar Cascade)
   -> Resize + normalize (ImageNet stats)
   -> Augmentation (train only): random crop, flip, color jitter
   -> Train / Val / Test split (70 / 15 / 15)
   -> MobileNetV2 (ImageNet-pretrained, custom binary head)
   -> Evaluation (accuracy, precision, recall, F1, confusion matrix)
   -> Error analysis (correct / false positive / false negative examples)
   -> Streamlit app (upload -> face crop -> prediction -> confidence -> Grad-CAM)
```

## Dataset

**rvf10k** ([Kaggle, sachchitkunichetty](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k))
— itself built from the same source as the larger 140k Real and Fake Faces
dataset (real = FFHQ photographs, fake = StyleGAN-generated), pre-split into
train/valid folders. This project uses **only the `valid` split** (1,500 real +
1,500 fake = 3,000 images), not the full 10k, to keep the dataset small.

After face detection + crop (some images are dropped when no face is detected),
[`notebooks/01_prepare_dataset.ipynb`](notebooks/01_prepare_dataset.ipynb) produced:

| Split | Real | Fake |
|---|---|---|
| train | 989 | 1,041 |
| val | 214 | 222 |
| test | 215 | 223 |

(96 of 3,000 images were dropped — no face detected by the Haar Cascade — a
~3.2% skip rate.)

## Model

**MobileNetV2**, ImageNet-pretrained, with the final classifier layer replaced
by a single-unit linear layer (binary classification via `BCEWithLogitsLoss`,
sigmoid output = P(manipulated)).

Chosen over ResNet/EfficientNet because of its small size (~14MB) and much
faster CPU training/inference — this project trains and runs on CPU only (no
GPU available in this environment), so that tradeoff mattered more than
squeezing out a few extra points of accuracy from a heavier backbone.

## Results

Trained for 7 epochs (Adam, lr 1e-4, batch size 32) on CPU — early stopping
was set up for up to 8 epochs/patience 3, but this run was manually capped at
7 epochs. Best checkpoint (lowest validation loss) was epoch 7 itself.

Test set (438 images, held out, never seen during training):

| Metric | Value |
|---|---|
| Accuracy | 91.1% |
| Precision | 88.0% |
| Recall | 95.5% |
| F1 Score | 91.6% |

Full numbers: [`results/metrics/test_metrics.json`](results/metrics/test_metrics.json).
Training curves: [`results/plots/loss_curve.png`](results/plots/loss_curve.png),
[`accuracy_curve.png`](results/plots/accuracy_curve.png). Confusion matrix:
[`results/plots/confusion_matrix.png`](results/plots/confusion_matrix.png).

**Note on the UNCERTAIN band**: the validation-derived confidence threshold
came out to exactly 0.50 (see [Confidence Score](#9-confidence-score) logic in
`src/evaluate.py`) — meaning this particular model already exceeds 90% accuracy
across its full confidence range, so the UNCERTAIN label will rarely or never
trigger in the app for this checkpoint. That's an honest property of this run,
not a bug — a weaker model would show a more active UNCERTAIN band.

## Error Analysis

Ran [`notebooks/02_error_analysis.ipynb`](notebooks/02_error_analysis.ipynb) on
all 438 test predictions: 186 correct real, 213 correct fake, **29 false
positives**, **10 false negatives**.

- **False positives** (real photos called MANIPULATED) mostly involved
  occlusion or unusual objects near the face — a spoon, a straw, hands,
  patterned headwear, off-center crops with background clutter — several
  flagged with high confidence (0.91–1.00). Likely cause: StyleGAN training
  faces are almost always clean and unobstructed, so the model may have
  partly learned to associate occlusion/clutter with "fake" rather than
  relying purely on photorealism cues — a real dataset-bias limitation.
- **False negatives** (StyleGAN fakes called REAL) were clean, well-lit,
  front-facing portraits with no visible GAN artifacts — StyleGAN's better
  outputs are genuinely hard to distinguish even for this model, which is an
  expected model limitation rather than a preprocessing bug.

See the notebook's Notes section for the full writeup.

## Streamlit Demo

Upload an image -> the app detects and crops the face -> runs it through the
trained model -> shows REAL / POTENTIALLY MANIPULATED / UNCERTAIN with a
confidence percentage, and optionally a Grad-CAM heatmap. Uploaded images are
held in memory only, never written to disk.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then set up your own Kaggle API token (`kaggle.json` at `C:\Users\<you>\.kaggle\kaggle.json`,
generated from your Kaggle account settings) and download the dataset:

```bash
kaggle datasets download -d sachchitkunichetty/rvf10k -p data/raw --unzip
```

This downloads the full rvf10k dataset (10,000 images, ~273MB, since Kaggle
doesn't support downloading a single subfolder), but the project only uses its
`valid` split (3,000 images) — the notebook below reads only from
`data/raw/rvf10k/valid`, and you can safely delete `data/raw/rvf10k/train`
(7,000 images) afterward to save disk space.

## Usage

```bash
# 1. Build the 3,000-image face-cropped subset from data/raw/rvf10k/valid
jupyter notebook notebooks/01_prepare_dataset.ipynb

# 2. Train (this run used --epochs 8 --patience 3, manually stopped after epoch 7)
python -m src.train --data-dir data/processed --epochs 8 --patience 3 --batch-size 32

# 3. Evaluate on the test set
python -m src.evaluate --data-dir data/processed

# 4. Inspect errors
jupyter notebook notebooks/02_error_analysis.ipynb

# 5. Run the demo
streamlit run app.py
```

## Limitations

- **Confirmed by manual testing, not just theory**: this model does not detect
  diffusion-model-generated images (Midjourney, Stable Diffusion, Lensa-style
  avatars) — it classifies them as REAL. It was trained exclusively on
  StyleGAN-generated faces vs. FFHQ photos, and that's the only generation
  method it can reliably flag. This is a hard, known-unsolved problem industry
  -wide (generalizing across generator families), not something specific to
  this implementation — even large-scale industrial detectors struggle with
  new/unseen generators.
- Trained on a small (~2,900-image, after face-detection drops) subset of one
  dataset — will likely generalize poorly to other generation methods
  (diffusion models, face-swap deepfakes, video compression artifacts) it
  never saw during training.
- **Not suitable for real decisions about real images** (e.g. determining
  whether a specific photo used against someone is AI-generated). This is a
  portfolio project demonstrating the ML workflow, not a safety or forensic
  tool — a wrong call here could cause real harm if relied upon.
- Error analysis (above) found the model may be partly relying on framing and
  occlusion as a real/fake signal rather than purely photorealism cues, since
  training fakes were almost always clean and unobstructed — a real,
  observed dataset-bias limitation, not a hypothetical one.
- Haar Cascade face detection misses faces at extreme angles or poor lighting;
  any image where no face is detected is rejected rather than classified
  (happened for ~3.2% of this dataset).
- Confidence score is a model probability, not proof — see the in-app disclaimer.
- The UNCERTAIN band is effectively inactive for this checkpoint (see Results)
  since the model already exceeds 90% accuracy at every confidence level.
- Evaluated only on held-out data from the *same* source distribution as
  training data; real-world social-media images (different compression,
  resolution, editing) were not tested.

## Future Scope

- Video deepfake detection (frame extraction + aggregation)
- Broader, more diverse training data across multiple generation techniques
- Stronger explainability beyond Grad-CAM
- Privacy-preserving content protection mechanisms
- Better temporal modeling for video

These are explicitly **not implemented** — listed only as possible directions.
