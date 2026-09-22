# Interview Preparation

Answers reflect exactly what this project implements. Wherever a real number
(accuracy, dataset size, epoch count) is asked for, fill it in from your own
`results/metrics/test_metrics.json` and `results/metrics/training_history.json`
after you actually train — do not memorize a number that isn't yours.

### 1. Why did you choose deepfake detection?
It's a real, current problem (AI-generated media misleading people) that maps
cleanly onto a classic, explainable CV task: binary image classification. It
let me practice the full ML lifecycle without needing exotic infrastructure.
*Follow-up: "What's a case where this problem actually caused harm?" — be ready with one real example (e.g. a fabricated image/video used for fraud or disinformation) — don't overstate your project as a solution to it.*

### 2. What problem does your project solve?
It gives a first-pass signal on whether a single uploaded face image looks
AI-generated/manipulated, with a confidence score — a decision aid, not an
automated verdict.
*Follow-up: "Would you trust this to auto-remove content?" — no; explain why (false positives/negatives, no proof, see Limitations).*

### 3. What is a deepfake?
Media (image or video) where a person's face or likeness has been synthetically
generated or swapped, typically using GANs or diffusion models, to make it look
like something that didn't happen.
*Follow-up: "Is a GAN-generated face from scratch a 'deepfake'?" — technically it's "AI-generated", a related but distinct category from "face-swapped" deepfakes; explain your dataset covers the former (StyleGAN-generated faces), not the latter.*

### 4. What dataset did you use?
140k Real and Fake Faces (Kaggle, `xhlulu`) — real photos from FFHQ, fake faces
from StyleGAN. Used a 4,000-image subset (2,000/2,000) sampled and face-cropped
myself, not the full 140k.
*Follow-up: "Why not the full 140k?" — CPU-only laptop training time; explain the subset was randomly sampled with a fixed seed, not cherry-picked.*

### 5. Why did you choose that dataset?
It's public, already labeled real/fake, images are pre-aligned faces (no video
decoding needed), and it's small enough to iterate on quickly while still being
a genuine AI-generation detection task (not just Photoshop edits).
*Follow-up: "What's a weakness of this dataset choice?" — it only covers one generation method (StyleGAN); a model trained on it won't necessarily catch other generators.*

### 6. How did you preprocess the images?
Face detection + crop (OpenCV Haar Cascade) -> resize -> normalize with
ImageNet mean/std -> (training only) random crop/flip/color-jitter augmentation
-> tensor conversion.
*Follow-up: "Why ImageNet normalization specifically?" — because the backbone (MobileNetV2) was pretrained on ImageNet with that normalization; mismatching it would degrade the pretrained features.*

### 7. Why did you detect/crop faces?
The label (real/fake) is about the *face*, not the background — cropping
removes irrelevant pixels the model could otherwise latch onto (dataset bias),
and matches how a real deployment would receive images (arbitrary photos, not
pre-cropped faces).
*Follow-up: "What happens if no face is detected?" — the image is rejected with an explicit error rather than guessed on.*

### 8. Why did you resize images?
The pretrained backbone requires a fixed input size (224x224 for MobileNetV2);
resizing also keeps batches uniform in shape and keeps computation bounded.
*Follow-up: "Doesn't resizing distort faces?" — yes, slightly, for non-square crops; that's a real, acknowledged tradeoff versus more complex aspect-ratio-preserving padding.*

### 9. Why normalization?
Neural nets trained with gradient descent converge faster and more stably when
inputs are on a consistent, small-magnitude scale — and matching the exact
ImageNet mean/std the backbone was pretrained with keeps its pretrained
features meaningful.
*Follow-up: "What if you normalized with your own dataset's mean/std instead?" — possible, but would still need consistency between train and inference, and would slightly mismatch the pretrained backbone's expected input distribution.*

### 10. Why augmentation?
It teaches the model to ignore small, irrelevant variations (pose, lighting,
framing) instead of memorizing exact pixels, which reduces overfitting on a
small (4,000-image) dataset.
*Follow-up: "Why no augmentation on val/test?" — those must reflect real, unmodified performance; augmenting them would make evaluation non-deterministic and not representative of real inference.*

### 11. Why must inference use the same preprocessing as training?
The model only learned to interpret inputs shaped and scaled exactly like its
training data; any mismatch (different resize, missing normalization, no face
crop) shifts the input distribution and silently degrades predictions. `src/preprocessing.py` is shared by both the data-prep notebook and `predict.py` for exactly this reason.
*Follow-up: "How did you guarantee this in code?" — point to the shared module, not duplicated logic.*

### 12. Why transfer learning?
4,000 images is far too little to train a CNN from scratch well; starting from
ImageNet-pretrained weights means the model already knows general visual
features (edges, textures, shapes) and only needs to learn to repurpose them
for this task.
*Follow-up: "Did you freeze any layers?" — be honest about what `src/model.py` actually does (fine-tunes the whole network, not just the head) — check the code before answering.*

### 13. Why MobileNetV2 over ResNet/EfficientNet?
Much smaller (~14MB vs ~100MB+) and faster on CPU — this project has no GPU
available, so training and inference speed mattered more than the last couple
points of accuracy a bigger backbone might offer.
*Follow-up: "What would you pick with a GPU and more data?" — probably EfficientNet or a ResNet variant, trading training time for accuracy headroom.*

### 14. How does your model classify an image?
Face crop -> preprocessed tensor -> MobileNetV2 -> single logit -> sigmoid ->
probability of "manipulated" -> thresholded at 0.5 (below: REAL, above:
MANIPULATED), with an UNCERTAIN band near the boundary based on a
validation-derived confidence threshold.
*Follow-up: "Why 0.5 as the decision boundary?" — standard default for a calibrated binary sigmoid; explain the UNCERTAIN band exists precisely because 0.5 is a hard cutoff on what's really a continuous confidence.*

### 15. What loss function did you use?
`BCEWithLogitsLoss` (binary cross-entropy on raw logits, numerically stable
combination of sigmoid + BCE in one op).
*Follow-up: "Why not 2-class CrossEntropyLoss?" — this is a strictly binary problem; single-logit BCE gives a direct P(manipulated) without softmax's redundant second output.*

### 16. What optimizer did you use?
Adam, learning rate 1e-4 (a small LR since the backbone is pretrained — large
updates would destroy the useful pretrained weights).
*Follow-up: "Did you use a learning rate scheduler?" — answer honestly from `src/train.py` (currently: no, fixed LR) — mention it as a possible improvement if asked.*

### 17. How did you prevent overfitting?
Data augmentation, transfer learning (starting from good weights instead of
random ones), and early stopping on validation loss (training halts if val
loss doesn't improve for a set number of epochs, and the best-val-loss
checkpoint — not the last epoch — is what's saved and used).
*Follow-up: "What if train accuracy is high but val accuracy is low?" — that's the overfitting signature; describe what you'd check (increase augmentation, reduce epochs, get more data).*

### 18. What is data augmentation?
Randomly modifying training images each epoch (crop, flip, color jitter) so
the model sees slightly different versions of the same image, instead of the
exact same pixels every time — improves generalization.

### 19. Difference between validation and test data?
Validation data is used *during* training to pick the best checkpoint and tune
decisions (like the UNCERTAIN threshold) — the model's training process
"sees" it indirectly. Test data is only touched once, after training is fully
done, to report a final, unbiased performance estimate.
*Follow-up: "What would leak if you tuned on test data instead?" — you'd overestimate real-world performance because you'd be optimizing for the exact data you're reporting numbers on.*

### 20. What does your confusion matrix show?
Counts of REAL images correctly called REAL, MANIPULATED images correctly
called MANIPULATED, REAL images wrongly called MANIPULATED (false positives),
and MANIPULATED images wrongly called REAL (false negatives) — on the test set.

### 21. What are precision and recall?
Precision: of everything the model flagged as MANIPULATED, what fraction
actually was. Recall: of everything that was actually MANIPULATED, what
fraction did the model catch. There's a tradeoff between them controlled by
the decision threshold.

### 22. What are false positives and false negatives here?
False positive: a REAL image predicted as MANIPULATED. False negative: a
MANIPULATED image predicted as REAL. False negatives are arguably more
concerning in this domain — they mean actual synthetic/manipulated content
slips through undetected, which is the more harmful failure mode for a
detection tool. False positives cause user friction (real content flagged) but
the harm is more contained.

### 23. How did you perform error analysis?
`notebooks/02_error_analysis.ipynb` runs the trained model over the full test
set, buckets predictions into correct-real / correct-fake / false-positive /
false-negative, and displays example images from each bucket for manual
inspection — this surfaces failure patterns a single accuracy number hides.
*Follow-up: "What did you actually find?" — this must come from your own run; see the notebook's notes section.*

### 24. What are the limitations of your system?
Small single-source training subset, one generation technique (StyleGAN) —
won't necessarily catch other deepfake/generation methods; Haar Cascade misses
some faces; confidence is a probability estimate, not proof; not tested on
real-world social-media-quality images. Full list in the README.

### 25. How would you improve it?
More diverse training data (multiple generation techniques, real-world
compression artifacts), a stronger face detector (e.g. a DNN-based detector
instead of Haar Cascade), calibrating the confidence score, and — as future
scope — extending to video.

### 26. How does Streamlit communicate with your model?
`app.py` imports `Predictor` from `src/predict.py` directly (same Python
process) — the uploaded image (in-memory NumPy array, never written to disk)
is passed straight into the same preprocessing + model-inference functions
used everywhere else in the project. No separate backend/API layer.
*Follow-up: "Why not a separate Flask/FastAPI backend?" — unnecessary complexity for a single-process demo; Streamlit's own Python runtime is sufficient here.*

### 27. How did you deploy the application?
*Fill in honestly based on what you actually did* — if you deployed to
Streamlit Community Cloud, describe that; if not deployed yet, say "Not
implemented yet — runs locally via `streamlit run app.py`."

### 28. What happens when a user uploads an image?
Bytes are read into memory -> decoded with OpenCV -> face detected and cropped
-> resized/normalized identically to training data -> passed through the
model -> sigmoid probability -> thresholded (with UNCERTAIN band) -> result
and confidence displayed, with the disclaimer always shown. Nothing is
persisted to disk.

### 29. How is the confidence score calculated?
The model outputs one logit; `sigmoid(logit)` gives P(manipulated) in [0, 1].
The displayed confidence is that probability if predicting MANIPULATED, or
`1 - probability` if predicting REAL — i.e., "how sure the model is of
whichever label it picked."
*Follow-up: "Is this confidence calibrated?" — no formal calibration (e.g. temperature scaling) was done; be upfront that sigmoid output isn't guaranteed to match true likelihood, it's a relative confidence signal.*
