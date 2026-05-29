# Project Journey — Failures & Successes Log
(For Thesis / Project Report)

---

## Phase 1 — Environment Setup

**Success:**
- Python 3.11.9 virtual environment created cleanly
- PyTorch 2.12.0 CPU installed successfully
- All core libraries installed

**Failure:**
- Python 3.13 incompatible with PyTorch — had to install 3.11 separately
- facenet-pytorch 2.6.0 incompatible with torch 2.12 — dependency conflict
- MediaPipe 0.10+ removed solutions API — breaking change from older tutorials
- PowerShell handles mkdir differently than CMD — caused confusion

**Lesson:**
Library compatibility is a real engineering problem. Always verify
version compatibility before installing. Python ecosystem moves fast
and breaking changes are common.

---

## Phase 2 — Media Loader

**Success:**
- Image loading and frame extraction working perfectly
- Correct numpy array shapes (224, 224, 3)
- Video frame sampling with configurable interval

No major failures.

---

## Phase 3 — Face Validator

**Failure 1 — MTCNN + TensorFlow:**
- Original plan used mtcnn library
- Discovered it requires TensorFlow as backend
- TensorFlow is 2GB+ and conflicts with our PyTorch setup
- Wasted time installing and uninstalling

**Failure 2 — facenet-pytorch:**
- Switched to facenet-pytorch for PyTorch-native MTCNN
- facenet-pytorch 2.6.0 requires torch<=2.2
- Our project uses torch 2.12 — incompatible
- facenet-pytorch silently downgraded our torch to 2.2.2
- Caused transformers warning and version conflicts

**Failure 3 — MediaPipe:**
- Switched to MediaPipe as alternative
- MediaPipe 0.10+ completely removed mp.solutions API
- All online tutorials use old API — none worked
- Wasted time debugging deprecated code

**Final Solution:**
- Switched to OpenCV Haar Cascade
- Already bundled with opencv-python — zero extra dependencies
- Works immediately with no version conflicts

**Lesson:**
Modern ML libraries have aggressive breaking changes. Sometimes the
simplest solution (OpenCV built-in) is better than the bleeding-edge
one. Dependency management is a core engineering skill.

---

## Phase 4 — Face Extractor

**Success:**
- Clean crop with padding
- Correct tensor shape (1, 3, 224, 224)
- ImageNet normalization applied correctly

No major failures.

---

## Phase 5 — Detectors

**Success:**
- EfficientNet and ResNet50 loaded via timm
- ELA implemented from scratch without neural network
- All 3 detectors returning confidence scores 0-1

**Minor Issue:**
- HuggingFace symlink warning on Windows
- Harmless but appears every run
- Cause: Windows requires Developer Mode for symlinks

---

## Phase 6 — Ensemble

**Success:**
- Weighted soft voting implemented
- Tie-breaking logic working
- Reason generation at 3 severity levels

No major failures.

---

## Phase 7 — Streamlit UI

**Failure:**
- PowerShell breaks f-strings with nested quotes in one-liner commands
- Had to move test code to separate test_pipeline.py file

**Success:**
- Full working web app on first run
- File upload, preview, analysis, results all working
- Clean UI with confidence bar, detector breakdown, face crops

---

## Phase 8 — Model Training (Most Important)

### Training Run 1 — Pure StyleGAN Dataset

**What we did:**
- Trained on 140k Real and Fake Faces dataset only
- 20,000 samples, 5 epochs
- Detector A: 98.20% validation accuracy
- Detector B: 98.55% validation accuracy

**The failure:**
- Both models classified EVERYTHING as FAKE
- Real photos, edited photos, AI photos — all FAKE
- 98% accuracy was completely meaningless

**Root cause analysis:**
- 140k dataset "fake" = StyleGAN generated faces
- StyleGAN faces are perfectly clean, studio-lit, symmetrical
- Model learned "clean face = real, everything else = fake"
- Real camera photos have compression, noise, lighting variation
- Model interpreted natural photo imperfections as manipulation

**Lesson:**
High validation accuracy on a biased dataset is worse than useless
— it gives false confidence. Always test on out-of-distribution
data before claiming success.

---

### Training Run 2 — First Mixed Dataset Attempt

**What we did:**
- Added 2,000 DFDC video frames to training data
- Mixed with 10,000 140k images
- Detector A: 92.46% accuracy

**The failure:**
- Model still classified everything as FAKE
- 385 real DFDC frames vs 5,000 real 140k images
- Model completely ignored the minority DFDC data
- 92.8% of real class was still sterile 140k data

**Root cause analysis:**
- Domain imbalance — model took the lazy path
- Learned 140k distribution, ignored DFDC
- Validation set drawn from same distribution as training
- 92% accuracy was validation leakage, not real performance

**Lesson:**
Mixing datasets without balancing domains is worse than useless.
The model will always take the path of least resistance. You must
force it to learn the harder distribution.

---

### Engineering Problems Identified Between Run 2 and Run 3

**Problem 1 — Silent Data Poisoning:**
- try-except in Dataset class returned black images with real labels
- Neural network was memorizing that black = real/fake
- Corrupted gradient updates silently
- Fix: Pre-validate all paths, remove try-except entirely

**Problem 2 — Domain Imbalance Math:**
- 385 DFDC real vs 5,000 140k real = 92.8% sterile
- Fix: Cap 140k real at 1,500 samples
- 77 videos × 15 frames = 1,155 DFDC real frames
- New ratio: ~43% DFDC real, ~57% 140k real

**Problem 3 — Validation Leakage:**
- Random 80/20 split on combined dataset
- Validation drew from same distribution as training
- 92% accuracy told us nothing about real-world performance
- Fix: OOD validation using isolated DFDC videos

**Problem 4 — GroupShuffleSplit Requirement:**
- Need video_id column to prevent frame leakage
- Frames from same video can't be in both train and val
- Original extraction script didn't track video_id properly
- Fix: Embed video_id in filename during extraction

**Problem 5 — cap.set() Hanging:**
- OpenCV cap.set() hangs on corrupted video files
- Extraction took 24+ minutes before manual interrupt
- Fix: Use cap.grab() + cap.retrieve() instead
- Sequential reading is more robust than random seeking

**Problem 6 — Class Imbalance:**
- ~6,345 fake vs ~2,655 real samples
- Fix: pos_weight = 2.39 in BCEWithLogitsLoss
- Wrong pos_weight flips model to classify everything as REAL
- Math: pos_weight = total_fake / total_real

**Problem 7 — Domain Augmentation Gap:**
- 140k images: high resolution, clean, centered
- DFDC images: compressed, blurry, varied lighting
- Model uses image quality as shortcut not facial features
- Fix: GaussianBlur + ColorJitter augmentation

---

### Training Run 3 — Proper Engineering (COMPLETE)

**Fixes implemented:**
- Path validation pre-filters corrupt files
- cap.grab()/retrieve() fixes extraction hanging
- video_id tracked in filename and DataFrame
- 140k real capped at 1,500
- GroupShuffleSplit for OOD validation
- GaussianBlur + ColorJitter augmentation
- pos_weight=2.39 in loss function

**Results:**
- Detector A (EfficientNet v3): 77.18% OOD accuracy
- Detector B (ResNet50 v3):     76.03% OOD accuracy
- Validation: 65% unseen DFDC camera footage
- Zero video_id leakage confirmed

**Why 77% is better than 98%:**
- 98% was measured on same distribution as training
- 77% is measured on completely unseen real-world camera footage
- 77% OOD = honest deployment performance
- 98% in-distribution = meaningless lab result

---

## Phase 9 — Local Fixes and Production Hardening

### Fix 1 — Silent Crash in ensemble.py (COMPLETE ✅)

**Problem:**
- Pipeline threw KeyError when no face was detected
- 'total_faces_analyzed' key missing from ERROR return dict
- Any non-human image crashed the entire system

**Fix applied:**
- Changed prediction from 'ERROR' to 'UNKNOWN'
- Added total_faces_analyzed: 0
- Added fake_faces_found: 0
- Added final_score: 0
- Updated message to 'No human face detected in media'

**Result:**
- Feed a brick wall photo → clean UNKNOWN verdict
- Zero crashes on non-human media

**Lesson:**
Production systems must handle ALL inputs gracefully.
A crash is always worse than a clean unknown verdict.

---

### Fix 2 — Haar Cascade Replacement (PENDING)

**Problem identified:**
- OpenCV Haar Cascade is a 2001-era algorithm
- Fails on low light, angled faces, motion blur
- Real-world photos often have these conditions
- Pipeline misses faces it should detect

**Planned fix:**
- Replace with OpenCV DNN Caffe model
- Files: deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel
- Implementation: cv2.dnn.readNetFromCaffe()
- No new dependencies — works on CPU
- Handles angled, low light, blurry faces correctly

**Lesson:**
Face detection quality is the foundation of the entire pipeline.
A state-of-the-art classifier is useless if the face detector
can't find the face to analyze.

---

### Fix 3 — Band-Aid Re-evaluation (PENDING)

**Band-aids applied during debugging:**
- ELA thresholds changed from 30/25 to 80/60
  (made ELA less sensitive after it scored everything 0.9+)
- Ensemble weights changed from 35/35/30 to 25/50/25
  (gave ResNet50 more weight as it was most reliable)

**Plan:**
- After DNN face detector feeds cleaner crops
- Retest with original thresholds and weights
- Only keep band-aids if still needed
- Goal: production logic, not debugging hacks

**Lesson:**
Temporary fixes must be explicitly re-evaluated.
Never let debugging workarounds become permanent
production logic without verification.

---

## Overall Technical Decisions Log

| Decision | Original | Changed To | Reason |
|----------|----------|------------|--------|
| Face detector | MTCNN | OpenCV Haar | Dependency conflicts |
| Face detector lib | mtcnn | facenet-pytorch | TF dependency |
| Face detector v2 | facenet-pytorch | OpenCV Haar | torch version conflict |
| Face detector final | OpenCV Haar | OpenCV DNN (pending) | Accuracy on real photos |
| AI detector model | ViT | ResNet50 | Simpler, equally effective |
| Dataset | FaceForensics++ | 140k + DFDC | Availability on Kaggle |
| Training split | Random 80/20 | GroupShuffleSplit | Validation leakage |
| Loss function | BCE | BCE + pos_weight=2.39 | Class imbalance |
| Frame extraction | cap.set() | cap.grab/retrieve() | Hanging on corrupt files |
| No-face output | ERROR + crash | UNKNOWN + clean exit | Production robustness |

---

## Metrics Summary

| Run | Dataset | Det A Acc | Det B Acc | Real World |
|-----|---------|-----------|-----------|------------|
| Run 1 | 140k only | 98.20% | 98.55% | ❌ All FAKE |
| Run 2 | 140k+DFDC | 92.46% | N/A | ❌ All FAKE |
| Run 3 | Balanced+OOD | 77.18% | 76.03% | ✅ Realistic |

---

## Key Thesis Points (Elaborate These)

1. **Ensemble learning superiority** — 3 detectors catch
   different manipulation types that single models miss

2. **OOD validation importance** — 98% in-distribution
   vs 77% OOD demonstrates why dataset choice matters
   more than model architecture

3. **Domain gap problem** — StyleGAN faces vs real camera
   footage are fundamentally different distributions.
   Mixing without balancing destroys generalization.

4. **Data engineering > model engineering** — Most of
   the project improvement came from fixing data pipelines
   not from changing model architectures

5. **Dependency management in ML** — MTCNN → facenet →
   MediaPipe → Haar Cascade journey shows real-world
   ML engineering challenges not covered in tutorials

6. **Production vs research mindset** — 98% accuracy
   sounds impressive but meant nothing in deployment.
   Honest evaluation requires adversarial test conditions.

7. **Class imbalance handling** — pos_weight mathematical
   derivation and why wrong values flip model behavior

8. **Video_id tracking for GroupShuffleSplit** — preventing
   data leakage in video datasets is a non-trivial
   engineering problem unique to temporal data

9. **Silent failures in ML** — black image poisoning,
   validation leakage, domain imbalance all fail silently
   with no error messages — only wrong predictions

10. **Face detection as pipeline foundation** — entire
    system accuracy depends on face detector quality.
    Haar Cascade limitation directly limits final accuracy.