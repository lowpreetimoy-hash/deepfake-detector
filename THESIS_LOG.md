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

**Failure 4 — OpenCV Haar Cascade (temporary solution):**
- Adopted as emergency fallback — zero dependencies
- Works on frontal faces in good lighting
- Failed on low light, angled, blurry faces
- Failed completely on webcam photos
- Root cause: 2001-era algorithm not suited for modern use

**Final Solution — OpenCV DNN Caffe Model:**
- deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel
- cv2.dnn.readNetFromCaffe() implementation
- No new dependencies — works on CPU only
- Detects faces at 99%+ confidence
- Handles low light, angled, blurry, webcam photos
- Bug found: RGB vs BGR color space — DNN needs RGB directly
- Bug found: Extra quotes in file path caused silent UNKNOWN

**Lesson:**
Modern ML libraries have aggressive breaking changes. Sometimes the
simplest solution is not the best one. Face detection quality is the
foundation of the entire pipeline — a weak face detector makes a
strong classifier useless. Always test components independently
before integrating into the pipeline.

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

**Band-aids applied during debugging:**
- ELA thresholds changed from 30/25 to 80/60
  (ELA was scoring everything 0.9+ — too sensitive)
- FAKE_THRESHOLD raised from 0.5 to 0.75
  (needed to correctly classify real webcam photos)
- Ensemble weights tested 35/35/30 vs 25/50/25
  (reverted to 35/35/30 — lower false positive confidence)

**Lesson:**
Temporary debugging fixes must be explicitly re-evaluated.
Never let workarounds become permanent production logic.
Document them clearly and revisit after each major fix.

---

## Phase 7 — Streamlit UI

**Failure:**
- PowerShell breaks f-strings with nested quotes in one-liner commands
- Had to move test code to separate test_pipeline.py file

**Success:**
- Full working web app on first run
- File upload, preview, analysis, results all working
- Clean UI with confidence bar, detector breakdown, face crops
- Green banner for REAL, red banner for FAKE
- Yellow warning for no face detected
- All 5 UI components verified in final acid test

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

### Training Run 3 — Proper Engineering (COMPLETE ✅)

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
- Feed a keyboard/mouse photo → clean UNKNOWN verdict
- Zero crashes on non-human media ✅

**Lesson:**
Production systems must handle ALL inputs gracefully.
A crash is always worse than a clean unknown verdict.

---

### Fix 2 — Haar Cascade → OpenCV DNN (COMPLETE ✅)

**Problem:**
- Haar Cascade failed on low light, angled, blurry faces
- Webcam photo (720x1280) completely undetected
- Pipeline returning UNKNOWN on valid human faces

**Debugging journey:**
- test_dnn.py confirmed DNN detects face at 99.99%
- test_debug.py confirmed frame is correct 720x1280
- Root cause: extra quotes in IMAGE_PATH → silent failure
- Secondary bug: BGR conversion unnecessary — RGB works directly

**Fix applied:**
- Downloaded deploy.prototxt + res10 caffemodel to data/
- Rewrote face_validator.py using cv2.dnn.readNetFromCaffe()
- Removed unnecessary RGB→BGR conversion
- Removed resize from media_loader — detection on original size
- FACE_CONFIDENCE_THRESHOLD = 0.5

**Result:**
- Webcam photo detected at 99.99% confidence ✅
- Low light photo detected at 100% confidence ✅
- All previous failing photos now detected correctly ✅

**Lesson:**
Always test components independently before blaming the model.
The face detector was the bottleneck, not the classifier.
Color space bugs (RGB vs BGR) cause completely silent failures.

---

### Fix 3 — Band-Aid Re-evaluation (COMPLETE ✅)

**Tested:**
- 35/35/30 vs 25/50/25 ensemble weights
- Original 30/25 vs adjusted 80/60 ELA thresholds
- FAKE_THRESHOLD 0.5 vs 0.75 vs 0.80

**Final decisions:**
- Ensemble weights: 35/35/30 (original — lower false confidence)
- ELA thresholds: 80/60 (kept — ELA working correctly now)
- FAKE_THRESHOLD: 0.75 (raised — real photos correctly classified)

**Lesson:**
No threshold perfectly separates all cases when model
accuracy is limited. Document the trade-offs honestly.

---

### Fix 4 — Acid Test (COMPLETE ✅)

**Results:**
| Image Type | Prediction | Confidence | Correct? |
|------------|-----------|------------|---------|
| Keyboard/mouse | UNKNOWN | 0% | ✅ |
| Real webcam selfie | REAL | 28.9% | ✅ |
| Real portrait photo | REAL | 25.6% | ✅ |
| AI generated face | REAL | 36.7% | ❌ |
| Edited/filtered photo | FAKE | 75.6% | ✅ |

**Known limitation confirmed:**
Modern AI generated faces (thispersondoesnotexist.com)
classified as REAL. Root cause: training data predates
modern diffusion-based generation techniques.

---

### Fix 5 — Streamlit Final Verification (COMPLETE ✅)

**All UI components verified:**
- File upload ✅
- Image preview ✅
- Analyze button ✅
- Green REAL banner ✅
- Red FAKE banner ✅
- Yellow no-face warning ✅
- Confidence percentage ✅
- Manipulation confidence bar ✅
- Detector breakdown (A, B, C scores) ✅
- Detected face crop display ✅
- No crashes or hangs on CPU ✅

---

## Overall Technical Decisions Log

| Decision | Original | Changed To | Reason |
|----------|----------|------------|--------|
| Face detector | MTCNN | OpenCV Haar | TF dependency |
| Face detector v2 | mtcnn lib | facenet-pytorch | TF dependency |
| Face detector v3 | facenet-pytorch | OpenCV Haar | torch conflict |
| Face detector final | OpenCV Haar | OpenCV DNN | Accuracy |
| AI detector model | ViT | ResNet50 | Simpler, effective |
| Dataset | FaceForensics++ | 140k + DFDC | Kaggle availability |
| Training split | Random 80/20 | GroupShuffleSplit | Validation leakage |
| Loss function | BCE | BCE + pos_weight=2.39 | Class imbalance |
| Frame extraction | cap.set() | cap.grab/retrieve() | Corrupt file hang |
| No-face output | ERROR + crash | UNKNOWN + clean exit | Robustness |
| FAKE threshold | 0.50 | 0.75 | Real photo accuracy |
| ELA thresholds | 30/25 | 80/60 | Sensitivity control |
| Ensemble weights | 35/35/30 | tested 25/50/25 → back to 35/35/30 | Lower false confidence |

---

## Metrics Summary

| Run | Dataset | Det A Acc | Det B Acc | Real World |
|-----|---------|-----------|-----------|------------|
| Run 1 | 140k only | 98.20% | 98.55% | ❌ All FAKE |
| Run 2 | 140k+DFDC | 92.46% | N/A | ❌ All FAKE |
| Run 3 | Balanced+OOD | 77.18% | 76.03% | ✅ Realistic |

---

## Final System Performance

| Test Case | Result | Notes |
|-----------|--------|-------|
| No human face | UNKNOWN ✅ | Correct rejection |
| Real webcam photo | REAL ✅ | Low light handled |
| Real portrait photo | REAL ✅ | High quality handled |
| Edited/filtered photo | FAKE ✅ | Manipulation detected |
| AI generated face | REAL ❌ | Known limitation |
| Video with face | Working ✅ | Frame extraction ok |

---

## Key Thesis Points (Elaborate These)

1. **Ensemble learning superiority** — 3 detectors catch
   different manipulation types that single models miss.
   ELA catches manual editing, EfficientNet catches face
   swaps, ResNet50 catches AI generation patterns.

2. **OOD validation importance** — 98% in-distribution
   vs 77% OOD demonstrates why dataset choice matters
   more than model architecture. The gap proves the model
   was not truly generalizing.

3. **Domain gap problem** — StyleGAN faces vs real camera
   footage are fundamentally different distributions.
   Mixing without domain balancing produces a model that
   ignores the minority distribution entirely.

4. **Data engineering > model engineering** — All major
   accuracy improvements came from fixing data pipelines
   (GroupShuffleSplit, domain capping, augmentation)
   not from changing model architectures.

5. **Dependency management in ML** — MTCNN → facenet →
   MediaPipe → Haar Cascade → OpenCV DNN journey shows
   real-world ML engineering challenges not covered in
   academic tutorials.

6. **Production vs research mindset** — 98% accuracy
   sounds impressive but meant nothing in deployment.
   Honest evaluation requires adversarial test conditions
   on unseen real-world data.

7. **Class imbalance handling** — pos_weight mathematical
   derivation (6345/2655 = 2.39) and why wrong values
   completely flip model behavior from all-FAKE to all-REAL.

8. **Video_id tracking for GroupShuffleSplit** — preventing
   data leakage in video datasets is a non-trivial
   engineering problem unique to temporal data. Frames
   from the same video must not appear in both train and val.

9. **Silent failures in ML** — black image poisoning,
   validation leakage, domain imbalance, RGB/BGR color
   space bugs all fail completely silently with no error
   messages — only incorrect predictions that look normal.

10. **Face detection as pipeline foundation** — entire
    system accuracy depends on face detector quality.
    A 99% accurate deepfake classifier is 0% useful if
    the face detector can't find the face to analyze.

11. **Training data recency problem** — model trained on
    DFDC (2020) and 140k StyleGAN (2020) cannot detect
    faces from modern diffusion models (2022+). AI
    generation technology evolves faster than datasets.

12. **Threshold tuning trade-offs** — raising FAKE_THRESHOLD
    from 0.5 to 0.75 correctly classifies real photos but
    allows some AI faces through. There is no perfect
    threshold when model scores overlap between classes.
    This is a fundamental limitation requiring better
    training data, not better thresholds.

13. **Component isolation in debugging** — when pipeline
    fails, test each component independently before
    blaming the model. The DNN detected faces at 99.99%
    in isolation but failed in pipeline due to a path
    quoting bug — not a model problem at all.

14. **Refusing pretrained models as contribution** — using
    a HuggingFace pretrained deepfake detector would have
    given better accuracy but reduced original contribution
    to "UI wrapper around someone else's model." The
    decision to train from scratch, despite lower accuracy,
    produced deeper understanding and genuine contribution.

---

## Future Improvements

### Short Term
1. Retrain with full 140k dataset (not just 3,000 samples)
2. Add modern AI generated faces to training data
3. Fix FAKE_THRESHOLD calibration with temperature scaling
4. Add Grad-CAM heatmap visualization

### Medium Term
5. Real-time webcam detection in Streamlit
6. Multi-face analysis per image
7. Audio deepfake detection module
8. Weighted frame confidence for video verdict

### Long Term
9. Mobile application (React Native or Flutter)
10. REST API endpoint using FastAPI
11. Integration with social media platforms
12. Continuous model retraining pipeline