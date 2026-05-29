# Deepfake Detector — Project Log

## Environment
- OS: Windows 11
- Python: 3.11.9 (dvenv)
- Project Path: D:\deepfake-detector
- PyTorch: CPU only (Intel Iris Xe, no dedicated GPU)
- Kaggle: used for heavy model training (Tesla T4 GPU)
- Kaggle notebook: CLOSED — not needed anymore

## Project Structure
deepfake-detector/
├── src/
│   ├── __init__.py
│   ├── media_loader.py
│   ├── face_validator.py
│   ├── face_extractor.py
│   ├── detectors.py
│   ├── ensemble.py
│   └── output.py
├── models/
│   ├── detector_a_efficientnet_best.pth      (Run 1 - backup)
│   ├── detector_b_resnet50_best.pth          (Run 1 - backup)
│   ├── detector_a_efficientnet_v2_best.pth   (Run 2 - backup)
│   ├── detector_a_efficientnet_v3_best.pth   (Run 3 - ACTIVE)
│   └── detector_b_resnet50_v3_best.pth       (Run 3 - ACTIVE)
├── data/
├── tests/
├── main.py
├── app.py
├── test_pipeline.py
├── requirements.txt
└── PROJECT_LOG.md

## Phases
- [x] Phase 1 — Setup, folder structure, dependencies ✅
- [x] Phase 2 — Media loader ✅
- [x] Phase 3 — Face validator ✅
- [x] Phase 4 — Face extractor ✅
- [x] Phase 5 — Detectors ✅
- [x] Phase 6 — Ensemble layer ✅
- [x] Phase 7 — Streamlit UI ✅
- [x] Phase 8 — Fine-tuning on Kaggle ✅
- [ ] Phase 9 — Local fixes and final testing

## Key Decisions Made
- Using CPU-only PyTorch (no GPU)
- Heavy training ran on Kaggle (Tesla T4 GPU)
- Virtual env named dvenv
- Detector A → EfficientNet (face swap detection)
- Detector B → ResNet50 (AI generated detection)
- Detector C → ELA (manual edit detection)
- Video: >50% fake frames = fake
- Soft voting ensemble with weighted tie-breaking rule
- If 2 say Real, 1 says Fake → use weighted strategy
- Output includes confidence score + specific reasons
- pos_weight = 2.39 in BCEWithLogitsLoss
- source column in DataFrame
  ('dfdc' for DFDC frames, '140k' for StyleGAN)
- OpenCV Haar Cascade used temporarily for face detection
  → REPLACING with OpenCV DNN Caffe model (Phase 9)
- ELA thresholds adjusted to 80/60 (temporary band-aid)
  → Re-evaluate after DNN face detector upgrade
- Ensemble weights adjusted to 25/50/25 (temporary band-aid)
  → Re-evaluate after DNN face detector upgrade

## Corrections Applied to Original Design
1. Added timm, scikit-learn to tech stack
2. Detector C uses ELA for manual edit detection
3. Ensemble has tie-breaking rule, not just simple average
4. Video verdict = aggregated across frames (>50% fake = fake)
5. Datasets: DFDC + 140k mixed with domain balancing
6. Face validation reworded to "prevents unnecessary
   processing of media without human faces"
7. Switched from Xception to EfficientNet + ResNet50
8. Switched from MTCNN → facenet-pytorch → MediaPipe
   → finally OpenCV Haar Cascade (dependency conflicts)
9. Path validation before training (no silent data poisoning)
10. Aggressive augmentation to bridge domain gap
11. OOD validation using GroupShuffleSplit on video_id
12. Capped 140k real at 1,500 to prevent domain imbalance
13. Class weights in loss function for fake/real imbalance
14. ensemble.py ERROR → UNKNOWN for no-face media
15. cap.set() → cap.grab/retrieve() for video extraction

## Installed Libraries (Local)
- torch==2.12.0+cpu
- torchvision==0.27.0+cpu
- torchaudio==2.11.0+cpu
- opencv-python==4.13.0.92
- mediapipe==0.10.35 (installed but not used)
- streamlit==1.57.0
- transformers==5.9.0
- timm==1.0.27
- scikit-learn==1.8.0
- pandas==3.0.3
- numpy==2.4.4
- pillow==12.2.0

## Training Results (Kaggle — Phase 8)

### Run 1 — Pure 140k Dataset (FAILED)
- Dataset: 140k StyleGAN faces only
- Detector A: 98.20% | Detector B: 98.55%
- Problem: Classified everything as FAKE
- Root cause: Only learned StyleGAN artifacts
- Status: DISCARDED

### Run 2 — Mixed Dataset v1 (FAILED)
- Dataset: 140k (10k) + DFDC frames (2,000)
- Detector A: 92.46% | Detector B: STOPPED
- Problem: 92.8% real images still from 140k
- Root cause: Domain imbalance not fixed
- Status: DISCARDED

### Run 3 — Mixed Dataset v2 (COMPLETE ✅)
#### Fixes Applied
1. Path validation — pre-filter corrupt files
2. Domain augmentation — GaussianBlur + JPEG + ColorJitter
3. Balanced domain — 140k real capped at 1,500
4. OOD validation — GroupShuffleSplit on video_id

#### Dataset
- DFDC frames: 6,000 (1,155 real + 4,845 fake)
- 140k samples: 3,000 (1,500 real + 1,500 fake)
- Total: 9,000 | Train: 7,256 | Val: 1,744
- Val: 65% DFDC — true OOD validation
- Zero video_id leakage confirmed
- pos_weight: 2.39

#### Results
- Detector A (EfficientNet v3): 77.18% OOD accuracy
- Detector B (ResNet50 v3):     76.03% OOD accuracy
- Note: Honest real-world scores on unseen camera footage

## Phase 9 — Local Fixes Checklist

### ✅ Fix 1 — ensemble.py Silent Crash (DONE)
- Changed 'ERROR' → 'UNKNOWN' prediction
- Added total_faces_analyzed: 0
- Added fake_faces_found: 0
- Added final_score: 0
- Message: 'No human face detected in media'
- Tested: clean exit on no-face image ✅

### ⏳ Fix 2 — Replace Haar Cascade with OpenCV DNN
- Current: OpenCV Haar Cascade (2001 algorithm)
- Problem: Fails on low light, angled, blurry faces
- Fix: OpenCV DNN Caffe model
- Files needed:
  * deploy.prototxt
  * res10_300x300_ssd_iter_140000.caffemodel
- Download from:
  https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
- Implementation: cv2.dnn.readNetFromCaffe()
- Works on CPU, no new dependencies
- Affects: face_validator.py + face_extractor.py

### ⏳ Fix 3 — Re-evaluate Band-Aid Fixes
After DNN face detector is working:
- Test with original ELA thresholds (30/25)
- Test with original ensemble weights (35/35/30)
- Only keep 80/60 and 25/50/25 if still needed
- Don't let debugging hacks become permanent

### ⏳ Fix 4 — Acid Test (3 specific images)
Test test_pipeline.py on:
1. Image with no human face → must return UNKNOWN
2. High quality real unedited selfie → should return REAL
3. AI generated face or deepfake → should return FAKE
Success: zero crashes, logical confidence scores

### ⏳ Fix 5 — Streamlit Final Verification
- Run: streamlit run app.py
- Upload same 3 test images through browser
- Verify confidence bars, reasons display correctly
- Check no UI hanging on CPU

## Current Status
Phase 8 complete. Phase 9 in progress.
Fix 1 (ensemble crash) done.
Next: Fix 2 — Replace Haar Cascade with OpenCV DNN.

## Known Bugs
1. Haar Cascade fails on low light/angled/blurry faces
   STATUS: NEXT — Fix 2
2. ELA thresholds 80/60 are temporary band-aids
   STATUS: Re-evaluate after Fix 2
3. Ensemble weights 25/50/25 are temporary band-aids
   STATUS: Re-evaluate after Fix 2
4. HuggingFace symlink warning on Windows
   STATUS: Harmless, not critical

## ML Engineering Lessons Learned
1. High accuracy on biased data means nothing
2. Always validate file paths before training
3. Domain gap between datasets destroys generalization
4. Validation must be out-of-distribution (OOD)
5. Augmentation must simulate real-world conditions
6. Always do the actual math on class distribution
7. video_id tracking required for GroupShuffleSplit
8. Class weights fix label imbalance in loss function
9. cap.set() hangs on corrupt videos — use cap.grab()
10. Mixing datasets without domain balance is useless
11. 92% accuracy on same distribution = meaningless
12. Download model weights immediately after training
13. Temporary fixes must be re-evaluated not kept forever
14. Face detector quality directly affects model accuracy

## How to Resume in New Claude Conversation
1. Paste this PROJECT_LOG.md
2. Say: "Continue deepfake detector project from where we stopped"
3. Current task: Fix 2 — Replace Haar Cascade with OpenCV DNN
   Files to modify: face_validator.py + face_extractor.py
   Download needed: deploy.prototxt + res10 caffemodel

## How to Run App Locally
1. Open VS Code → D:\deepfake-detector
2. Open terminal → dvenv\Scripts\activate
3. streamlit run app.py

## File Purposes Quick Reference
| File | Purpose |
|------|---------|
| media_loader.py | Load image/video, extract frames |
| face_validator.py | Check if human face exists |
| face_extractor.py | Crop and normalize face region |
| detectors.py | Run 3 detectors on face tensor |
| ensemble.py | Combine scores into final verdict |
| app.py | Streamlit web UI |
| test_pipeline.py | Quick pipeline testing script |