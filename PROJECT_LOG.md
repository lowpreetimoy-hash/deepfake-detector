# Deepfake Detector — Project Log

## Environment
- OS: Windows 11
- Python: 3.11.9 (dvenv)
- Project Path: D:\deepfake-detector
- PyTorch: CPU only (Intel Iris Xe, no dedicated GPU)
- Kaggle: used for heavy model training (Tesla T4 GPU)
- Kaggle notebook: CLOSED — not needed unless retraining

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
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── tests/
├── main.py
├── app.py
├── test_pipeline.py
├── test_dnn.py
├── test_debug.py
├── .gitignore
├── requirements.txt
├── PROJECT_LOG.md
└── THESIS_LOG.md

## Phases
- [x] Phase 1 — Setup, folder structure, dependencies ✅
- [x] Phase 2 — Media loader ✅
- [x] Phase 3 — Face validator ✅
- [x] Phase 4 — Face extractor ✅
- [x] Phase 5 — Detectors ✅
- [x] Phase 6 — Ensemble layer ✅
- [x] Phase 7 — Streamlit UI ✅
- [x] Phase 8 — Fine-tuning on Kaggle ✅
- [x] Phase 9 — Local fixes and final testing ✅
- [ ] Phase 10 — Future improvements (optional)

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
- FAKE_THRESHOLD = 0.75
- ELA thresholds = 80/60
- Ensemble weights = 35/35/30
- OpenCV DNN Caffe model for face detection
- DNN uses RGB frame directly
- media_loader loads original size for face detection
- GitHub: https://github.com/lowpreetimoy-hash/deepfake-detector

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
   → Haar Cascade → OpenCV DNN (final)
9. Path validation before training (no silent data poisoning)
10. Aggressive augmentation to bridge domain gap
11. OOD validation using GroupShuffleSplit on video_id
12. Capped 140k real at 1,500 to prevent domain imbalance
13. Class weights in loss function for fake/real imbalance
14. ensemble.py ERROR → UNKNOWN for no-face media
15. cap.set() → cap.grab/retrieve() for video extraction
16. FAKE_THRESHOLD raised to 0.75

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
2. Domain augmentation — GaussianBlur + ColorJitter
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

## Phase 9 — Local Fixes (ALL COMPLETE ✅)

### ✅ Fix 1 — ensemble.py Silent Crash
- Changed ERROR → UNKNOWN
- Added missing keys to return dict
- Clean exit on no-face media

### ✅ Fix 2 — Replace Haar Cascade with OpenCV DNN
- Downloaded deploy.prototxt + res10 caffemodel
- Detects faces at 99%+ confidence
- Handles low light, angled, blurry faces
- Bug fixed: RGB frame used directly

### ✅ Fix 3 — Band-Aid Re-evaluation
- Reverted to 35/35/30 ensemble weights
- Kept ELA thresholds at 80/60
- FAKE_THRESHOLD set to 0.75

### ✅ Fix 4 — Acid Test
| Image | Prediction | Correct? |
|-------|-----------|---------|
| No face | UNKNOWN 0% | ✅ |
| Real webcam | REAL 28.9% | ✅ |
| AI generated | REAL 36.7% | ❌ |
| Real portrait | REAL 25.6% | ✅ |

### ✅ Fix 5 — Streamlit Final Verification
- File upload working ✅
- Image preview working ✅
- Analyze button working ✅
- Green/red verdict banners working ✅
- Confidence bar working ✅
- Detector breakdown working ✅
- Face crop display working ✅
- No-face warning working ✅
- No UI crashes or hangs ✅

## Current Model Performance
Real webcam selfie → REAL ✅
Real portrait photo → REAL ✅
No face image → UNKNOWN ✅
AI generated face → REAL ❌ (known limitation)
Edited/filtered photo → FAKE ✅
## Known Limitation
Modern AI generated faces (Stable Diffusion, MidJourney,
DALL-E, thispersondoesnotexist.com) are misclassified as REAL.

Root cause: Training data (DFDC + 140k StyleGAN) predates
modern diffusion-based generation. Model never saw these
during training so cannot detect them.

## Future Improvements (Phase 10)

### Priority 1 — Better Training Data
- Add Stable Diffusion generated faces to training
- Add MidJourney faces to training
- Add DALL-E faces to training
- Use full 140k dataset (not just 3,000 samples)
- Expected accuracy improvement: 85-90% OOD

### Priority 2 — Run 4 on Kaggle
- Retrain with larger + more diverse dataset
- Include modern AI generated faces
- Keep same pipeline architecture
- Same GroupShuffleSplit OOD validation
- Expected: correctly detect modern AI faces

### Priority 3 — Real-time Webcam Detection
- Add webcam input option to Streamlit UI
- Process frames in real-time
- Show live prediction overlay
- Requires: optimized inference pipeline

### Priority 4 — Heatmap Visualization
- Show which regions of face triggered detection
- Use Grad-CAM visualization technique
- Highlight manipulated areas in red
- Helps user understand WHY prediction was made

### Priority 5 — Audio Deepfake Detection
- Add audio analysis module
- Detect AI generated voices
- Combine with video detection for full media analysis
- Libraries: librosa, pyaudio

### Priority 6 — Multi-face Analysis
- Currently analyzes first detected face only
- Extend to analyze all faces in frame
- Useful for group photos and videos
- Return per-face verdict + overall verdict

### Priority 7 — Mobile Application
- Convert Streamlit to React Native or Flutter
- On-device inference using ONNX export
- Real-time camera analysis on phone
- Share results directly from app

### Priority 8 — API Endpoint
- Build REST API using FastAPI
- Allow external applications to use detector
- Return JSON with prediction + confidence
- Enable integration with social media platforms

### Priority 9 — Confidence Calibration
- Current confidence scores not well-calibrated
- Apply temperature scaling post-training
- Makes confidence scores more meaningful
- 75% confidence should mean 75% correct

### Priority 10 — Weighted Frame Confidence for Video
- Current: >50% fake frames = fake video
- Upgrade: weighted average of frame confidences
- High confidence frames weigh more than low confidence
- More robust video-level verdict

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
15. Extra quotes in file paths cause silent failures
16. Training data recency matters — old data misses new threats
17. RGB vs BGR color space bugs cause silent failures
18. Always test face detector independently before pipeline
19. Pretrained models reduce original contribution
20. Document failures honestly — they are thesis content

## How to Resume in New Claude Conversation
1. Paste this PROJECT_LOG.md
2. Say: "Continue deepfake detector project from where we stopped"
3. Project is COMPLETE — only future improvements remain

## How to Run App Locally
1. Open VS Code → D:\deepfake-detector
2. Open terminal → dvenv\Scripts\activate
3. streamlit run app.py

## GitHub
Repository: https://github.com/lowpreetimoy-hash/deepfake-detector
Push updates: git add . → git commit -m "message" → git push

## File Purposes Quick Reference
| File | Purpose |
|------|---------|
| media_loader.py | Load image/video, extract frames |
| face_validator.py | Check if human face exists (OpenCV DNN) |
| face_extractor.py | Crop and normalize face region |
| detectors.py | Run 3 detectors on face tensor |
| ensemble.py | Combine scores into final verdict |
| app.py | Streamlit web UI |
| test_pipeline.py | Quick pipeline testing script |
| test_dnn.py | Test DNN face detector directly |
| test_debug.py | Debug frame pipeline issues |
| PROJECT_LOG.md | Technical progress tracker |
| THESIS_LOG.md | Thesis evidence and lessons |