# 🔍 Deepfake Detector

A multi-modal media authenticity analysis system that detects manipulated human faces in images and videos using ensemble learning.

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds an intelligent media authenticity analysis system capable of detecting manipulated human media across both images and videos. The system supports multiple manipulation scenarios including AI-generated content, face-swapped media, and manually edited face replacements.

### Key Features

- ✅ Supports both **image and video** input
- ✅ **Face validation** before analysis — rejects non-human media
- ✅ **Three independent detectors** targeting different manipulation types
- ✅ **Ensemble learning** with weighted soft voting
- ✅ **Explainable output** with confidence scores and reasons
- ✅ **Streamlit web UI** for easy interaction
- ✅ **OpenCV DNN** face detection (handles low light, angled faces)

---

## 🏗️ System Architecture
User Upload (Image/Video)
↓
Media Loader (frame extraction)
↓
Face Validator (OpenCV DNN)
↓
Face Extractor (crop + normalize)
↓
┌──────────────────────────────┐
│  Detector A  │  Detector B  │  Detector C  │
│ EfficientNet │   ResNet50   │     ELA      │
│  Face Swap   │ AI Generated │ Manual Edit  │
└──────────────────────────────┘
↓
Ensemble Layer (weighted soft voting)
↓
Final Output (REAL/FAKE/UNKNOWN + confidence + reasons)
---

## 🔬 Detectors

| Detector | Model | Target | Method |
|----------|-------|--------|--------|
| A | EfficientNet-B0 | Face swap artifacts | Deep learning |
| B | ResNet50 | AI generated faces | Deep learning |
| C | ELA | Manual editing | Error Level Analysis |

### Ensemble Weights
Detector A (Face Swap):   35%
Detector B (AI Generated): 35%
Detector C (Manual Edit): 30%
---

## 📊 Training Results

| Run | Dataset | Det A Accuracy | Det B Accuracy | Real World |
|-----|---------|---------------|---------------|------------|
| Run 1 | 140k StyleGAN only | 98.20% | 98.55% | ❌ All FAKE |
| Run 2 | 140k + 2k DFDC | 92.46% | N/A | ❌ All FAKE |
| Run 3 | Balanced + OOD | **77.18%** | **76.03%** | ✅ Realistic |

> **Why 77% is better than 98%:** Run 1 and 2 suffered from validation leakage — the validation set was drawn from the same distribution as training. Run 3 uses Out-of-Distribution (OOD) validation on completely unseen DFDC camera footage, giving an honest measure of real-world performance.

### Training Dataset (Run 3)
DFDC video frames:  6,000  (1,155 real + 4,845 fake)
140k StyleGAN:      3,000  (1,500 real + 1,500 fake)
─────────────────────────────────────────────────────
Total:              9,000  samples
Train:              7,256  (80%)
Validation:         1,744  (20% — 65% unseen DFDC)
pos_weight:         2.39   (handles class imbalance)
### Key Engineering Decisions

- **GroupShuffleSplit** — prevents frame leakage between train/val sets
- **Domain balancing** — 140k real capped at 1,500 to prevent model ignoring DFDC footage
- **GaussianBlur + ColorJitter** augmentation — bridges domain gap between clean 140k and real DFDC footage
- **pos_weight = 2.39** — corrects class imbalance (6,345 fake vs 2,655 real)

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11.9 |
| Deep Learning | PyTorch 2.12.0 (CPU) |
| Models | EfficientNet-B0, ResNet50 via timm |
| Face Detection | OpenCV DNN (ResNet SSD) |
| UI | Streamlit |
| Video Processing | OpenCV |
| Training Platform | Kaggle (Tesla T4 GPU) |

---

## 📁 Project Structure
deepfake-detector/
├── src/
│   ├── init.py
│   ├── media_loader.py        # Image/video loading + frame extraction
│   ├── face_validator.py      # Face detection (OpenCV DNN)
│   ├── face_extractor.py      # Face cropping + normalization
│   ├── detectors.py           # 3 detectors + ELA implementation
│   ├── ensemble.py            # Weighted soft voting
│   └── output.py              # Output formatting
├── models/
│   ├── detector_a_efficientnet_v3_best.pth   (ACTIVE)
│   └── detector_b_resnet50_v3_best.pth       (ACTIVE)
├── data/
│   ├── deploy.prototxt                        (DNN face detector)
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── app.py                     # Streamlit web UI
├── main.py                    # CLI entry point
├── test_pipeline.py           # Pipeline testing script
├── requirements.txt
├── PROJECT_LOG.md             # Technical progress log
└── THESIS_LOG.md              # Failures and lessons learned
---

## 🚀 Installation

### Prerequisites
- Python 3.11.x
- 4GB+ RAM
- Windows/Mac/Linux

### Step 1 — Clone repository
```bash
git clone https://github.com/lowpreetimoy-hash/deepfake-detector.git
cd deepfake-detector
```

### Step 2 — Create virtual environment
```bash
py -3.11 -m venv dvenv
dvenv\Scripts\activate        # Windows
source dvenv/bin/activate     # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python streamlit transformers timm scikit-learn pandas numpy pillow
```

### Step 4 — Download face detector model files
```bash
curl -o data/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -o data/res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

### Step 5 — Download trained model weights
Download from the releases section and place in `models/` folder:
- `detector_a_efficientnet_v3_best.pth`
- `detector_b_resnet50_v3_best.pth`

### Step 6 — Run the app
```bash
streamlit run app.py
```

---

## 💻 Usage

### Web UI (Recommended)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

1. Upload an image (JPG, PNG) or video (MP4, AVI, MOV)
2. Click **Analyze Media**
3. View prediction, confidence score, and reasons

### Command Line
```bash
python test_pipeline.py
```
Edit `IMAGE_PATH` in `test_pipeline.py` to point to your file.

---

## 📈 Sample Results

| Input | Prediction | Confidence | Notes |
|-------|-----------|------------|-------|
| Real webcam photo | ✅ REAL | 28.9% | Correctly identified |
| Real portrait photo | ✅ REAL | 25.6% | Correctly identified |
| Edited/filtered photo | ⚠️ FAKE | 75.6% | Manipulation detected |
| No face (keyboard) | ❓ UNKNOWN | 0% | Correctly rejected |

---

## ⚠️ Known Limitations

1. **Modern AI-generated faces** — The model struggles to detect faces generated by modern diffusion models (Stable Diffusion, MidJourney, DALL-E). Training data predates these techniques.

2. **Small dataset** — Trained on 9,000 samples. More data would improve accuracy significantly.

3. **CPU only** — No GPU acceleration. Analysis takes 3-5 seconds per image.

4. **OOD accuracy** — 77% OOD accuracy means approximately 1 in 4 predictions may be incorrect.

---

## 🔮 Future Improvements

- [ ] Retrain with modern AI-generated face datasets
- [ ] Real-time webcam detection
- [ ] Heatmap visualization (Grad-CAM)
- [ ] Audio deepfake detection
- [ ] Multi-face analysis
- [ ] Mobile application
- [ ] REST API endpoint
- [ ] Confidence calibration with temperature scaling

---

## 📚 Datasets Used

| Dataset | Purpose | Source |
|---------|---------|--------|
| 140k Real and Fake Faces | Real + StyleGAN fake faces | [Kaggle (xhlulu)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |
| DFDC (DeepFake Detection Challenge) | Real camera footage + deepfakes | [Kaggle Competition](https://www.kaggle.com/c/deepfake-detection-challenge) |

---

## 🧠 ML Engineering Lessons Learned

1. High accuracy on biased data means nothing
2. Validation must be out-of-distribution (OOD)
3. Domain gap between datasets destroys generalization
4. Data engineering matters more than model architecture
5. Class imbalance must be handled mathematically
6. Face detector quality is the foundation of everything
7. Training data recency matters — old data misses new threats

---

## 👨‍💻 Author

**Lowpreetimoy**
- GitHub: [@lowpreetimoy-hash](https://github.com/lowpreetimoy-hash)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [FaceForensics++ Team](https://github.com/ondyari/FaceForensics)
- [Kaggle DFDC Competition](https://www.kaggle.com/c/deepfake-detection-challenge)
- [xhlulu — 140k Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [OpenCV DNN Face Detector](https://github.com/opencv/opencv)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
