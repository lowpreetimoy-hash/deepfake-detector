import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image

# Use CPU (no GPU available)
DEVICE = torch.device('cpu')

# Confidence threshold — above this = FAKE
FAKE_THRESHOLD = 0.5

print(f"Detectors running on: {DEVICE}")


class FaceSwapDetector(nn.Module):
    """
    Detector A — Detects face swap artifacts using EfficientNet.
    Looks for: blending artifacts, texture mismatch, facial inconsistencies
    """

    def __init__(self):
        super(FaceSwapDetector, self).__init__()

        # Load pretrained EfficientNet, replace final layer for binary classification
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=1
        )
        self.sigmoid = nn.Sigmoid()
        self.model.to(DEVICE)
        self.model.eval()

    def forward(self, face_tensor):
        """
        Returns confidence score between 0 and 1.
        0 = Real, 1 = Fake
        """
        with torch.no_grad():
            face_tensor = face_tensor.to(DEVICE)
            output = self.model(face_tensor)
            confidence = self.sigmoid(output)
            return confidence.item()


class AIGeneratedDetector(nn.Module):
    """
    Detector B — Detects AI generated content using ResNet50.
    Looks for: synthetic textures, diffusion artifacts, generation fingerprints
    """

    def __init__(self):
        super(AIGeneratedDetector, self).__init__()

        # Load pretrained ResNet50, replace final layer for binary classification
        self.model = timm.create_model(
            'resnet50',
            pretrained=True,
            num_classes=1
        )
        self.sigmoid = nn.Sigmoid()
        self.model.to(DEVICE)
        self.model.eval()

    def forward(self, face_tensor):
        """
        Returns confidence score between 0 and 1.
        0 = Real, 1 = Fake
        """
        with torch.no_grad():
            face_tensor = face_tensor.to(DEVICE)
            output = self.model(face_tensor)
            confidence = self.sigmoid(output)
            return confidence.item()


class ManualEditDetector:
    """
    Detector C — Detects manual editing using Error Level Analysis (ELA).
    Looks for: lighting mismatch, boundary artifacts, inconsistent compression
    """

    def __init__(self):
        self.ela_quality = 90
        self.ela_amplifier = 10

    def get_ela_image(self, face_crop):
        """
        Performs Error Level Analysis on a face crop.
        ELA reveals regions that were edited by showing compression differences.
        """
        try:
            # Convert to PIL Image
            if isinstance(face_crop, np.ndarray):
                pil_image = Image.fromarray(face_crop.astype(np.uint8))
            else:
                pil_image = face_crop

            pil_image = pil_image.convert('RGB')

            # Save at specific quality to introduce compression
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=self.ela_quality)
            buffer.seek(0)
            compressed = Image.open(buffer)

            # Calculate difference between original and compressed
            original = np.array(pil_image, dtype=np.float32)
            compressed = np.array(compressed, dtype=np.float32)
            ela_diff = np.abs(original - compressed) * self.ela_amplifier

            # Normalize to 0-255
            ela_diff = np.clip(ela_diff, 0, 255).astype(np.uint8)
            return ela_diff

        except Exception as e:
            print(f"ELA error: {e}")
            return None

    def analyze(self, face_crop):
        """
        Analyzes ELA result to detect manipulation.
        Returns confidence score between 0 and 1.
        0 = Real, 1 = Fake
        """
        try:
            ela_image = self.get_ela_image(face_crop)
            if ela_image is None:
                return 0.5

            # Calculate mean and std of ELA differences
            mean_diff = np.mean(ela_image)
            std_diff = np.std(ela_image)

            # High variation in ELA = signs of editing
            # Normalize to 0-1 range using empirical thresholds
            mean_score = min(mean_diff / 80.0, 1.0)
            std_score = min(std_diff / 60.0, 1.0)

            # Combine both scores
            confidence = (mean_score * 0.5) + (std_score * 0.5)
            return float(confidence)

        except Exception as e:
            print(f"ELA analysis error: {e}")
            return 0.5


def load_detectors():
    """
    Initializes all 3 detectors and loads fine-tuned weights.
    Returns: tuple (detector_a, detector_b, detector_c)
    """
    print("Loading Detector A — EfficientNet (Face Swap)...")
    detector_a = FaceSwapDetector()
    model_a_path = os.path.join(
        'models', 'detector_a_efficientnet_v3_best.pth')
    if os.path.exists(model_a_path):
        detector_a.model.load_state_dict(
            torch.load(model_a_path, map_location=DEVICE)
        )
        print("  ✅ Fine-tuned weights loaded for Detector A")
    else:
        print("  ⚠️ Using pretrained weights for Detector A (fine-tuned not found)")

    print("Loading Detector B — ResNet50 (AI Generated)...")
    detector_b = AIGeneratedDetector()
    model_b_path = os.path.join('models', 'detector_b_resnet50_v3_best.pth')
    if os.path.exists(model_b_path):
        detector_b.model.load_state_dict(
            torch.load(model_b_path, map_location=DEVICE)
        )
        print("  ✅ Fine-tuned weights loaded for Detector B")
    else:
        print("  ⚠️ Using pretrained weights for Detector B (fine-tuned not found)")

    print("Loading Detector C — ELA (Manual Edit)...")
    detector_c = ManualEditDetector()
    print("  ✅ ELA detector ready (no training needed)")

    print("All detectors loaded successfully!")
    return detector_a, detector_b, detector_c


def run_detectors(face_tensor, face_crop, detector_a, detector_b, detector_c):
    """
    Runs all 3 detectors on a single face.
    Returns: dictionary with all confidence scores
    """
    score_a = detector_a.forward(face_tensor)
    score_b = detector_b.forward(face_tensor)
    score_c = detector_c.analyze(face_crop)

    return {
        'face_swap_score': round(score_a, 4),
        'ai_generated_score': round(score_b, 4),
        'manual_edit_score': round(score_c, 4)
    }
