import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Face crop padding — adds extra pixels around detected face
# so we don't cut off edges of the face
PADDING = 20

# Final size for model input
TARGET_SIZE = (224, 224)

# Normalization values — standard ImageNet values
# All pretrained models (EfficientNet, ViT) expect this normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Transform pipeline — converts face crop into model-ready tensor
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])


def crop_face(frame, box, padding=PADDING):
    """
    Crops a single face from a frame using the bounding box.
    Returns: cropped face as numpy array or None if failed
    """
    try:
        x, y, w, h = box
        height, width = frame.shape[:2]

        # Apply padding while staying within image boundaries
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        # Crop the face region
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            print("Warning: Empty face crop detected")
            return None

        return face_crop

    except Exception as e:
        print(f"Error cropping face: {e}")
        return None


def preprocess_face(face_crop):
    """
    Converts a face crop into a normalized tensor ready for model input.
    Returns: tensor of shape (1, 3, 224, 224) or None if failed
    """
    try:
        # Convert numpy array to PIL Image
        if isinstance(face_crop, np.ndarray):
            face_pil = Image.fromarray(face_crop.astype(np.uint8))
        else:
            face_pil = face_crop

        # Ensure RGB
        face_pil = face_pil.convert('RGB')

        # Apply transform pipeline (resize → tensor → normalize)
        face_tensor = transform(face_pil)

        # Add batch dimension — models expect (batch, channels, height, width)
        # (3, 224, 224) → (1, 3, 224, 224)
        face_tensor = face_tensor.unsqueeze(0)

        return face_tensor

    except Exception as e:
        print(f"Error preprocessing face: {e}")
        return None


def extract_faces(frames, face_details):
    """
    Extracts and preprocesses all faces from all frames.
    Returns: list of dictionaries containing face crops and tensors
    """
    extracted = []

    for frame_info in face_details:
        frame_index = frame_info['frame_index']
        frame = frames[frame_index]
        faces = frame_info['faces']

        for face in faces:
            box = face['box']

            # Crop the face
            face_crop = crop_face(frame, box)
            if face_crop is None:
                continue

            # Preprocess into tensor
            face_tensor = preprocess_face(face_crop)
            if face_tensor is None:
                continue

            extracted.append({
                'frame_index': frame_index,
                'box': box,
                'face_crop': face_crop,
                'face_tensor': face_tensor
            })

    print(f"Successfully extracted {len(extracted)} face(s)")
    return extracted
