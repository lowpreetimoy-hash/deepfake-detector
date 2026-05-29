import cv2
import numpy as np
from PIL import Image

# Load OpenCV's built-in face detector (comes bundled with opencv)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Minimum face size to avoid detecting tiny/blurry faces
MIN_FACE_SIZE = 30  # pixels


def detect_faces_in_frame(frame):
    """
    Detects faces in a single frame/image.
    Returns: list of detected faces with their details
    """
    try:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        detections = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        valid_faces = []
        for (x, y, w, h) in detections:
            valid_faces.append({
                'confidence': 1.0,
                'box': [x, y, w, h],
                'keypoints': None
            })

        return valid_faces

    except Exception as e:
        print(f"Error detecting faces in frame: {e}")
        return []


def validate_media_for_faces(frames):
    """
    Checks all frames for human faces.
    Returns: dictionary with validation results
    """
    result = {
        'has_face': False,
        'total_frames_checked': len(frames),
        'frames_with_faces': 0,
        'face_details': [],
        'message': ''
    }

    for i, frame in enumerate(frames):
        faces = detect_faces_in_frame(frame)

        if len(faces) > 0:
            result['has_face'] = True
            result['frames_with_faces'] += 1
            result['face_details'].append({
                'frame_index': i,
                'faces_found': len(faces),
                'faces': faces
            })

    if result['has_face']:
        result['message'] = f"Human face detected in {result['frames_with_faces']} out of {result['total_frames_checked']} frames."
    else:
        result['message'] = "No human face detected. Please upload media containing a visible human face."

    return result


def has_human_face(frames):
    """
    Simple True/False check for human face existence.
    Returns: tuple (bool, str) — (face_found, message)
    """
    validation = validate_media_for_faces(frames)
    return validation['has_face'], validation['message']
