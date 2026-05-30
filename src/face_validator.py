import cv2
import numpy as np
from PIL import Image

# Paths to DNN model files
PROTOTXT_PATH = 'data/deploy.prototxt'
CAFFEMODEL_PATH = 'data/res10_300x300_ssd_iter_140000.caffemodel'

# Load OpenCV DNN face detector
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

# Minimum confidence for a valid face detection
FACE_CONFIDENCE_THRESHOLD = 0.5

# Minimum face size to ignore tiny detections
MIN_FACE_SIZE = 30


def detect_faces_in_frame(frame):
    """
    Detects faces using OpenCV DNN (ResNet SSD).
    Returns: list of detected faces with details
    """
    try:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        h, w = frame.shape[:2]

        # DNN works with RGB frame directly
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        valid_faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence >= FACE_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                face_w = x2 - x1
                face_h = y2 - y1

                if face_w >= MIN_FACE_SIZE and face_h >= MIN_FACE_SIZE:
                    valid_faces.append({
                        'confidence': round(confidence, 4),
                        'box': [x1, y1, face_w, face_h],
                        'keypoints': None
                    })

        return valid_faces

    except Exception as e:
        print(f"Error detecting faces: {e}")
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
    Returns: tuple (bool, str)
    """
    validation = validate_media_for_faces(frames)
    return validation['has_face'], validation['message']
