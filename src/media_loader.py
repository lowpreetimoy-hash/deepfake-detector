import cv2
import os
import numpy as np
from PIL import Image

# Supported file types
SUPPORTED_IMAGES = ['.jpg', '.jpeg', '.png']
SUPPORTED_VIDEOS = ['.mp4', '.avi', '.mov']

# How many frames to extract from a video
FRAME_INTERVAL = 15  # extract 1 frame every 15 frames
MAX_FRAMES = 20      # maximum frames to analyze per video

# Standard size all images/frames will be resized to
TARGET_SIZE = (224, 224)


def get_file_type(file_path):
    """
    Detects whether the uploaded file is an image, video, or unsupported.
    Returns: 'image', 'video', or 'unsupported'
    """
    if not os.path.exists(file_path):
        return 'not_found'

    extension = os.path.splitext(file_path)[1].lower()

    if extension in SUPPORTED_IMAGES:
        return 'image'
    elif extension in SUPPORTED_VIDEOS:
        return 'video'
    else:
        return 'unsupported'


def load_image(file_path):
    """
    Loads image without resizing.
    Resizing happens after face detection.
    """
    try:
        image = Image.open(file_path).convert('RGB')
        image_array = np.array(image)
        return image_array

    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def extract_frames(file_path):
    """
    Extracts frames from a video at fixed intervals.
    Returns: list of numpy arrays (frames) or empty list if failed
    """
    frames = []

    try:
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file: {file_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")

        frame_count = 0
        extracted_count = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_count % FRAME_INTERVAL == 0:
                frame = cv2.resize(frame, TARGET_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                extracted_count += 1

            if extracted_count >= MAX_FRAMES:
                break

            frame_count += 1

        cap.release()
        print(f"Extracted {extracted_count} frames from video")
        return frames

    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []


def load_media(file_path):
    """
    Main function that handles both images and videos.
    Returns: dictionary with media info and frames
    """
    result = {
        'file_path': file_path,
        'media_type': None,
        'frames': [],
        'error': None
    }

    file_type = get_file_type(file_path)

    if file_type == 'not_found':
        result['error'] = 'File not found. Please check the path.'
        return result

    elif file_type == 'unsupported':
        result['error'] = 'Unsupported file type. Please upload jpg, png, mp4, avi or mov.'
        return result

    elif file_type == 'image':
        result['media_type'] = 'image'
        image = load_image(file_path)

        if image is None:
            result['error'] = 'Failed to load image.'
            return result

        result['frames'] = [image]
        return result

    elif file_type == 'video':
        result['media_type'] = 'video'
        frames = extract_frames(file_path)

        if len(frames) == 0:
            result['error'] = 'Failed to extract frames from video.'
            return result

        result['frames'] = frames
        return result
