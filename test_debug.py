import cv2
import numpy as np
from PIL import Image
from src.media_loader import load_media

# Load through pipeline
result = load_media(r'D:\DeepFake Detector\tested photos\real_webcam.jpg')
frame = result['frames'][0]

print(f"Frame type: {type(frame)}")
print(f"Frame shape: {frame.shape}")
print(f"Frame dtype: {frame.dtype}")
print(f"Frame min/max: {frame.min()}, {frame.max()}")

# Now test DNN directly on this frame
net = cv2.dnn.readNetFromCaffe(
    'data/deploy.prototxt',
    'data/res10_300x300_ssd_iter_140000.caffemodel'
)

# Try with RGB frame as-is
print("\n--- Test 1: RGB frame as-is ---")
blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)),
    1.0, (300, 300),
    (104.0, 177.0, 123.0)
)
net.setInput(blob)
dets = net.forward()
for i in range(dets.shape[2]):
    conf = float(dets[0, 0, i, 2])
    if conf > 0.1:
        print(f"  Detection {i}: {conf:.4f}")

# Try converting RGB to BGR
print("\n--- Test 2: Convert RGB→BGR ---")
bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
blob2 = cv2.dnn.blobFromImage(
    cv2.resize(bgr, (300, 300)),
    1.0, (300, 300),
    (104.0, 177.0, 123.0)
)
net.setInput(blob2)
dets2 = net.forward()
for i in range(dets2.shape[2]):
    conf = float(dets2[0, 0, i, 2])
    if conf > 0.1:
        print(f"  Detection {i}: {conf:.4f}")
