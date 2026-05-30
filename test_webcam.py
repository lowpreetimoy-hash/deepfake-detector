import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    'data/deploy.prototxt',
    'data/res10_300x300_ssd_iter_140000.caffemodel'
)

# Load image directly with OpenCV (BGR by default)
img = cv2.imread(r'D:\DeepFake Detector\tested photos\real_webcam.jpg')

if img is None:
    print("ERROR: Could not load image — check path!")
else:
    print(f"Image shape: {img.shape}")
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    print(f"Total detections: {detections.shape[2]}")
    print("\nAll detections above 0.1:")
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > 0.1:
            print(f"  Detection {i}: confidence={conf:.4f}")
