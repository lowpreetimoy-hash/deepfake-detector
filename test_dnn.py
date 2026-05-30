import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    'data/deploy.prototxt',
    'data/res10_300x300_ssd_iter_140000.caffemodel'
)

img = cv2.imread(r'D:\DeepFake Detector\tested photos\16.jpeg')
h, w = img.shape[:2]
print(f'Image shape: {img.shape}')

blob = cv2.dnn.blobFromImage(
    cv2.resize(img, (300, 300)),
    1.0, (300, 300),
    (104.0, 177.0, 123.0)
)

net.setInput(blob)
detections = net.forward()
print(f'Total detections: {detections.shape[2]}')

for i in range(detections.shape[2]):
    conf = float(detections[0, 0, i, 2])
    if conf > 0.1:
        print(f'Detection {i}: confidence={conf:.4f}')
