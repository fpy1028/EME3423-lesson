import cv2
import numpy as np
# Configuration

confThreshold = 0.80  # ≥ 80% confidence

# class names
classesFile = 'coco80.names'
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

# Fruit setup (COCO)
fruit_classes = ['apple', 'banana', 'orange']
fruit_price = {
    'apple': 3,
    'banana': 2,
    'orange': 4
}

# Load image
img = cv2.imread("Resources/stock-photo-orange-apple-and-banana.jpg")
height, width, _ = img.shape

# Load YOLO
net = cv2.dnn.readNetFromDarknet(
    'yolov3-608.cfg',
    'yolov3-608.weights'
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# YOLO forward pass
blob = cv2.dnn.blobFromImage(
    img, 1 / 255, (320, 320),
    (0, 0, 0), swapRB=True, crop=False
)
net.setInput(blob)

output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

bboxes = []
confidences = []
class_ids = []

# Detection
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >= confThreshold:
            class_name = classes[class_id]

            if class_name in fruit_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# Non‑Maximum Suppression
indexes = cv2.dnn.NMSBoxes(
    bboxes, confidences,
    confThreshold, 0.4
)

total_fruits = 0
total_price = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw detections
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = bboxes[i]
        label = classes[class_ids[i]]
        conf_percent = int(confidences[i] * 100)

        total_fruits += 1
        total_price += fruit_price[label]

        # Bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # Label INSIDE top-left
        text = f"{label} {conf_percent}%"
        cv2.putText(img, text, (x + 5, y + 25),
                    font, 0.8, (255, 255, 255), 2)

# Top-right summary
cv2.putText(img,
            f"Total Fruits: {total_fruits}",
            (width - 320, 40),
            font, 0.9, (0, 0, 255), 2)

cv2.putText(img,
            f"Total Price: ${total_price}",
            (width - 320, 80),
            font, 0.9, (0, 0, 255), 2)

cv2.imshow("YOLOv3 Fruit Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
