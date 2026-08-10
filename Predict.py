from ultralytics import YOLO
import cv2
import os

model = YOLO("models/best.pt")


def detect_plate(image_path):

    image = cv2.imread(image_path)

    results = model(image)

    boxes = results[0].boxes

    cropped_images = []

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = image[y1:y2, x1:x2]

        save_path = f"cropped/plate_{i}.jpg"

        cv2.imwrite(save_path, crop)

        cropped_images.append(save_path)

    return cropped_images