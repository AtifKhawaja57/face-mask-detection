from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2


class MaskDetector:
    def __init__(self, prototxt_path, weights_path, mask_net_path, confidence):
        self.face_net = cv2.dnn.readNet(prototxt_path, weights_path)
        self.mask_net = load_model(mask_net_path)
        self.face_threshold = confidence

    def detect_face(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        locations = []

        try:
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.face_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    faces.append(face)
                    locations.append((startX, startY, endX, endY))
            return locations[0], faces[0]
        except Exception as e:
            return [], []

    def detect_mask(self, frame):
        location, face = self.detect_face(frame)
        if len(face) > 0:
            faces = np.array([face], dtype="float32")
            prediction = self.mask_net.predict(faces, batch_size=32)[0]
        else:
            prediction = [0, 0]

        return location, prediction

