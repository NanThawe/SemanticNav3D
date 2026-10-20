from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float


class YOLODetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        image_size: int = 640,
    ):
        self.model = YOLO(model_name)

        self.confidence_threshold = confidence_threshold
        self.image_size = image_size


    def detect(
        self,
        image: np.ndarray,
    ) -> list[Detection]:

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )

        detections = []

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            class_name = result.names[
                class_id
            ]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections