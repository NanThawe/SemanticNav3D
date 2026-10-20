import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.tum_rgbd import TUMRGBDSource
from src.object_detector import YOLODetector


def load_keyframe_indices(
    csv_path: Path,
) -> list[int]:

    indices = []

    with csv_path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            indices.append(
                int(row["frame_index"])
            )

    return indices


def draw_detections(
    image,
    detections,
):

    output = image.copy()

    for detection in detections:

        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run object detection on "
            "ORB-SLAM3 keyframes."
        )
    )

    parser.add_argument(
        "--sequence",
        required=True,
    )

    parser.add_argument(
        "--keyframes",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = TUMRGBDSource(
        args.sequence
    )

    keyframe_indices = (
        load_keyframe_indices(
            Path(args.keyframes)
        )
    )

    detector = YOLODetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
    )

    detections_path = (
        output_path / "detections.csv"
    )

    image_output_path = (
        output_path / "annotated"
    )

    if args.save_images:
        image_output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    rows = []
    inference_times = []

    print("\nObject detection")
    print("----------------")
    print(
        f"Keyframes:          "
        f"{len(keyframe_indices)}"
    )

    for semantic_index, frame_index in enumerate(
        keyframe_indices
    ):

        frame = source[frame_index]

        start = time.perf_counter()

        detections = detector.detect(
            frame.rgb
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        inference_times.append(
            elapsed
        )

        for detection_id, detection in enumerate(
            detections
        ):

            rows.append(
                {
                    "frame_index":
                        frame_index,

                    "timestamp":
                        frame.timestamp,

                    "detection_id":
                        detection_id,

                    "class_id":
                        detection.class_id,

                    "class_name":
                        detection.class_name,

                    "confidence":
                        detection.confidence,

                    "x1":
                        detection.x1,

                    "y1":
                        detection.y1,

                    "x2":
                        detection.x2,

                    "y2":
                        detection.y2,
                }
            )

        if args.save_images:

            annotated = draw_detections(
                frame.rgb,
                detections,
            )

            filename = (
                image_output_path
                / f"frame_{frame_index:04d}.jpg"
            )

            cv2.imwrite(
                str(filename),
                annotated,
            )

        print(
            f"\r"
            f"{semantic_index + 1}"
            f"/{len(keyframe_indices)}",
            end="",
        )

    print()

    fieldnames = [
        "frame_index",
        "timestamp",
        "detection_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with detections_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    inference_times = np.asarray(
        inference_times
    )

    mean_time = float(
        np.mean(inference_times)
    )

    median_time = float(
        np.median(inference_times)
    )

    print("\nDetection summary")
    print("-----------------")

    print(
        f"Detections:         {len(rows)}"
    )

    print(
        f"Mean inference:     "
        f"{mean_time:.4f} s"
    )

    print(
        f"Median inference:   "
        f"{median_time:.4f} s"
    )

    print(
        f"Mean detector FPS:  "
        f"{1 / mean_time:.2f}"
    )

    print(
        f"Saved to:           "
        f"{detections_path}"
    )


if __name__ == "__main__":
    main()