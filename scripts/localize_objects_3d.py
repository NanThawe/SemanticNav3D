import argparse
import bisect
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.tum_rgbd import TUMRGBDSource
from src.trajectory_eval import load_tum_trajectory
from src.object_localization import (
    estimate_object_position,
)


def nearest_pose(
    poses,
    timestamps,
    timestamp,
    max_difference=0.02,
):
    index = bisect.bisect_left(
        timestamps,
        timestamp,
    )

    candidates = []

    if index < len(poses):
        candidates.append(
            poses[index]
        )

    if index > 0:
        candidates.append(
            poses[index - 1]
        )

    if not candidates:
        return None

    pose = min(
        candidates,
        key=lambda p:
            abs(p.timestamp - timestamp),
    )

    if (
        abs(pose.timestamp - timestamp)
        > max_difference
    ):
        return None

    return pose


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Estimate lightweight 3D object "
            "positions from RGB-D detections."
        )
    )

    parser.add_argument(
        "--sequence",
        required=True,
    )

    parser.add_argument(
        "--detections",
        required=True,
    )

    parser.add_argument(
        "--trajectory",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    source = TUMRGBDSource(
        args.sequence
    )

    # --------------------------------------------------
    # Load ORB-SLAM3 poses
    # --------------------------------------------------

    poses = load_tum_trajectory(
        args.trajectory
    )

    pose_timestamps = [
        pose.timestamp
        for pose in poses
    ]

    # --------------------------------------------------
    # Load detections
    # --------------------------------------------------

    with open(
        args.detections,
        "r",
        newline="",
    ) as f:

        detections = list(
            csv.DictReader(f)
        )

    rows = []

    failed_depth = 0
    failed_pose = 0

    # --------------------------------------------------
    # Process each detection
    # --------------------------------------------------

    for detection in detections:

        frame_index = int(
            detection["frame_index"]
        )

        frame = source[
            frame_index
        ]

        pose = nearest_pose(
            poses,
            pose_timestamps,
            frame.timestamp,
        )

        if pose is None:
            failed_pose += 1
            continue

        bbox = (
            float(detection["x1"]),
            float(detection["y1"]),
            float(detection["x2"]),
            float(detection["y2"]),
        )

        position = (
            estimate_object_position(
                depth_image=frame.depth,
                bbox=bbox,
                intrinsics=frame.intrinsics,
                camera_position=pose.position,
                camera_quaternion=pose.quaternion,
            )
        )

        if position is None:
            failed_depth += 1
            continue

        rows.append(
            {
                **detection,

                "depth_m":
                    position.depth,

                "pixel_u":
                    position.pixel_u,

                "pixel_v":
                    position.pixel_v,

                "x_camera":
                    position.x_camera,

                "y_camera":
                    position.y_camera,

                "z_camera":
                    position.z_camera,

                "x_world":
                    position.x_world,

                "y_world":
                    position.y_world,

                "z_world":
                    position.z_world,
            }
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        "depth_m",
        "pixel_u",
        "pixel_v",

        "x_camera",
        "y_camera",
        "z_camera",

        "x_world",
        "y_world",
        "z_world",
    ]

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n3D object localisation")
    print("----------------------")

    print(
        f"2D detections:       "
        f"{len(detections)}"
    )

    print(
        f"3D positions:        "
        f"{len(rows)}"
    )

    print(
        f"Depth failures:      "
        f"{failed_depth}"
    )

    print(
        f"Pose failures:       "
        f"{failed_pose}"
    )

    if detections:

        success_rate = (
            100
            * len(rows)
            / len(detections)
        )

        print(
            f"3D success rate:     "
            f"{success_rate:.2f}%"
        )

    print(
        f"Saved to:            "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()