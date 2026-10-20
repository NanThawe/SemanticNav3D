import argparse
import csv
import sys
from bisect import bisect_left
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.tum_rgbd import TUMRGBDSource
from src.trajectory_eval import load_tum_trajectory


def build_timestamp_index(source):
    """
    Read each RGB-D frame once and cache only its timestamp.

    The previous implementation scanned the full source again for every
    SLAM keyframe. Because source[index] loads an RGB-D frame, that caused
    hundreds of thousands of repeated image/depth reads on longer sequences.
    """
    timestamps = []
    total = len(source)

    print("Indexing RGB-D timestamps...")

    for index in range(total):
        frame = source[index]
        timestamps.append(frame.timestamp)

        if (
            (index + 1) % 250 == 0
            or index + 1 == total
        ):
            print(
                f"\rIndexed {index + 1}/{total} frames",
                end="",
                flush=True,
            )

    print()
    return timestamps


def find_nearest_frame(
    frame_timestamps,
    timestamp,
):
    """
    Find the nearest timestamp in O(log N) using binary search.
    """
    if not frame_timestamps:
        return None, float("inf")

    position = bisect_left(
        frame_timestamps,
        timestamp,
    )

    candidates = []

    if position < len(frame_timestamps):
        candidates.append(position)

    if position > 0:
        candidates.append(position - 1)

    if not candidates:
        return None, float("inf")

    best_index = min(
        candidates,
        key=lambda index: abs(
            frame_timestamps[index]
            - timestamp
        ),
    )

    best_difference = abs(
        frame_timestamps[best_index]
        - timestamp
    )

    return best_index, best_difference


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select TUM RGB-D frames corresponding "
            "to ORB-SLAM3 keyframes."
        )
    )

    parser.add_argument(
        "--sequence",
        required=True,
        help="Path to TUM RGB-D sequence.",
    )

    parser.add_argument(
        "--keyframes",
        required=True,
        help="ORB-SLAM3 KeyFrameTrajectory.txt",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--max-time-difference",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save selected RGB images.",
    )

    args = parser.parse_args()

    sequence_path = Path(args.sequence)
    output_path = Path(args.output)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = TUMRGBDSource(
        sequence_path
    )

    keyframe_poses = load_tum_trajectory(
        args.keyframes
    )

    print("\nKeyframe selection")
    print("------------------")
    print(f"RGB-D frames:       {len(source)}")
    print(f"SLAM keyframes:     {len(keyframe_poses)}")

    frame_timestamps = build_timestamp_index(
        source
    )

    selected = []
    used_indices = set()

    image_dir = None

    if args.save_images:
        image_dir = (
            output_path
            / "rgb"
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for keyframe_number, keyframe in enumerate(
        keyframe_poses,
        start=1,
    ):
        index, difference = find_nearest_frame(
            frame_timestamps,
            keyframe.timestamp,
        )

        if index is None:
            continue

        if difference > args.max_time_difference:
            continue

        if index in used_indices:
            continue

        used_indices.add(index)

        selected.append(
            {
                "frame_index": index,
                "frame_timestamp": frame_timestamps[index],
                "keyframe_timestamp": keyframe.timestamp,
                "time_difference": difference,
            }
        )

        if args.save_images:
            frame = source[index]

            filename = (
                image_dir
                / f"keyframe_{len(selected) - 1:04d}.png"
            )

            cv2.imwrite(
                str(filename),
                frame.rgb,
            )

        if (
            keyframe_number % 100 == 0
            or keyframe_number == len(keyframe_poses)
        ):
            print(
                f"\rMatched "
                f"{keyframe_number}/"
                f"{len(keyframe_poses)} "
                f"SLAM keyframes",
                end="",
                flush=True,
            )

    print()

    csv_path = (
        output_path
        / "keyframes.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "frame_timestamp",
                "keyframe_timestamp",
                "time_difference",
            ],
        )

        writer.writeheader()
        writer.writerows(selected)

    print(f"Matched keyframes:  {len(selected)}")

    percentage = (
        100.0
        * len(selected)
        / len(source)
    )

    print(
        f"Semantic workload:  "
        f"{percentage:.2f}% of frames"
    )

    print(
        f"\nSaved metadata to:\n"
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()
