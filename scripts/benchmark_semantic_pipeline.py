from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.object_detector import YOLODetector

from src.object_map import (
    ObjectObservation,
    PersistentObjectMap,
)

from src.robot_relative_query import (
    CameraTrajectory,
)

from scripts.demo_tum_semanticnav import (
    load_keyframes,
    load_tum_associations,
    localize_detection,
)


# =====================================================================
# Statistics helpers
# =====================================================================

def statistics(values):
    """
    Return common timing statistics in milliseconds.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "total": 0.0,
        }

    return {
        "mean": float(
            values.mean()
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "p90": float(
            np.percentile(
                values,
                90,
            )
        ),

        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),

        "p99": float(
            np.percentile(
                values,
                99,
            )
        ),

        "min": float(
            values.min()
        ),

        "max": float(
            values.max()
        ),

        "total": float(
            values.sum()
        ),
    }


def print_statistics(
    title,
    values,
):
    """
    Print timing statistics.
    """

    stats = statistics(
        values
    )

    print()
    print(title)

    print(
        "-" * len(title)
    )

    print(
        f"Mean:       "
        f"{stats['mean']:.2f} ms"
    )

    print(
        f"Median:     "
        f"{stats['median']:.2f} ms"
    )

    print(
        f"P90:        "
        f"{stats['p90']:.2f} ms"
    )

    print(
        f"P95:        "
        f"{stats['p95']:.2f} ms"
    )

    print(
        f"P99:        "
        f"{stats['p99']:.2f} ms"
    )

    print(
        f"Minimum:    "
        f"{stats['min']:.2f} ms"
    )

    print(
        f"Maximum:    "
        f"{stats['max']:.2f} ms"
    )

    print(
        f"Total:      "
        f"{stats['total']:.2f} ms"
    )


# =====================================================================
# Detection conversion
# =====================================================================

def detection_to_dict(
    detection,
):
    """
    Convert src.object_detector.Detection into the dictionary
    format expected by localize_detection().
    """

    return {
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


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SemanticNav3D semantic "
            "perception pipeline on CPU."
        )
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    parser.add_argument(
        "--sequence",
        default=(
            "data/tum/"
            "rgbd_dataset_freiburg1_desk"
        ),
    )

    parser.add_argument(
        "--trajectory",
        default=(
            "results/slam/"
            "freiburg1_desk/"
            "CameraTrajectory.txt"
        ),
    )

    parser.add_argument(
        "--keyframes",
        default=(
            "results/keyframes/"
            "freiburg1_desk/"
            "keyframes.csv"
        ),
    )

    # -----------------------------------------------------------------
    # Detector
    #
    # These defaults intentionally match detect_keyframes.py.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Mapping
    # -----------------------------------------------------------------

    parser.add_argument(
        "--association-distance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--min-observations",
        type=int,
        default=3,
    )

    # -----------------------------------------------------------------
    # TUM Freiburg1 camera intrinsics
    # -----------------------------------------------------------------

    parser.add_argument(
        "--fx",
        type=float,
        default=517.3,
    )

    parser.add_argument(
        "--fy",
        type=float,
        default=516.5,
    )

    parser.add_argument(
        "--cx",
        type=float,
        default=318.6,
    )

    parser.add_argument(
        "--cy",
        type=float,
        default=255.3,
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=5000.0,
    )

    parser.add_argument(
        "--pose-max-gap",
        type=float,
        default=0.05,
    )

    # -----------------------------------------------------------------
    # Benchmark settings
    # -----------------------------------------------------------------

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help=(
            "YOLO warm-up runs before timing."
        ),
    )

    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=None,
        help=(
            "Optional keyframe limit for quick testing. "
            "Leave unset for the full benchmark."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/benchmark/"
            "freiburg1_desk"
        ),
    )

    args = parser.parse_args()

    # =================================================================
    # Load dataset metadata
    # =================================================================

    sequence_dir = Path(
        args.sequence
    )

    frames = load_tum_associations(
        sequence_dir
    )

    keyframes = load_keyframes(
        Path(
            args.keyframes
        )
    )

    trajectory = CameraTrajectory(
        Path(
            args.trajectory
        )
    )

    semantic_indices = sorted(
        keyframes
    )

    # Keep the full keyframe count separately so efficiency calculations
    # still refer to the real 147/596 configuration.
    full_semantic_frame_count = len(
        semantic_indices
    )

    if (
        args.max_keyframes
        is not None
    ):
        semantic_indices = (
            semantic_indices[
                :args.max_keyframes
            ]
        )

    print()
    print(
        "SemanticNav3D Semantic Pipeline Benchmark"
    )

    print(
        "========================================"
    )

    print(
        f"RGB-D frames:        "
        f"{len(frames)}"
    )

    print(
        f"Semantic keyframes:  "
        f"{full_semantic_frame_count}"
    )

    print(
        f"Frames benchmarked:  "
        f"{len(semantic_indices)}"
    )

    print(
        f"Semantic workload:   "
        f"{100.0 * full_semantic_frame_count / len(frames):.2f}%"
    )

    print()
    print(
        "Detector configuration"
    )

    print(
        "----------------------"
    )

    print(
        f"Model:               "
        f"{args.model}"
    )

    print(
        f"Confidence:          "
        f"{args.confidence:.2f}"
    )

    print(
        f"Image size:          "
        f"{args.image_size}"
    )

    print(
        "Device:              CPU"
    )

    # =================================================================
    # Detector
    #
    # Uses the exact same detector abstraction as detect_keyframes.py.
    # =================================================================

    detector = YOLODetector(
        model_name=args.model,
        confidence_threshold=(
            args.confidence
        ),
        image_size=(
            args.image_size
        ),
    )

    # =================================================================
    # YOLO warm-up
    # =================================================================

    if semantic_indices:

        warmup_index = (
            semantic_indices[0]
        )

        warmup_frame = (
            frames[
                warmup_index
            ]
        )

        warmup_image = cv2.imread(
            str(
                warmup_frame[
                    "rgb_path"
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        if warmup_image is None:
            raise RuntimeError(
                "Failed to load warm-up RGB image: "
                f"{warmup_frame['rgb_path']}"
            )

        print()
        print(
            f"YOLO warm-up runs: "
            f"{args.warmup}"
        )

        for _ in range(
            args.warmup
        ):
            detector.detect(
                warmup_image
            )

    # =================================================================
    # Persistent object mapper
    # =================================================================

    mapper = PersistentObjectMap(
        association_distance=(
            args.association_distance
        )
    )

    # =================================================================
    # Timing storage
    # =================================================================

    image_load_times = []

    detector_times = []

    localisation_times = []

    association_times = []

    semantic_total_times = []

    detection_counts = []

    localisation_counts = []

    benchmark_rows = []

    # =================================================================
    # Benchmark semantic keyframes
    # =================================================================

    for (
        benchmark_index,
        frame_index,
    ) in enumerate(
        semantic_indices,
        start=1,
    ):

        frame_info = (
            frames[
                frame_index
            ]
        )

        semantic_start = (
            time.perf_counter_ns()
        )

        # -------------------------------------------------------------
        # RGB + depth loading
        # -------------------------------------------------------------

        load_start = (
            time.perf_counter_ns()
        )

        rgb_image = cv2.imread(
            str(
                frame_info[
                    "rgb_path"
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        depth_image = cv2.imread(
            str(
                frame_info[
                    "depth_path"
                ]
            ),
            cv2.IMREAD_UNCHANGED,
        )

        image_load_ms = (
            time.perf_counter_ns()
            - load_start
        ) / 1_000_000.0

        if rgb_image is None:
            raise RuntimeError(
                "Failed to load RGB image: "
                f"{frame_info['rgb_path']}"
            )

        if depth_image is None:
            raise RuntimeError(
                "Failed to load depth image: "
                f"{frame_info['depth_path']}"
            )

        # -------------------------------------------------------------
        # Camera pose
        # -------------------------------------------------------------

        timestamp = float(
            frame_info[
                "timestamp"
            ]
        )

        pose = trajectory.nearest_timestamp(
            timestamp,
            max_gap=(
                args.pose_max_gap
            ),
        )

        # -------------------------------------------------------------
        # YOLO11n CPU inference
        # -------------------------------------------------------------

        detector_start = (
            time.perf_counter_ns()
        )

        detections = detector.detect(
            rgb_image
        )

        detector_ms = (
            time.perf_counter_ns()
            - detector_start
        ) / 1_000_000.0

        # -------------------------------------------------------------
        # RGB-D object localisation
        # -------------------------------------------------------------

        localisation_start = (
            time.perf_counter_ns()
        )

        observations = []

        for detection in detections:

            detection_dict = (
                detection_to_dict(
                    detection
                )
            )

            point_world = (
                localize_detection(
                    detection=(
                        detection_dict
                    ),

                    depth_image=(
                        depth_image
                    ),

                    pose=pose,

                    fx=args.fx,

                    fy=args.fy,

                    cx=args.cx,

                    cy=args.cy,

                    depth_scale=(
                        args.depth_scale
                    ),
                )
            )

            if point_world is None:
                continue

            observation = (
                ObjectObservation(
                    frame_index=(
                        frame_index
                    ),

                    timestamp=(
                        timestamp
                    ),

                    class_name=(
                        detection.class_name
                    ),

                    confidence=(
                        detection.confidence
                    ),

                    position=(
                        point_world
                    ),

                    bbox=np.array(
                        [
                            detection.x1,
                            detection.y1,
                            detection.x2,
                            detection.y2,
                        ],
                        dtype=np.float64,
                    ),
                )
            )

            observations.append(
                observation
            )

        localisation_ms = (
            time.perf_counter_ns()
            - localisation_start
        ) / 1_000_000.0

        # -------------------------------------------------------------
        # Persistent association
        # -------------------------------------------------------------

        association_start = (
            time.perf_counter_ns()
        )

        mapper.process_frame(
            observations
        )

        association_ms = (
            time.perf_counter_ns()
            - association_start
        ) / 1_000_000.0

        # -------------------------------------------------------------
        # Complete semantic keyframe update
        # -------------------------------------------------------------

        semantic_total_ms = (
            time.perf_counter_ns()
            - semantic_start
        ) / 1_000_000.0

        # -------------------------------------------------------------
        # Store statistics
        # -------------------------------------------------------------

        image_load_times.append(
            image_load_ms
        )

        detector_times.append(
            detector_ms
        )

        localisation_times.append(
            localisation_ms
        )

        association_times.append(
            association_ms
        )

        semantic_total_times.append(
            semantic_total_ms
        )

        detection_counts.append(
            len(
                detections
            )
        )

        localisation_counts.append(
            len(
                observations
            )
        )

        benchmark_rows.append(
            {
                "frame_index":
                    frame_index,

                "timestamp":
                    timestamp,

                "image_load_ms":
                    image_load_ms,

                "yolo_ms":
                    detector_ms,

                "localisation_ms":
                    localisation_ms,

                "association_ms":
                    association_ms,

                "semantic_total_ms":
                    semantic_total_ms,

                "detections":
                    len(
                        detections
                    ),

                "localized_objects":
                    len(
                        observations
                    ),

                "candidate_tracks":
                    len(
                        mapper.objects
                    ),
            }
        )

        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        print(
            f"\rKeyframe "
            f"{benchmark_index:>3}"
            f"/{len(semantic_indices)} "
            f"| YOLO "
            f"{detector_ms:>7.1f} ms "
            f"| detections "
            f"{len(detections):>2}",
            end="",
            flush=True,
        )

    print()

    # =================================================================
    # Timing statistics
    # =================================================================

    print_statistics(
        "RGB-D image loading",
        image_load_times,
    )

    print_statistics(
        "YOLO11n CPU inference",
        detector_times,
    )

    print_statistics(
        "3D object localisation",
        localisation_times,
    )

    print_statistics(
        "Persistent object association",
        association_times,
    )

    print_statistics(
        "Complete semantic keyframe update",
        semantic_total_times,
    )

    # =================================================================
    # Keyframe efficiency
    # =================================================================

    total_frames = len(
        frames
    )

    semantic_frames = (
        full_semantic_frame_count
    )

    skipped_frames = (
        total_frames
        - semantic_frames
    )

    semantic_ratio = (
        semantic_frames
        / total_frames
    )

    avoided_ratio = (
        1.0
        - semantic_ratio
    )

    invocation_reduction_factor = (
        total_frames
        / semantic_frames
        if semantic_frames > 0
        else 0.0
    )

    # =================================================================
    # Detector CPU estimate
    # =================================================================

    detector_stats = statistics(
        detector_times
    )

    mean_detector_ms = (
        detector_stats[
            "mean"
        ]
    )

    estimated_naive_detector_ms = (
        mean_detector_ms
        * total_frames
    )

    estimated_keyframe_detector_ms = (
        mean_detector_ms
        * semantic_frames
    )

    estimated_saved_detector_ms = (
        estimated_naive_detector_ms
        - estimated_keyframe_detector_ms
    )

    # =================================================================
    # Amortised semantic cost
    #
    # Only valid as a full-sequence measurement when all semantic
    # keyframes were benchmarked.
    # =================================================================

    actual_total_semantic_ms = float(
        np.sum(
            semantic_total_times
        )
    )

    if (
        len(semantic_indices)
        == semantic_frames
    ):

        amortised_semantic_ms = (
            actual_total_semantic_ms
            / total_frames
        )

    else:

        # For a partial benchmark, estimate using the mean measured
        # semantic-keyframe cost.
        mean_semantic_ms = (
            statistics(
                semantic_total_times
            )["mean"]
        )

        estimated_total_semantic_ms = (
            mean_semantic_ms
            * semantic_frames
        )

        amortised_semantic_ms = (
            estimated_total_semantic_ms
            / total_frames
        )

    # =================================================================
    # Persistent mapping result
    # =================================================================

    persistent_objects = [
        obj
        for obj in mapper.objects
        if (
            obj.observation_count
            >= args.min_observations
        )
    ]

    print()
    print(
        "Keyframe efficiency"
    )

    print(
        "-------------------"
    )

    print(
        f"Total RGB-D frames:          "
        f"{total_frames}"
    )

    print(
        f"Semantic keyframes:          "
        f"{semantic_frames}"
    )

    print(
        f"Detector frames skipped:     "
        f"{skipped_frames}"
    )

    print(
        f"Semantic workload:           "
        f"{semantic_ratio * 100:.2f}%"
    )

    print(
        f"Detector invocations avoided: "
        f"{avoided_ratio * 100:.2f}%"
    )

    print(
        f"Invocation reduction factor: "
        f"{invocation_reduction_factor:.2f}x"
    )

    print()
    print(
        "Estimated detector CPU cost"
    )

    print(
        "---------------------------"
    )

    print(
        f"YOLO on all "
        f"{total_frames} frames: "
        f"{estimated_naive_detector_ms / 1000:.2f} s"
    )

    print(
        f"YOLO on "
        f"{semantic_frames} keyframes:  "
        f"{estimated_keyframe_detector_ms / 1000:.2f} s"
    )

    print(
        f"Estimated CPU time avoided: "
        f"{estimated_saved_detector_ms / 1000:.2f} s"
    )

    print()
    print(
        f"Amortised semantic cost/input frame: "
        f"{amortised_semantic_ms:.2f} ms"
    )

    # =================================================================
    # Mapping result
    # =================================================================

    print()
    print(
        "Mapping result"
    )

    print(
        "--------------"
    )

    print(
        f"YOLO detections:       "
        f"{sum(detection_counts)}"
    )

    print(
        f"3D localisations:      "
        f"{sum(localisation_counts)}"
    )

    print(
        f"Candidate tracks:      "
        f"{len(mapper.objects)}"
    )

    print(
        f"Persistent tracks:     "
        f"{len(persistent_objects)}"
    )

    # =================================================================
    # Save benchmark results
    # =================================================================

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Per-keyframe timing CSV
    # -----------------------------------------------------------------

    csv_path = (
        output_dir
        / "semantic_pipeline_timings.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "timestamp",
                "image_load_ms",
                "yolo_ms",
                "localisation_ms",
                "association_ms",
                "semantic_total_ms",
                "detections",
                "localized_objects",
                "candidate_tracks",
            ],
        )

        writer.writeheader()

        writer.writerows(
            benchmark_rows
        )

    # -----------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------

    summary_path = (
        output_dir
        / "semantic_pipeline_summary.txt"
    )

    image_stats = statistics(
        image_load_times
    )

    localisation_stats = statistics(
        localisation_times
    )

    association_stats = statistics(
        association_times
    )

    semantic_stats = statistics(
        semantic_total_times
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SemanticNav3D Semantic Pipeline Benchmark\n"
        )

        f.write(
            "========================================\n\n"
        )

        f.write(
            "Detector configuration\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            f"Model: {args.model}\n"
        )

        f.write(
            f"Confidence threshold: "
            f"{args.confidence}\n"
        )

        f.write(
            f"Image size: "
            f"{args.image_size}\n"
        )

        f.write(
            "Device: CPU\n\n"
        )

        f.write(
            "Dataset\n"
        )

        f.write(
            "-------\n"
        )

        f.write(
            f"Total RGB-D frames: "
            f"{total_frames}\n"
        )

        f.write(
            f"Semantic keyframes: "
            f"{semantic_frames}\n"
        )

        f.write(
            f"Frames benchmarked: "
            f"{len(semantic_indices)}\n\n"
        )

        f.write(
            "Timing\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            f"Mean RGB-D load: "
            f"{image_stats['mean']:.3f} ms\n"
        )

        f.write(
            f"Mean YOLO CPU inference: "
            f"{detector_stats['mean']:.3f} ms\n"
        )

        f.write(
            f"Median YOLO CPU inference: "
            f"{detector_stats['median']:.3f} ms\n"
        )

        f.write(
            f"Mean 3D localisation: "
            f"{localisation_stats['mean']:.3f} ms\n"
        )

        f.write(
            f"Mean object association: "
            f"{association_stats['mean']:.3f} ms\n"
        )

        f.write(
            f"Mean semantic update: "
            f"{semantic_stats['mean']:.3f} ms\n\n"
        )

        f.write(
            "Keyframe efficiency\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            f"Detector frames skipped: "
            f"{skipped_frames}\n"
        )

        f.write(
            f"Semantic workload: "
            f"{semantic_ratio * 100:.2f}%\n"
        )

        f.write(
            f"Detector invocations avoided: "
            f"{avoided_ratio * 100:.2f}%\n"
        )

        f.write(
            f"Invocation reduction factor: "
            f"{invocation_reduction_factor:.2f}x\n\n"
        )

        f.write(
            "Estimated detector CPU cost\n"
        )

        f.write(
            "---------------------------\n"
        )

        f.write(
            f"YOLO on all frames: "
            f"{estimated_naive_detector_ms / 1000:.3f} s\n"
        )

        f.write(
            f"YOLO on semantic keyframes: "
            f"{estimated_keyframe_detector_ms / 1000:.3f} s\n"
        )

        f.write(
            f"Estimated CPU time avoided: "
            f"{estimated_saved_detector_ms / 1000:.3f} s\n"
        )

        f.write(
            f"Amortised semantic cost/input frame: "
            f"{amortised_semantic_ms:.3f} ms\n\n"
        )

        f.write(
            "Mapping result\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            f"YOLO detections: "
            f"{sum(detection_counts)}\n"
        )

        f.write(
            f"3D localisations: "
            f"{sum(localisation_counts)}\n"
        )

        f.write(
            f"Candidate tracks: "
            f"{len(mapper.objects)}\n"
        )

        f.write(
            f"Persistent tracks: "
            f"{len(persistent_objects)}\n"
        )

    # =================================================================
    # Finish
    # =================================================================

    print()
    print(
        f"Timing CSV:  "
        f"{csv_path}"
    )

    print(
        f"Summary:     "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()