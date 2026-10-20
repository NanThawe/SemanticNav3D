from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.object_map import (
    ObjectObservation,
    PersistentObjectMap,
)

from src.robot_relative_query import (
    CameraTrajectory,
)


# =====================================================================
# TUM sequence loading
# =====================================================================

def load_tum_associations(
    sequence_dir: Path,
) -> list[dict]:

    association_path = (
        sequence_dir
        / "associations.txt"
    )

    if not association_path.exists():
        raise FileNotFoundError(
            f"Could not find: {association_path}"
        )

    frames = []

    with association_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            values = line.split()

            if len(values) < 4:
                continue

            rgb_timestamp = float(
                values[0]
            )

            rgb_relative = values[1]

            depth_timestamp = float(
                values[2]
            )

            depth_relative = values[3]

            frames.append(
                {
                    "frame_index": len(frames),
                    "timestamp": rgb_timestamp,
                    "depth_timestamp":
                        depth_timestamp,

                    "rgb_path":
                        sequence_dir
                        / rgb_relative,

                    "depth_path":
                        sequence_dir
                        / depth_relative,
                }
            )

    return frames


# =====================================================================
# Semantic keyframes
# =====================================================================

def load_keyframes(
    csv_path: Path,
) -> set[int]:

    keyframes = set()

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return keyframes

        possible_names = [
            "frame_index",
            "rgbd_frame_index",
            "index",
        ]

        index_column = None

        for name in possible_names:

            if name in reader.fieldnames:
                index_column = name
                break

        if index_column is None:
            raise ValueError(
                "Could not determine frame-index "
                f"column in {csv_path}. "
                f"Columns: {reader.fieldnames}"
            )

        for row in reader:

            keyframes.add(
                int(
                    row[
                        index_column
                    ]
                )
            )

    return keyframes


# =====================================================================
# Cached detections
# =====================================================================

def load_cached_detections(
    csv_path: Path,
) -> dict[int, list[dict]]:

    detections = defaultdict(
        list
    )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            frame_index = int(
                row["frame_index"]
            )

            detections[
                frame_index
            ].append(
                {
                    "class_name":
                        row["class_name"],

                    "confidence":
                        float(
                            row["confidence"]
                        ),

                    "x1":
                        float(row["x1"]),

                    "y1":
                        float(row["y1"]),

                    "x2":
                        float(row["x2"]),

                    "y2":
                        float(row["y2"]),
                }
            )

    return detections


# =====================================================================
# Live YOLO detector
# =====================================================================

class LiveYOLODetector:

    def __init__(
        self,
        model_path: str,
        confidence: float,
    ):

        from ultralytics import YOLO

        print(
            f"Loading YOLO model: "
            f"{model_path}"
        )

        self.model = YOLO(
            model_path
        )

        self.confidence = float(
            confidence
        )


    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[list[dict], float]:

        start = time.perf_counter()

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        detections = []

        if not results:
            return detections, elapsed_ms

        result = results[0]

        names = result.names

        if result.boxes is None:
            return detections, elapsed_ms

        boxes = result.boxes

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )

        classes = (
            boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for (
            bbox,
            confidence,
            class_index,
        ) in zip(
            xyxy,
            confidences,
            classes,
        ):

            x1, y1, x2, y2 = bbox

            detections.append(
                {
                    "class_name":
                        str(
                            names[
                                class_index
                            ]
                        ),

                    "confidence":
                        float(
                            confidence
                        ),

                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

        return detections, elapsed_ms


# =====================================================================
# Lightweight object 3D localisation
# =====================================================================

def localize_detection(
    detection: dict,
    depth_image: np.ndarray,
    pose,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 5000.0,
    roi_fraction: float = 0.60,
    min_depth: float = 0.20,
    max_depth: float = 8.0,
):
    """
    Localise one detected object using:

        central bbox region
        -> robust median depth
        -> bbox-centre backprojection
        -> camera-to-world transform

    This follows the stable localisation strategy used by the project.
    """

    height, width = (
        depth_image.shape[:2]
    )

    x1 = float(
        detection["x1"]
    )

    y1 = float(
        detection["y1"]
    )

    x2 = float(
        detection["x2"]
    )

    y2 = float(
        detection["y2"]
    )

    # -------------------------------------------------------------
    # Clamp bbox
    # -------------------------------------------------------------

    x1 = np.clip(
        x1,
        0,
        width - 1,
    )

    x2 = np.clip(
        x2,
        0,
        width - 1,
    )

    y1 = np.clip(
        y1,
        0,
        height - 1,
    )

    y2 = np.clip(
        y2,
        0,
        height - 1,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    # -------------------------------------------------------------
    # Central ROI
    # -------------------------------------------------------------

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    margin_fraction = (
        1.0
        - roi_fraction
    ) / 2.0

    sx1 = int(
        round(
            x1
            + margin_fraction
            * box_width
        )
    )

    sx2 = int(
        round(
            x2
            - margin_fraction
            * box_width
        )
    )

    sy1 = int(
        round(
            y1
            + margin_fraction
            * box_height
        )
    )

    sy2 = int(
        round(
            y2
            - margin_fraction
            * box_height
        )
    )

    sx1 = int(
        np.clip(
            sx1,
            0,
            width - 1,
        )
    )

    sx2 = int(
        np.clip(
            sx2,
            sx1 + 1,
            width,
        )
    )

    sy1 = int(
        np.clip(
            sy1,
            0,
            height - 1,
        )
    )

    sy2 = int(
        np.clip(
            sy2,
            sy1 + 1,
            height,
        )
    )

    depth_crop = (
        depth_image[
            sy1:sy2,
            sx1:sx2,
        ].astype(
            np.float64
        )
        / depth_scale
    )

    valid = (
        np.isfinite(
            depth_crop
        )
        & (
            depth_crop
            >= min_depth
        )
        & (
            depth_crop
            <= max_depth
        )
    )

    valid_values = (
        depth_crop[
            valid
        ]
    )

    if len(valid_values) < 5:
        return None

    # -------------------------------------------------------------
    # Robust depth
    # -------------------------------------------------------------

    median_depth = float(
        np.median(
            valid_values
        )
    )

    absolute_deviation = (
        np.abs(
            valid_values
            - median_depth
        )
    )

    mad = float(
        np.median(
            absolute_deviation
        )
    )

    if mad > 1e-6:

        robust_sigma = (
            1.4826
            * mad
        )

        filtered_values = (
            valid_values[
                absolute_deviation
                <= (
                    3.0
                    * robust_sigma
                )
            ]
        )

        if len(
            filtered_values
        ) >= 5:

            median_depth = float(
                np.median(
                    filtered_values
                )
            )

    # -------------------------------------------------------------
    # Stable representative image location:
    # bbox centre
    # -------------------------------------------------------------

    u = float(
        (
            x1 + x2
        )
        / 2.0
    )

    v = float(
        (
            y1 + y2
        )
        / 2.0
    )

    z = median_depth

    x_camera = (
        (
            u - cx
        )
        * z
        / fx
    )

    y_camera = (
        (
            v - cy
        )
        * z
        / fy
    )

    point_camera = np.array(
        [
            x_camera,
            y_camera,
            z,
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # Twc:
    #
    # p_world =
    #     R_world_camera @ p_camera
    #     + camera_position
    # -------------------------------------------------------------

    point_world = (
        pose.rotation
        @ point_camera
        + pose.translation
    )

    return point_world


# =====================================================================
# Current graph construction
# =====================================================================

def get_persistent_objects(
    mapper: PersistentObjectMap,
    min_observations: int,
):

    return [
        obj
        for obj in mapper.objects
        if (
            obj.observation_count
            >= min_observations
        )
    ]


def build_current_edges(
    objects,
    very_near_distance: float,
    near_distance: float,
):

    edges = []

    for i in range(
        len(objects)
    ):

        object_a = objects[i]

        for j in range(
            i + 1,
            len(objects),
        ):

            object_b = objects[j]

            distance = float(
                np.linalg.norm(
                    object_a.position
                    - object_b.position
                )
            )

            if (
                distance
                <= very_near_distance
            ):

                relation = (
                    "VERY_NEAR"
                )

            elif (
                distance
                <= near_distance
            ):

                relation = "NEAR"

            else:
                continue

            edges.append(
                (
                    object_a,
                    object_b,
                    relation,
                    distance,
                )
            )

    return edges


# =====================================================================
# Camera-relative guidance
# =====================================================================

def relative_to_camera(
    point_world: np.ndarray,
    pose,
):

    point_camera = (
        pose.rotation.T
        @ (
            point_world
            - pose.translation
        )
    )

    x = float(
        point_camera[0]
    )

    y = float(
        point_camera[1]
    )

    z = float(
        point_camera[2]
    )

    distance = float(
        np.linalg.norm(
            point_camera
        )
    )

    bearing = float(
        np.degrees(
            np.arctan2(
                x,
                z,
            )
        )
    )

    return {
        "x_right_m": x,
        "y_down_m": y,
        "z_forward_m": z,
        "distance_m": distance,
        "bearing_deg": bearing,
    }


def find_nearest_target(
    objects,
    target_class: str,
    pose,
):

    target_class = (
        target_class
        .strip()
        .lower()
    )

    candidates = []

    for obj in objects:

        if (
            obj.class_name.lower()
            != target_class
        ):
            continue

        relative = (
            relative_to_camera(
                obj.position,
                pose,
            )
        )

        candidates.append(
            (
                relative[
                    "distance_m"
                ],
                obj,
                relative,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item[0]
    )

    return candidates[0]


def guidance_action(
    relative: dict,
    heading_tolerance: float,
    stop_distance: float,
):

    distance = (
        relative[
            "distance_m"
        ]
    )

    bearing = (
        relative[
            "bearing_deg"
        ]
    )

    z_forward = (
        relative[
            "z_forward_m"
        ]
    )

    if (
        distance
        <= stop_distance
    ):
        return "ARRIVED"

    if z_forward <= 0:
        return "TURN_AROUND"

    if (
        bearing
        < -heading_tolerance
    ):
        return "TURN_LEFT"

    if (
        bearing
        > heading_tolerance
    ):
        return "TURN_RIGHT"

    return "FORWARD"


def target_neighbours(
    target,
    edges,
    limit: int = 5,
):

    neighbours = []

    for (
        object_a,
        object_b,
        relation,
        distance,
    ) in edges:

        if (
            object_a.object_id
            == target.object_id
        ):

            neighbour = object_b

        elif (
            object_b.object_id
            == target.object_id
        ):

            neighbour = object_a

        else:
            continue

        neighbours.append(
            (
                distance,
                neighbour,
                relation,
            )
        )

    neighbours.sort(
        key=lambda item:
            item[0]
    )

    return neighbours[
        :limit
    ]


# =====================================================================
# Drawing helpers
# =====================================================================

def draw_text(
    image,
    text,
    x,
    y,
    scale=0.55,
    thickness=1,
    color=(240, 240, 240),
):

    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    image,
    detections,
):

    for detection in detections:

        x1 = int(
            detection["x1"]
        )

        y1 = int(
            detection["y1"]
        )

        x2 = int(
            detection["x2"]
        )

        y2 = int(
            detection["y2"]
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            1,
        )

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        text_y = max(
            18,
            y1 - 6,
        )

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )


def create_demo_frame(
    rgb_image,
    frame_index,
    total_frames,
    timestamp,
    is_semantic_keyframe,
    detections,
    mode,
    mapper,
    persistent_objects,
    edges,
    target_class,
    target_result,
    target_nearby,
    guidance_enabled,
    show_boxes,
    last_detector_ms,
    last_semantic_ms,
    paused,
    presentation=False,
):

    image = rgb_image.copy()

    # -------------------------------------------------------------
    # Draw detections only on actual semantic keyframes.
    # Do not carry stale 2D boxes across camera motion.
    # -------------------------------------------------------------

    if (
        show_boxes
        and is_semantic_keyframe
    ):
        draw_detections(
            image,
            detections,
        )

    height, width = image.shape[:2]

    # Slightly wider than before so text never crowds.
    panel_width = 500

    canvas = np.zeros(
        (
            height,
            width + panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :height,
        :width,
    ] = image

    canvas[
        :,
        width:,
    ] = 24

    panel_x = width + 24

    # =============================================================
    # LEFT VIDEO OVERLAY
    # =============================================================

    # SLAM is always conceptually active.
    draw_text(
        canvas,
        "SLAM ACTIVE",
        14,
        27,
        scale=0.58,
        thickness=2,
        color=(240, 240, 240),
    )

    # Semantic inference is an event, not a different system mode.
    if is_semantic_keyframe:

        draw_text(
            canvas,
            "SEMANTIC UPDATE",
            14,
            52,
            scale=0.48,
            thickness=2,
            color=(0, 230, 255),
        )

    # Frame counter
    draw_text(
        canvas,
        f"Frame {frame_index + 1} / {total_frames}",
        14,
        height - 20,
        scale=0.48,
        color=(220, 220, 220),
    )

    # Small playback progress bar
    progress = (
        (frame_index + 1)
        / max(total_frames, 1)
    )

    bar_x1 = 14
    bar_y = height - 10
    bar_width = max(
        1,
        width - 28,
    )

    cv2.rectangle(
        canvas,
        (bar_x1, bar_y),
        (
            bar_x1 + bar_width,
            bar_y + 3,
        ),
        (70, 70, 70),
        -1,
    )

    cv2.rectangle(
        canvas,
        (bar_x1, bar_y),
        (
            bar_x1
            + int(
                bar_width
                * progress
            ),
            bar_y + 3,
        ),
        (220, 220, 220),
        -1,
    )

    if paused:

        draw_text(
            canvas,
            "PAUSED",
            width - 88,
            27,
            scale=0.55,
            thickness=2,
            color=(0, 230, 255),
        )

    # =============================================================
    # RIGHT PANEL HEADER
    # =============================================================

    y = 31

    draw_text(
        canvas,
        "SemanticNav3D",
        panel_x,
        y,
        scale=0.78,
        thickness=2,
        color=(245, 245, 245),
    )

    y += 25

    draw_text(
        canvas,
        "CPU semantic spatial mapping",
        panel_x,
        y,
        scale=0.40,
        color=(165, 165, 165),
    )

    y += 27

    cv2.line(
        canvas,
        (panel_x, y),
        (
            width
            + panel_width
            - 24,
            y,
        ),
        (70, 70, 70),
        1,
    )

    # =============================================================
    # SYSTEM STATE
    # =============================================================

    y += 25

    draw_text(
        canvas,
        "SYSTEM STATE",
        panel_x,
        y,
        scale=0.49,
        thickness=2,
        color=(210, 210, 210),
    )

    y += 23

    semantic_status = (
        "YOLO + RGB-D update"
        if is_semantic_keyframe
        else "SLAM tracking"
    )

    if mode == "replay":
        backend = "Cached YOLO"
    else:
        backend = "YOLO CPU"

    system_lines = [
        (
            "Semantic backend",
            backend,
        ),
        (
            "Current stage",
            semantic_status,
        ),
        (
            "Mapped objects",
            str(
                len(
                    persistent_objects
                )
            ),
        ),
        (
            "Scene graph edges",
            str(
                len(
                    edges
                )
            ),
        ),
    ]

    for label, value in system_lines:

        draw_text(
            canvas,
            f"{label}:",
            panel_x,
            y,
            scale=0.43,
            color=(170, 170, 170),
        )

        draw_text(
            canvas,
            value,
            panel_x + 175,
            y,
            scale=0.45,
            color=(238, 238, 238),
        )

        y += 21

    # Only show timing where it is actually meaningful.
    if (
        is_semantic_keyframe
        and last_semantic_ms
        is not None
    ):

        if (
            mode == "live-semantic"
            and last_detector_ms
            is not None
        ):
            timing_value = (
                f"{last_detector_ms:.1f} ms YOLO"
            )

        else:
            timing_value = (
                f"{last_semantic_ms:.1f} ms update"
            )

        draw_text(
            canvas,
            "Last semantic:",
            panel_x,
            y,
            scale=0.43,
            color=(170, 170, 170),
        )

        draw_text(
            canvas,
            timing_value,
            panel_x + 175,
            y,
            scale=0.45,
            color=(238, 238, 238),
        )

        y += 21

    # =============================================================
    # TARGET SECTION
    # =============================================================

    y += 10

    cv2.line(
        canvas,
        (panel_x, y),
        (
            width
            + panel_width
            - 24,
            y,
        ),
        (70, 70, 70),
        1,
    )

    y += 24

    draw_text(
        canvas,
        "SEMANTIC TARGET",
        panel_x,
        y,
        scale=0.49,
        thickness=2,
        color=(210, 210, 210),
    )

    y += 24

    draw_text(
        canvas,
        "Requested:",
        panel_x,
        y,
        scale=0.43,
        color=(170, 170, 170),
    )

    draw_text(
        canvas,
        target_class,
        panel_x + 175,
        y,
        scale=0.48,
        color=(238, 238, 238),
    )

    y += 22

    # -------------------------------------------------------------
    # Target not yet persistent
    # -------------------------------------------------------------

    if target_result is None:

        draw_text(
            canvas,
            "Waiting for persistent target...",
            panel_x,
            y,
            scale=0.47,
            color=(150, 190, 230),
        )

        y += 32

    # -------------------------------------------------------------
    # Target available
    # -------------------------------------------------------------

    else:

        (
            _,
            target,
            relative,
            action,
        ) = target_result

        target_id = (
            f"{target.class_name}_"
            f"{target.object_id}"
        )

        target_lines = [
            (
                "Object",
                target_id,
            ),
            (
                "Distance",
                f"{relative['distance_m']:.2f} m",
            ),
            (
                "Bearing",
                f"{relative['bearing_deg']:+.1f} deg",
            ),
        ]

        for label, value in target_lines:

            draw_text(
                canvas,
                f"{label}:",
                panel_x,
                y,
                scale=0.43,
                color=(170, 170, 170),
            )

            draw_text(
                canvas,
                value,
                panel_x + 175,
                y,
                scale=0.47,
                color=(238, 238, 238),
            )

            y += 21

        if guidance_enabled:

            y += 5

            draw_text(
                canvas,
                "GUIDANCE",
                panel_x,
                y,
                scale=0.43,
                color=(170, 170, 170),
            )

            draw_text(
                canvas,
                action,
                panel_x + 175,
                y,
                scale=0.68,
                thickness=2,
                color=(0, 230, 255),
            )

            y += 32

    # =============================================================
    # LOCAL GRAPH CONTEXT
    # =============================================================

    if (
        target_result is not None
        and target_nearby
    ):

        y += 3

        cv2.line(
            canvas,
            (panel_x, y),
            (
                width
                + panel_width
                - 24,
                y,
            ),
            (70, 70, 70),
            1,
        )

        y += 23

        draw_text(
            canvas,
            "NEAR TARGET",
            panel_x,
            y,
            scale=0.47,
            thickness=2,
            color=(210, 210, 210),
        )

        y += 22

        # Three neighbours are enough for the presentation.
        for (
            distance,
            neighbour,
            relation,
        ) in target_nearby[:3]:

            node_id = (
                f"{neighbour.class_name}_"
                f"{neighbour.object_id}"
            )

            draw_text(
                canvas,
                node_id,
                panel_x,
                y,
                scale=0.42,
                color=(225, 225, 225),
            )

            draw_text(
                canvas,
                f"{distance:.2f} m",
                panel_x + 205,
                y,
                scale=0.42,
                color=(170, 170, 170),
            )

            draw_text(
                canvas,
                relation,
                panel_x + 285,
                y,
                scale=0.37,
                color=(145, 145, 145),
            )

            y += 19

    # =============================================================
    # CONTROLS
    #
    # Hidden completely in recruiter / recorded presentation mode.
    # =============================================================

    if not presentation:

        controls_y = height - 56

        cv2.line(
            canvas,
            (
                panel_x,
                controls_y - 16,
            ),
            (
                width
                + panel_width
                - 24,
                controls_y - 16,
            ),
            (70, 70, 70),
            1,
        )

        draw_text(
            canvas,
            "SPACE pause   N next   B boxes   G guidance",
            panel_x,
            controls_y,
            scale=0.35,
            color=(135, 135, 135),
        )

        draw_text(
            canvas,
            "1 cup   2 keyboard   3 laptop   C cycle   ESC quit",
            panel_x,
            controls_y + 18,
            scale=0.35,
            color=(135, 135, 135),
        )

    return canvas

def create_end_card(
    width: int,
    height: int,
):
    """
    Recruiter-facing final card for the recorded demo.
    """

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    centre_x = width // 2

    def centred_text(
        text,
        y,
        scale,
        thickness=1,
        color=(235, 235, 235),
    ):
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )

        x = int(
            centre_x
            - size[0] / 2
        )

        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------
    # Title
    # -------------------------------------------------------------

    centred_text(
        "SemanticNav3D",
        80,
        1.15,
        thickness=2,
    )

    centred_text(
        "CPU-efficient semantic spatial mapping",
        115,
        0.52,
        color=(175, 175, 175),
    )

    # -------------------------------------------------------------
    # Divider
    # -------------------------------------------------------------

    cv2.line(
        canvas,
        (width // 4, 145),
        (3 * width // 4, 145),
        (80, 80, 80),
        1,
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    centred_text(
        "596 RGB-D frames",
        190,
        0.60,
    )

    centred_text(
        "147 semantic keyframes",
        225,
        0.60,
    )

    centred_text(
        "75.3% fewer detector invocations",
        260,
        0.60,
    )

    centred_text(
        "ORB-SLAM3 ATE RMSE: 1.68 cm",
        310,
        0.55,
    )

    centred_text(
        "Object-map median RMSE: 6.4 cm",
        345,
        0.55,
    )

    centred_text(
        "Scene-graph queries: < 0.4 ms CPU",
        380,
        0.55,
    )

    # -------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------

    cv2.line(
        canvas,
        (width // 4, 415),
        (3 * width // 4, 415),
        (80, 80, 80),
        1,
    )

    centred_text(
        "RGB-D  |  SLAM  |  3D Perception  |  Scene Graphs",
        455,
        0.47,
        color=(175, 175, 175),
    )

    return canvas
# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Interactive TUM RGB-D demonstration "
            "of SemanticNav3D."
        )
    )

    # -------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------

    parser.add_argument(
        "--sequence",
        default=(
            "data/tum/"
            "rgbd_dataset_freiburg1_desk"
        ),
    )

    parser.add_argument(
        "--end-card-seconds",
        type=float,
        default=3.0,
        help=(
            "Duration of the final results card "
            "when recording."
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

    parser.add_argument(
        "--detections",
        default=(
            "results/detections/"
            "freiburg1_desk/"
            "detections.csv"
        ),
    )

    # -------------------------------------------------------------
    # Demo mode
    # -------------------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=[
            "replay",
            "live-semantic",
        ],
        default="replay",
    )

    parser.add_argument(
        "--presentation",
        action="store_true",
        help=(
            "Hide interactive controls and use "
            "a cleaner recruiter-facing layout."
        ),
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    # -------------------------------------------------------------
    # Camera intrinsics
    #
    # TUM Freiburg1 calibrated intrinsics.
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Mapping
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Scene graph
    # -------------------------------------------------------------

    parser.add_argument(
        "--very-near-distance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--near-distance",
        type=float,
        default=0.40,
    )

    # -------------------------------------------------------------
    # Guidance
    # -------------------------------------------------------------

    parser.add_argument(
        "--target-class",
        default="cup",
    )

    parser.add_argument(
        "--heading-tolerance",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--stop-distance",
        type=float,
        default=0.50,
    )

    # -------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------
    last_canvas = None
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--record",
        default=None,
        help=(
            "Optional output .mp4 path "
            "for recording the demo."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    sequence_dir = Path(
        args.sequence
    )

    trajectory_path = Path(
        args.trajectory
    )

    keyframes_path = Path(
        args.keyframes
    )

    detections_path = Path(
        args.detections
    )

    # -------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------

    frames = load_tum_associations(
        sequence_dir
    )

    keyframes = load_keyframes(
        keyframes_path
    )

    trajectory = CameraTrajectory(
        trajectory_path
    )

    print()
    print(
        "SemanticNav3D TUM Demo"
    )

    print(
        "======================"
    )

    print(
        f"Frames:             "
        f"{len(frames)}"
    )

    print(
        f"Semantic keyframes: "
        f"{len(keyframes)}"
    )

    print(
        f"Mode:               "
        f"{args.mode}"
    )

    # -------------------------------------------------------------
    # Detection backend
    # -------------------------------------------------------------

    cached_detections = None
    live_detector = None

    if args.mode == "replay":

        cached_detections = (
            load_cached_detections(
                detections_path
            )
        )

        print(
            "Detector:           "
            "cached YOLO outputs"
        )

    else:

        live_detector = (
            LiveYOLODetector(
                model_path=(
                    args.model
                ),
                confidence=(
                    args.confidence
                ),
            )
        )

        print(
            "Detector:           "
            "YOLO CPU inference"
        )

    # -------------------------------------------------------------
    # Map state
    # -------------------------------------------------------------

    def create_mapper():

        return PersistentObjectMap(
            association_distance=(
                args.association_distance
            )
        )

    mapper = create_mapper()

    processed_frames = set()

    # -------------------------------------------------------------
    # Demo state
    # -------------------------------------------------------------

    frame_index = 0

    paused = False

    show_boxes = True

    guidance_enabled = True

    target_classes = [
        "cup",
        "keyboard",
        "laptop",
        "chair",
        "book",
        "mouse",
        "tv",
        "cell phone",
    ]

    target_class = (
        args.target_class
    )

    if (
        target_class
        not in target_classes
    ):
        target_classes.insert(
            0,
            target_class,
        )

    last_detector_ms = None
    last_semantic_ms = None

    current_detections = []

    # -------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------

    video_writer = None

    # -------------------------------------------------------------
    # Window
    # -------------------------------------------------------------

    window_name = (
        "SemanticNav3D"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    # -------------------------------------------------------------
    # Main playback loop
    # -------------------------------------------------------------

    while (
        0
        <= frame_index
        < len(frames)
    ):

        frame_info = (
            frames[
                frame_index
            ]
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

        if rgb_image is None:
            raise RuntimeError(
                "Failed to read RGB image: "
                f"{frame_info['rgb_path']}"
            )

        if depth_image is None:
            raise RuntimeError(
                "Failed to read depth image: "
                f"{frame_info['depth_path']}"
            )

        timestamp = float(
            frame_info[
                "timestamp"
            ]
        )

        # ---------------------------------------------------------
        # Camera pose
        # ---------------------------------------------------------

        try:

            pose = (
                trajectory.nearest_timestamp(
                    timestamp,
                    max_gap=(
                        args.pose_max_gap
                    ),
                )
            )

        except ValueError as error:

            print(
                f"Pose warning at "
                f"frame {frame_index}: "
                f"{error}"
            )

            frame_index += 1
            continue

        is_semantic_keyframe = (
            frame_index
            in keyframes
        )

        # ---------------------------------------------------------
        # Process semantic frame exactly once
        # ---------------------------------------------------------

        if (
            is_semantic_keyframe
            and frame_index
            not in processed_frames
        ):

            semantic_start = (
                time.perf_counter()
            )

            # -----------------------------------------------------
            # Detection
            # -----------------------------------------------------

            if (
                args.mode
                == "replay"
            ):

                current_detections = list(
                    cached_detections.get(
                        frame_index,
                        [],
                    )
                )

                last_detector_ms = None

            else:

                (
                    current_detections,
                    last_detector_ms,
                ) = (
                    live_detector.detect(
                        rgb_image
                    )
                )

            # -----------------------------------------------------
            # 3D localisation
            # -----------------------------------------------------

            observations = []

            for detection in (
                current_detections
            ):

                point_world = (
                    localize_detection(
                        detection=(
                            detection
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
                            detection[
                                "class_name"
                            ]
                        ),

                        confidence=(
                            detection[
                                "confidence"
                            ]
                        ),

                        position=(
                            point_world
                        ),

                        bbox=np.array(
                            [
                                detection["x1"],
                                detection["y1"],
                                detection["x2"],
                                detection["y2"],
                            ],
                            dtype=np.float64,
                        ),
                    )
                )

                observations.append(
                    observation
                )

            # -----------------------------------------------------
            # Persistent association
            # -----------------------------------------------------

            mapper.process_frame(
                observations
            )

            processed_frames.add(
                frame_index
            )

            last_semantic_ms = (
                (
                    time.perf_counter()
                    - semantic_start
                )
                * 1000.0
            )

        elif not is_semantic_keyframe:

            current_detections = []

        # ---------------------------------------------------------
        # Current persistent map
        # ---------------------------------------------------------

        persistent_objects = (
            get_persistent_objects(
                mapper,
                min_observations=(
                    args.min_observations
                ),
            )
        )

        # ---------------------------------------------------------
        # Current scene graph
        # ---------------------------------------------------------

        edges = (
            build_current_edges(
                persistent_objects,

                very_near_distance=(
                    args.very_near_distance
                ),

                near_distance=(
                    args.near_distance
                ),
            )
        )

        # ---------------------------------------------------------
        # Current semantic target
        # ---------------------------------------------------------

        target_search = (
            find_nearest_target(
                persistent_objects,
                target_class,
                pose,
            )
        )

        target_result = None
        target_nearby = []

        if target_search is not None:

            (
                distance,
                target,
                relative,
            ) = target_search

            action = (
                guidance_action(
                    relative,

                    heading_tolerance=(
                        args.heading_tolerance
                    ),

                    stop_distance=(
                        args.stop_distance
                    ),
                )
            )

            target_result = (
                distance,
                target,
                relative,
                action,
            )

            target_nearby = (
                target_neighbours(
                    target,
                    edges,
                    limit=5,
                )
            )

        # ---------------------------------------------------------
        # Render
        # ---------------------------------------------------------

        canvas = create_demo_frame(
            rgb_image=rgb_image,

            frame_index=frame_index,

            total_frames=len(
                frames
            ),

            timestamp=timestamp,

            is_semantic_keyframe=(
                is_semantic_keyframe
            ),

            detections=(
                current_detections
            ),

            mode=args.mode,

            mapper=mapper,

            persistent_objects=(
                persistent_objects
            ),

            edges=edges,

            target_class=(
                target_class
            ),

            target_result=(
                target_result
            ),

            target_nearby=(
                target_nearby
            ),

            guidance_enabled=(
                guidance_enabled
            ),

            show_boxes=(
                show_boxes
            ),

            last_detector_ms=(
                last_detector_ms
            ),

            last_semantic_ms=(
                last_semantic_ms
            ),

            paused=paused,

            presentation=(
                args.presentation
            )
        )

        last_canvas = canvas

        # ---------------------------------------------------------
        # Optional recording
        # ---------------------------------------------------------

        if (
            args.record
            is not None
        ):

            if video_writer is None:

                output_path = Path(
                    args.record
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                fourcc = (
                    cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )
                )

                video_writer = (
                    cv2.VideoWriter(
                        str(
                            output_path
                        ),
                        fourcc,
                        args.fps,
                        (
                            canvas.shape[1],
                            canvas.shape[0],
                        ),
                    )
                )

            video_writer.write(
                canvas
            )

        # ---------------------------------------------------------
        # Display
        # ---------------------------------------------------------

        cv2.imshow(
            window_name,
            canvas,
        )

        if paused:

            wait_time = 0

        else:

            wait_time = max(
                1,
                int(
                    1000.0
                    / args.fps
                ),
            )

        key = (
            cv2.waitKeyEx(
                wait_time
            )
        )

        # ---------------------------------------------------------
        # Keyboard controls
        # ---------------------------------------------------------

        if key == 27:
            # ESC
            break

        elif key == ord(" "):

            paused = not paused

            continue

        elif key in [
            ord("n"),
            ord("N"),
            ord("."),
            2555904,  # right arrow on Windows
        ]:

            paused = True

            frame_index += 1

            continue

        elif key in [
            ord("b"),
            ord("B"),
        ]:

            show_boxes = (
                not show_boxes
            )

            continue

        elif key in [
            ord("g"),
            ord("G"),
        ]:

            guidance_enabled = (
                not guidance_enabled
            )

            continue

        elif key == ord("1"):

            target_class = "cup"

            continue

        elif key == ord("2"):

            target_class = (
                "keyboard"
            )

            continue

        elif key == ord("3"):

            target_class = (
                "laptop"
            )

            continue

        elif key in [
            ord("c"),
            ord("C"),
        ]:

            current_index = (
                target_classes.index(
                    target_class
                )
                if target_class
                in target_classes
                else 0
            )

            next_index = (
                current_index + 1
            ) % len(
                target_classes
            )

            target_class = (
                target_classes[
                    next_index
                ]
            )

            continue

        elif key in [
            ord("r"),
            ord("R"),
        ]:

            print(
                "Restarting demo..."
            )

            mapper = (
                create_mapper()
            )

            processed_frames.clear()

            frame_index = 0

            current_detections = []

            last_detector_ms = None

            last_semantic_ms = None

            paused = False

            continue

        # ---------------------------------------------------------
        # Normal playback advance
        # ---------------------------------------------------------

        if not paused:

            frame_index += 1

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Final recruiter-facing results card
    # -----------------------------------------------------------------

    completed_sequence = (
        frame_index
        >= len(frames)
    )

    if (
        completed_sequence
        and last_canvas is not None
    ):

        end_card = create_end_card(
            width=last_canvas.shape[1],
            height=last_canvas.shape[0],
        )

        # Show it in the interactive window as well.
        cv2.imshow(
            window_name,
            end_card,
        )

        # Append it to recorded video.
        if video_writer is not None:

            end_frames = int(
                round(
                    args.end_card_seconds
                    * args.fps
                )
            )

            for _ in range(
                max(
                    1,
                    end_frames,
                )
            ):
                video_writer.write(
                    end_card
                )

        cv2.waitKey(
            int(
                args.end_card_seconds
                * 1000
            )
        )
    if (
        video_writer
        is not None
    ):
        video_writer.release()

    cv2.destroyAllWindows()

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    persistent_objects = (
        get_persistent_objects(
            mapper,
            min_observations=(
                args.min_observations
            ),
        )
    )

    final_edges = (
        build_current_edges(
            persistent_objects,

            very_near_distance=(
                args.very_near_distance
            ),

            near_distance=(
                args.near_distance
            ),
        )
    )

    print()
    print(
        "Demo finished"
    )

    print(
        "-------------"
    )

    print(
        f"Frames processed:      "
        f"{len(processed_frames)} "
        f"semantic keyframes"
    )

    print(
        f"Candidate tracks:      "
        f"{len(mapper.objects)}"
    )

    print(
        f"Persistent objects:    "
        f"{len(persistent_objects)}"
    )

    print(
        f"Scene graph edges:     "
        f"{len(final_edges)}"
    )

    if (
        args.record
        is not None
    ):

        print(
            f"Recorded demo:         "
            f"{args.record}"
        )


if __name__ == "__main__":
    main()