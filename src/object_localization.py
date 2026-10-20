from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectPosition3D:
    x_camera: float
    y_camera: float
    z_camera: float

    x_world: float
    y_world: float
    z_world: float

    depth: float
    pixel_u: int
    pixel_v: int


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    quaternion = [qx, qy, qz, qw]
    """

    qx, qy, qz, qw = quaternion

    norm = np.linalg.norm(quaternion)

    if norm == 0:
        raise ValueError("Invalid zero quaternion.")

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx*qy - qz*qw),
                2 * (qx*qz + qy*qw),
            ],
            [
                2 * (qx*qy + qz*qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy*qz - qx*qw),
            ],
            [
                2 * (qx*qz - qy*qw),
                2 * (qy*qz + qx*qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )


def estimate_object_position(
    depth_image: np.ndarray,
    bbox,
    intrinsics,
    camera_position: np.ndarray,
    camera_quaternion: np.ndarray,
    crop_fraction: float = 0.60,
    min_depth: float = 0.20,
    max_depth: float = 8.0,
):
    """
    Estimate one representative 3D point for an object.

    Uses only the central region of the detection box to reduce
    background contamination.

    depth_image must already be in metres.
    """

    x1, y1, x2, y2 = bbox

    height, width = depth_image.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width - 1, int(x2))
    y2 = min(height - 1, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # --------------------------------------------------
    # Shrink detection box towards its centre
    # --------------------------------------------------

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    box_width = (x2 - x1) * crop_fraction
    box_height = (y2 - y1) * crop_fraction

    sx1 = max(0, int(cx - box_width / 2))
    sx2 = min(width, int(cx + box_width / 2))

    sy1 = max(0, int(cy - box_height / 2))
    sy2 = min(height, int(cy + box_height / 2))

    depth_crop = depth_image[
        sy1:sy2,
        sx1:sx2,
    ]

    if depth_crop.size == 0:
        return None

    # --------------------------------------------------
    # Select valid depth measurements
    # --------------------------------------------------

    valid_mask = (
        np.isfinite(depth_crop)
        & (depth_crop > min_depth)
        & (depth_crop < max_depth)
    )

    if not np.any(valid_mask):
        return None

    valid_depths = depth_crop[valid_mask]

    # Initial robust depth estimate
    median_depth = np.median(valid_depths)

    # --------------------------------------------------
    # Reject strong depth outliers using MAD
    # --------------------------------------------------

    absolute_deviation = np.abs(
        valid_depths - median_depth
    )

    mad = np.median(
        absolute_deviation
    )

    threshold = max(
        0.05,
        2.5 * mad,
    )

    filtered_depths = valid_depths[
        absolute_deviation <= threshold
    ]

    if len(filtered_depths) > 0:
        median_depth = float(
            np.median(filtered_depths)
        )

    # --------------------------------------------------
    # Find an actual pixel whose depth is closest
    # to the representative depth
    # --------------------------------------------------

        # --------------------------------------------------
    # Use the geometric centre of the detection as the
    # representative image location.
    #
    # Depth is estimated robustly from the central ROI,
    # but the pixel location itself should remain stable
    # across frames.
    # --------------------------------------------------

    u = int(round(cx))
    v = int(round(cy))

    u = np.clip(
        u,
        0,
        width - 1,
    )

    v = np.clip(
        v,
        0,
        height - 1,
    )

    z = float(
        median_depth
    )

    # --------------------------------------------------
    # Pixel → camera coordinates
    #
    # X right
    # Y down
    # Z forward
    # --------------------------------------------------

    x_camera = (
        (u - intrinsics.cx)
        * z
        / intrinsics.fx
    )

    y_camera = (
        (v - intrinsics.cy)
        * z
        / intrinsics.fy
    )

    point_camera = np.array(
        [
            x_camera,
            y_camera,
            z,
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------
    # Camera coordinates → world coordinates
    # --------------------------------------------------

    R_world_camera = (
        quaternion_to_rotation_matrix(
            camera_quaternion
        )
    )

    point_world = (
        R_world_camera
        @ point_camera
        + camera_position
    )

    return ObjectPosition3D(
        x_camera=float(point_camera[0]),
        y_camera=float(point_camera[1]),
        z_camera=float(point_camera[2]),

        x_world=float(point_world[0]),
        y_world=float(point_world[1]),
        z_world=float(point_world[2]),

        depth=z,
        pixel_u=u,
        pixel_v=v,
    )