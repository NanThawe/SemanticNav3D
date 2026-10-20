from pathlib import Path
from bisect import bisect_left

import numpy as np

from .pose import CameraPose


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert quaternion (x, y, z, w) into a 3x3 rotation matrix.
    """

    norm = np.sqrt(
        qx * qx
        + qy * qy
        + qz * qz
        + qw * qw
    )

    if norm == 0:
        raise ValueError("Quaternion has zero magnitude.")

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    rotation = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )

    return rotation


def make_transform(tx, ty, tz, qx, qy, qz, qw):
    """
    Build a 4x4 camera-to-world transformation matrix.
    """

    rotation = quaternion_to_rotation_matrix(
        qx,
        qy,
        qz,
        qw,
    )

    transform = np.eye(4, dtype=np.float64)

    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]

    return transform


def load_groundtruth(sequence_path):
    """
    Load TUM groundtruth.txt.

    Returns:
        list[CameraPose]
    """

    sequence_path = Path(sequence_path)

    groundtruth_path = (
        sequence_path / "groundtruth.txt"
    )

    if not groundtruth_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {groundtruth_path}"
        )

    poses = []

    with open(groundtruth_path, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = line.split()

            if len(values) != 8:
                continue

            (
                timestamp,
                tx,
                ty,
                tz,
                qx,
                qy,
                qz,
                qw,
            ) = map(float, values)

            transform = make_transform(
                tx,
                ty,
                tz,
                qx,
                qy,
                qz,
                qw,
            )

            pose = CameraPose(
                timestamp=timestamp,
                translation=np.array(
                    [tx, ty, tz],
                    dtype=np.float64,
                ),
                quaternion=np.array(
                    [qx, qy, qz, qw],
                    dtype=np.float64,
                ),
                T_world_camera=transform,
            )

            poses.append(pose)

    print(
        f"Loaded {len(poses)} "
        f"ground-truth camera poses."
    )

    return poses


class GroundTruthTrajectory:
    """
    Provides timestamp-based access to TUM ground-truth poses.
    """

    def __init__(self, sequence_path):

        self.poses = load_groundtruth(
            sequence_path
        )

        self.timestamps = [
            pose.timestamp
            for pose in self.poses
        ]

    def nearest_pose(
        self,
        timestamp,
        max_difference=0.02,
    ):
        """
        Find the ground-truth pose closest to a frame timestamp.
        """

        index = bisect_left(
            self.timestamps,
            timestamp,
        )

        candidates = []

        if index < len(self.poses):
            candidates.append(
                self.poses[index]
            )

        if index > 0:
            candidates.append(
                self.poses[index - 1]
            )

        if not candidates:
            return None

        nearest = min(
            candidates,
            key=lambda pose: abs(
                pose.timestamp - timestamp
            ),
        )

        difference = abs(
            nearest.timestamp - timestamp
        )

        if difference > max_difference:
            return None

        return nearest