from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Camera trajectory
# ---------------------------------------------------------------------

@dataclass
class CameraTrajectoryPose:
    index: int
    timestamp: float

    # Camera origin expressed in world coordinates.
    translation: np.ndarray

    # Rotation from camera frame to world frame.
    rotation: np.ndarray


def quaternion_to_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) into a 3x3 rotation matrix.

    ORB-SLAM3 CameraTrajectory.txt stores the camera-to-world pose:

        Twc = [Rwc | twc]

    Therefore this matrix maps camera-frame vectors into the
    ORB-SLAM3 world frame.
    """

    quaternion = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        quaternion
    )

    if norm <= 1e-12:
        raise ValueError(
            "Invalid zero-length quaternion."
        )

    quaternion /= norm

    qx, qy, qz, qw = quaternion

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


class CameraTrajectory:
    """
    Load an ORB-SLAM3 TUM-format CameraTrajectory.txt file.

    Expected format:

        timestamp tx ty tz qx qy qz qw

    The pose is camera-to-world.
    """

    def __init__(
        self,
        trajectory_path: str | Path,
    ):
        self.trajectory_path = Path(
            trajectory_path
        )

        self.poses = self._load()

        if not self.poses:
            raise ValueError(
                "Trajectory contains no valid poses."
            )


    def _load(
        self,
    ) -> list[CameraTrajectoryPose]:

        poses = []

        with self.trajectory_path.open(
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

                if len(values) < 8:
                    continue

                timestamp = float(
                    values[0]
                )

                tx = float(values[1])
                ty = float(values[2])
                tz = float(values[3])

                qx = float(values[4])
                qy = float(values[5])
                qz = float(values[6])
                qw = float(values[7])

                rotation = (
                    quaternion_to_rotation_matrix(
                        qx,
                        qy,
                        qz,
                        qw,
                    )
                )

                pose = CameraTrajectoryPose(
                    index=len(poses),
                    timestamp=timestamp,

                    translation=np.array(
                        [tx, ty, tz],
                        dtype=np.float64,
                    ),

                    rotation=rotation,
                )

                poses.append(
                    pose
                )

        return poses


    def latest(
        self,
    ) -> CameraTrajectoryPose:

        return self.poses[-1]


    def by_index(
        self,
        index: int,
    ) -> CameraTrajectoryPose:

        if (
            index < 0
            or index >= len(self.poses)
        ):
            raise ValueError(
                f"Pose index {index} is outside "
                f"0..{len(self.poses) - 1}"
            )

        return self.poses[
            index
        ]


    def nearest_timestamp(
        self,
        timestamp: float,
        max_gap: float | None = 0.05,
    ) -> CameraTrajectoryPose:

        pose = min(
            self.poses,
            key=lambda p:
                abs(
                    p.timestamp
                    - timestamp
                ),
        )

        gap = abs(
            pose.timestamp
            - timestamp
        )

        if (
            max_gap is not None
            and gap > max_gap
        ):
            raise ValueError(
                f"No camera pose within "
                f"{max_gap:.3f} s of "
                f"{timestamp:.6f}. "
                f"Nearest gap: {gap:.3f} s."
            )

        return pose


# ---------------------------------------------------------------------
# Robot / camera-relative scene queries
# ---------------------------------------------------------------------

class RobotRelativeQuery:
    """
    Camera-relative queries over SemanticNav3D persistent objects.

    Current frame convention:

        +x : right
        +y : down
        +z : forward

    The camera optical centre is currently used as the robot position.
    """

    def __init__(
        self,
        graph_path: str | Path,
    ):
        graph_path = Path(
            graph_path
        )

        with graph_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            graph = json.load(f)

        self.nodes = graph.get(
            "nodes",
            [],
        )

        self.nodes_by_id = {
            node["node_id"]: node
            for node in self.nodes
        }


    def _require_node(
        self,
        node_id: str,
    ) -> dict:

        if node_id not in self.nodes_by_id:
            raise ValueError(
                f"Unknown object: {node_id}"
            )

        return self.nodes_by_id[
            node_id
        ]


    @staticmethod
    def _world_position(
        node: dict,
    ) -> np.ndarray:

        return np.array(
            [
                float(
                    node["x_world"]
                ),
                float(
                    node["y_world"]
                ),
                float(
                    node["z_world"]
                ),
            ],
            dtype=np.float64,
        )


    # -----------------------------------------------------------------
    # World -> camera transform
    # -----------------------------------------------------------------

    def world_to_camera(
        self,
        point_world: np.ndarray,
        pose: CameraTrajectoryPose,
    ) -> np.ndarray:
        """
        Transform world point into the current camera frame.

        Twc:
            p_world = Rwc @ p_camera + twc

        Therefore:

            p_camera = Rwc.T @ (p_world - twc)
        """

        return (
            pose.rotation.T
            @ (
                point_world
                - pose.translation
            )
        )


    # -----------------------------------------------------------------
    # Relative object description
    # -----------------------------------------------------------------

    def relative_position(
        self,
        node_id: str,
        pose: CameraTrajectoryPose,
        centre_deadband_deg: float = 5.0,
    ) -> dict:

        node = self._require_node(
            node_id
        )

        point_world = (
            self._world_position(
                node
            )
        )

        point_camera = (
            self.world_to_camera(
                point_world,
                pose,
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

        horizontal_distance = float(
            np.sqrt(
                x * x
                + z * z
            )
        )

        bearing_deg = float(
            np.degrees(
                np.arctan2(
                    x,
                    z,
                )
            )
        )

        # Positive elevation means above camera centre.
        elevation_deg = float(
            np.degrees(
                np.arctan2(
                    -y,
                    horizontal_distance,
                )
            )
        )

        if (
            abs(bearing_deg)
            <= centre_deadband_deg
        ):
            side = "CENTRE"

        elif bearing_deg < 0:
            side = "LEFT"

        else:
            side = "RIGHT"

        if z > 0:
            longitudinal = "FRONT"
        else:
            longitudinal = "BEHIND"

        return {
            "node_id":
                node["node_id"],

            "class_name":
                node["class_name"],

            "distance_m":
                distance,

            "x_right_m":
                x,

            "y_down_m":
                y,

            "z_forward_m":
                z,

            "bearing_deg":
                bearing_deg,

            "elevation_deg":
                elevation_deg,

            "side":
                side,

            "longitudinal":
                longitudinal,

            "in_front":
                z > 0,
        }


    # -----------------------------------------------------------------
    # All object positions relative to camera
    # -----------------------------------------------------------------

    def all_relative(
        self,
        pose: CameraTrajectoryPose,
    ) -> list[dict]:

        results = [
            self.relative_position(
                node["node_id"],
                pose,
            )
            for node in self.nodes
        ]

        return sorted(
            results,
            key=lambda item:
                item["distance_m"],
        )


    # -----------------------------------------------------------------
    # Nearest object
    # -----------------------------------------------------------------

    def nearest(
        self,
        pose: CameraTrajectoryPose,
        class_name: str | None = None,
        front_only: bool = False,
    ) -> dict | None:

        results = self.all_relative(
            pose
        )

        if class_name is not None:

            class_name = (
                class_name
                .strip()
                .lower()
            )

            results = [
                result
                for result in results
                if (
                    str(
                        result[
                            "class_name"
                        ]
                    ).lower()
                    == class_name
                )
            ]

        if front_only:

            results = [
                result
                for result in results
                if result[
                    "in_front"
                ]
            ]

        if not results:
            return None

        return results[0]


    # -----------------------------------------------------------------
    # Objects inside metric radius
    # -----------------------------------------------------------------

    def within_radius(
        self,
        pose: CameraTrajectoryPose,
        radius_m: float,
        front_only: bool = False,
    ) -> list[dict]:

        if radius_m <= 0:
            raise ValueError(
                "radius_m must be positive."
            )

        results = self.all_relative(
            pose
        )

        results = [
            result
            for result in results
            if (
                result["distance_m"]
                <= radius_m
            )
        ]

        if front_only:

            results = [
                result
                for result in results
                if result[
                    "in_front"
                ]
            ]

        return results


    # -----------------------------------------------------------------
    # Objects in front of camera
    # -----------------------------------------------------------------

    def objects_in_front(
        self,
        pose: CameraTrajectoryPose,
        max_bearing_deg: float = 45.0,
        max_distance_m: float | None = None,
    ) -> list[dict]:
        """
        Approximate frontal field using horizontal bearing.

        Example:
            max_bearing_deg = 45

        keeps objects from -45 degrees to +45 degrees.
        """

        if (
            max_bearing_deg <= 0
            or max_bearing_deg > 90
        ):
            raise ValueError(
                "max_bearing_deg must be in "
                "(0, 90]."
            )

        results = self.all_relative(
            pose
        )

        filtered = []

        for result in results:

            if not result[
                "in_front"
            ]:
                continue

            if (
                abs(
                    result[
                        "bearing_deg"
                    ]
                )
                > max_bearing_deg
            ):
                continue

            if (
                max_distance_m
                is not None
                and result[
                    "distance_m"
                ]
                > max_distance_m
            ):
                continue

            filtered.append(
                result
            )

        return filtered