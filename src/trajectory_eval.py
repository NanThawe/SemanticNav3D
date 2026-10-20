from dataclasses import dataclass
from pathlib import Path
import bisect
import numpy as np


@dataclass
class TrajectoryPose:
    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray


def load_tum_trajectory(path: str | Path) -> list[TrajectoryPose]:
    """
    Load a trajectory in TUM format:

    timestamp tx ty tz qx qy qz qw
    """
    path = Path(path)
    poses = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = line.split()

            if len(values) < 8:
                continue

            timestamp = float(values[0])

            position = np.array(
                [float(values[1]), float(values[2]), float(values[3])],
                dtype=np.float64,
            )

            quaternion = np.array(
                [
                    float(values[4]),
                    float(values[5]),
                    float(values[6]),
                    float(values[7]),
                ],
                dtype=np.float64,
            )

            poses.append(
                TrajectoryPose(
                    timestamp=timestamp,
                    position=position,
                    quaternion=quaternion,
                )
            )

    return poses


def associate_trajectories(
    estimated: list[TrajectoryPose],
    ground_truth: list[TrajectoryPose],
    max_difference: float = 0.02,
):
    """
    Match each estimated pose to the nearest ground-truth timestamp.
    """

    gt_timestamps = [pose.timestamp for pose in ground_truth]

    matched_est = []
    matched_gt = []
    matched_timestamps = []

    for est_pose in estimated:
        index = bisect.bisect_left(gt_timestamps, est_pose.timestamp)

        candidates = []

        if index < len(ground_truth):
            candidates.append(ground_truth[index])

        if index > 0:
            candidates.append(ground_truth[index - 1])

        if not candidates:
            continue

        closest = min(
            candidates,
            key=lambda pose: abs(pose.timestamp - est_pose.timestamp),
        )

        time_difference = abs(
            closest.timestamp - est_pose.timestamp
        )

        if time_difference <= max_difference:
            matched_est.append(est_pose.position)
            matched_gt.append(closest.position)
            matched_timestamps.append(est_pose.timestamp)

    return (
        np.asarray(matched_est),
        np.asarray(matched_gt),
        np.asarray(matched_timestamps),
    )


def align_trajectory(
    estimated_xyz: np.ndarray,
    ground_truth_xyz: np.ndarray,
):
    """
    Rigidly align the estimated trajectory to ground truth.

    Finds:
        p_gt ≈ R @ p_est + t

    No scale correction is applied.
    """

    est_mean = estimated_xyz.mean(axis=0)
    gt_mean = ground_truth_xyz.mean(axis=0)

    est_centered = estimated_xyz - est_mean
    gt_centered = ground_truth_xyz - gt_mean

    covariance = est_centered.T @ gt_centered

    U, _, Vt = np.linalg.svd(covariance)

    rotation = Vt.T @ U.T

    # Prevent reflection
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = gt_mean - rotation @ est_mean

    aligned = (
        rotation @ estimated_xyz.T
    ).T + translation

    return aligned, rotation, translation


def compute_ate(
    aligned_estimated: np.ndarray,
    ground_truth: np.ndarray,
):
    """
    Compute translational Absolute Trajectory Error.
    """

    differences = aligned_estimated - ground_truth

    errors = np.linalg.norm(
        differences,
        axis=1,
    )

    metrics = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }

    return errors, metrics