import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.trajectory_eval import (
    load_tum_trajectory,
    associate_trajectories,
    align_trajectory,
    compute_ate,
)


def save_error_csv(
    output_path,
    timestamps,
    estimated,
    ground_truth,
    errors,
):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "timestamp",
                "estimated_x",
                "estimated_y",
                "estimated_z",
                "groundtruth_x",
                "groundtruth_y",
                "groundtruth_z",
                "ate_m",
            ]
        )

        for timestamp, est, gt, error in zip(
            timestamps,
            estimated,
            ground_truth,
            errors,
        ):
            writer.writerow(
                [
                    timestamp,
                    *est,
                    *gt,
                    error,
                ]
            )


def plot_trajectory(
    estimated,
    ground_truth,
    output_path,
):
    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        ground_truth[:, 2],
        label="Ground truth",
    )

    ax.plot(
        estimated[:, 0],
        estimated[:, 1],
        estimated[:, 2],
        label="ORB-SLAM3",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "ORB-SLAM3 vs TUM Ground Truth"
    )

    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()


def plot_ate(
    timestamps,
    errors,
    output_path,
):
    relative_time = timestamps - timestamps[0]

    plt.figure(figsize=(10, 5))

    plt.plot(
        relative_time,
        errors,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("ATE (m)")

    plt.title(
        "Absolute Trajectory Error"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ORB-SLAM3 against TUM ground truth."
    )

    parser.add_argument(
        "--sequence",
        required=True,
        help="Path to TUM sequence.",
    )

    parser.add_argument(
        "--estimated",
        required=True,
        help="ORB-SLAM3 CameraTrajectory.txt",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--max-time-difference",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    sequence_path = Path(args.sequence)
    estimated_path = Path(args.estimated)
    output_path = Path(args.output)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    groundtruth_path = (
        sequence_path / "groundtruth.txt"
    )

    # --------------------------------------------------
    # Load trajectories
    # --------------------------------------------------

    estimated = load_tum_trajectory(
        estimated_path
    )

    ground_truth = load_tum_trajectory(
        groundtruth_path
    )

    print("\nTrajectory files")
    print("----------------")
    print(f"Estimated poses:    {len(estimated)}")
    print(f"Ground-truth poses: {len(ground_truth)}")

    # --------------------------------------------------
    # Timestamp association
    # --------------------------------------------------

    est_xyz, gt_xyz, timestamps = (
        associate_trajectories(
            estimated,
            ground_truth,
            max_difference=args.max_time_difference,
        )
    )

    print(f"Matched poses:      {len(est_xyz)}")

    if len(est_xyz) < 3:
        raise RuntimeError(
            "Not enough matched poses for trajectory evaluation."
        )

    # --------------------------------------------------
    # Align trajectories
    # --------------------------------------------------

    aligned_est, rotation, translation = (
        align_trajectory(
            est_xyz,
            gt_xyz,
        )
    )

    # --------------------------------------------------
    # Calculate ATE
    # --------------------------------------------------

    errors, metrics = compute_ate(
        aligned_est,
        gt_xyz,
    )

    print("\nAbsolute Trajectory Error")
    print("-------------------------")

    print(
        f"ATE RMSE:   {metrics['rmse']:.4f} m"
    )

    print(
        f"ATE mean:   {metrics['mean']:.4f} m"
    )

    print(
        f"ATE median: {metrics['median']:.4f} m"
    )

    print(
        f"ATE std:    {metrics['std']:.4f} m"
    )

    print(
        f"ATE min:    {metrics['min']:.4f} m"
    )

    print(
        f"ATE max:    {metrics['max']:.4f} m"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    metrics_path = (
        output_path / "ate_metrics.txt"
    )

    with metrics_path.open("w") as f:
        f.write(
            f"matched_poses: {len(est_xyz)}\n"
        )

        for name, value in metrics.items():
            f.write(
                f"{name}: {value:.6f}\n"
            )

    save_error_csv(
        output_path / "ate_errors.csv",
        timestamps,
        aligned_est,
        gt_xyz,
        errors,
    )

    plot_trajectory(
        aligned_est,
        gt_xyz,
        output_path / "trajectory_comparison.png",
    )

    plot_ate(
        timestamps,
        errors,
        output_path / "ate_over_time.png",
    )


if __name__ == "__main__":
    main()
    