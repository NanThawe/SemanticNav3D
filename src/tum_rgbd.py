from pathlib import Path

import cv2
import numpy as np

from .frame_source import CameraIntrinsics, RGBDFrame


# TUM RGB-D camera calibration parameters
TUM_INTRINSICS = {
    "freiburg1": CameraIntrinsics(
        fx=517.3,
        fy=516.5,
        cx=318.6,
        cy=255.3,
    ),

    "freiburg2": CameraIntrinsics(
        fx=520.9,
        fy=521.0,
        cx=325.1,
        cy=249.7,
    ),

    "freiburg3": CameraIntrinsics(
        fx=535.4,
        fy=539.2,
        cx=320.1,
        cy=247.6,
    ),
}


def read_timestamp_file(file_path: Path):
    """
    Read a TUM timestamp file such as rgb.txt or depth.txt.

    Returns:
        list of tuples:
            [(timestamp, relative_file_path), ...]
    """

    entries = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Ignore comments and empty lines
            if not line or line.startswith("#"):
                continue

            timestamp, relative_path = line.split()

            entries.append(
                (float(timestamp), relative_path)
            )

    return entries


def associate_frames(
    rgb_entries,
    depth_entries,
    max_difference=0.02,
):
    """
    Match RGB and depth frames using their timestamps.

    TUM's RGB and depth sensors are not guaranteed to have
    exactly identical timestamps, so we associate each RGB
    frame with the nearest depth frame.
    """

    matches = []

    depth_index = 0

    for rgb_timestamp, rgb_path in rgb_entries:

        while (
            depth_index + 1 < len(depth_entries)
            and abs(
                depth_entries[depth_index + 1][0]
                - rgb_timestamp
            )
            < abs(
                depth_entries[depth_index][0]
                - rgb_timestamp
            )
        ):
            depth_index += 1

        depth_timestamp, depth_path = depth_entries[depth_index]

        difference = abs(
            rgb_timestamp - depth_timestamp
        )

        if difference <= max_difference:
            matches.append(
                (
                    rgb_timestamp,
                    rgb_path,
                    depth_timestamp,
                    depth_path,
                )
            )

    return matches


def get_tum_intrinsics(sequence_path: Path):
    """
    Determine which Freiburg camera calibration belongs
    to this sequence.
    """

    sequence_name = sequence_path.name.lower()

    for camera_name, intrinsics in TUM_INTRINSICS.items():
        if camera_name in sequence_name:
            return intrinsics

    raise ValueError(
        f"Could not determine camera intrinsics from "
        f"sequence name: {sequence_path.name}"
    )


class TUMRGBDSource:
    """
    Reader for TUM RGB-D sequences.
    """

    DEPTH_SCALE = 5000.0

    def __init__(
        self,
        sequence_path,
        max_timestamp_difference=0.02,
    ):
        self.sequence_path = Path(sequence_path)

        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence not found: {self.sequence_path}"
            )

        self.intrinsics = get_tum_intrinsics(
            self.sequence_path
        )

        rgb_file = self.sequence_path / "rgb.txt"
        depth_file = self.sequence_path / "depth.txt"

        if not rgb_file.exists():
            raise FileNotFoundError(rgb_file)

        if not depth_file.exists():
            raise FileNotFoundError(depth_file)

        rgb_entries = read_timestamp_file(rgb_file)
        depth_entries = read_timestamp_file(depth_file)

        self.matches = associate_frames(
            rgb_entries,
            depth_entries,
            max_difference=max_timestamp_difference,
        )

        if not self.matches:
            raise RuntimeError(
                "No RGB-depth frame pairs could be associated."
            )

        print(
            f"Loaded {len(self.matches)} "
            f"synchronised RGB-D frames."
        )

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        (
            rgb_timestamp,
            rgb_relative_path,
            _,
            depth_relative_path,
        ) = self.matches[index]

        rgb_path = (
            self.sequence_path / rgb_relative_path
        )

        depth_path = (
            self.sequence_path / depth_relative_path
        )

        rgb = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        depth_raw = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if rgb is None:
            raise RuntimeError(
                f"Could not load RGB frame: {rgb_path}"
            )

        if depth_raw is None:
            raise RuntimeError(
                f"Could not load depth frame: {depth_path}"
            )

        # TUM stores depth as integer values where
        # 5000 units = 1 metre.
        depth_meters = (
            depth_raw.astype(np.float32)
            / self.DEPTH_SCALE
        )

        return RGBDFrame(
            index=index,
            timestamp=rgb_timestamp,
            rgb=rgb,
            depth=depth_meters,
            intrinsics=self.intrinsics,
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]