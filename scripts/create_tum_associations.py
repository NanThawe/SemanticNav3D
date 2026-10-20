import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.tum_rgbd import read_timestamp_file, associate_frames


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        required=True,
        help="Path to TUM RGB-D sequence",
    )

    args = parser.parse_args()

    sequence = Path(args.sequence)

    rgb_entries = read_timestamp_file(
        sequence / "rgb.txt"
    )

    depth_entries = read_timestamp_file(
        sequence / "depth.txt"
    )

    matches = associate_frames(
        rgb_entries,
        depth_entries,
        max_difference=0.02,
    )

    output_path = (
        sequence / "associations.txt"
    )

    with open(output_path, "w") as file:

        for (
            rgb_timestamp,
            rgb_path,
            depth_timestamp,
            depth_path,
        ) in matches:

            file.write(
                f"{rgb_timestamp:.6f} "
                f"{rgb_path} "
                f"{depth_timestamp:.6f} "
                f"{depth_path}\n"
            )

    print(
        f"Created {output_path}"
    )

    print(
        f"Associated RGB-D pairs: {len(matches)}"
    )


if __name__ == "__main__":
    main()