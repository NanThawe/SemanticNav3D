import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

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
    merge_persistent_duplicates,
)


# ---------------------------------------------------------------------
# Helper: resolve object-ID remapping chains
# ---------------------------------------------------------------------

def resolve_object_id(
    object_id: int,
    remap: dict,
) -> int:
    """
    Resolve chains such as:

        12 -> 7
         7 -> 3

    into:

        12 -> 3
    """

    visited = set()

    while object_id in remap:

        if object_id in visited:
            break

        visited.add(object_id)

        object_id = remap[
            object_id
        ]

    return object_id


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a persistent semantic object map "
            "from 3D object observations."
        )
    )

    parser.add_argument(
        "--objects3d",
        required=True,
        help=(
            "CSV produced by "
            "localize_objects_3d.py"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory where the persistent "
            "object map will be saved."
        ),
    )

    parser.add_argument(
        "--association-distance",
        type=float,
        default=0.25,
        help=(
            "Maximum same-class 3D association "
            "distance in metres."
        ),
    )

    parser.add_argument(
        "--min-observations",
        type=int,
        default=3,
        help=(
            "Minimum number of observations required "
            "for a persistent object."
        ),
    )

    parser.add_argument(
        "--final-merge-distance",
        type=float,
        default=0.05,
        help=(
            "Maximum distance for conservative final "
            "duplicate-track consolidation."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load 3D observations grouped by frame
    # -----------------------------------------------------------------

    frames = defaultdict(list)

    with open(
        args.objects3d,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            observation = ObjectObservation(
                frame_index=int(
                    row["frame_index"]
                ),

                timestamp=float(
                    row["timestamp"]
                ),

                class_name=row[
                    "class_name"
                ],

                confidence=float(
                    row["confidence"]
                ),

                position=np.array(
                    [
                        float(
                            row["x_world"]
                        ),
                        float(
                            row["y_world"]
                        ),
                        float(
                            row["z_world"]
                        ),
                    ],
                    dtype=np.float64,
                ),

                bbox=np.array(
                    [
                        float(
                            row["x1"]
                        ),
                        float(
                            row["y1"]
                        ),
                        float(
                            row["x2"]
                        ),
                        float(
                            row["y2"]
                        ),
                    ],
                    dtype=np.float64,
                ),
            )

            frames[
                observation.frame_index
            ].append(
                observation
            )

    # -----------------------------------------------------------------
    # Build persistent map
    # -----------------------------------------------------------------

    object_map = PersistentObjectMap(
        association_distance=(
            args.association_distance
        )
    )

    association_rows = []

    total_observations = 0

    for frame_index in sorted(
        frames.keys()
    ):

        observations = frames[
            frame_index
        ]

        total_observations += len(
            observations
        )

        assignments = (
            object_map.process_frame(
                observations
            )
        )

        for (
            observation,
            object_id,
        ) in assignments:

            association_rows.append(
                {
                    "frame_index":
                        observation.frame_index,

                    "timestamp":
                        observation.timestamp,

                    # Original detector class is retained.
                    # This lets us later measure semantic
                    # purity after tracks are merged.
                    "class_name":
                        observation.class_name,

                    "confidence":
                        observation.confidence,

                    "object_id":
                        object_id,

                    "x_world":
                        observation.position[0],

                    "y_world":
                        observation.position[1],

                    "z_world":
                        observation.position[2],
                }
            )

    # -----------------------------------------------------------------
    # First duplicate consolidation
    #
    # This is the generic merge stage contained inside
    # PersistentObjectMap.
    # -----------------------------------------------------------------

    candidate_objects_before_merge = len(
        object_map.objects
    )

    first_id_remap = (
        object_map.merge_duplicates()
    )

    candidate_objects_after_merge = len(
        object_map.objects
    )

    # Apply first-stage ID remapping to observation history.
    for row in association_rows:

        row["object_id"] = (
            resolve_object_id(
                int(
                    row["object_id"]
                ),
                first_id_remap,
            )
        )

    # -----------------------------------------------------------------
    # Persistence filtering
    # -----------------------------------------------------------------

    persistent_objects = [
        obj
        for obj in object_map.objects
        if (
            obj.observation_count
            >= args.min_observations
        )
    ]

    persistent_before_final_merge = len(
        persistent_objects
    )

    # Number rejected because they did not survive
    # the minimum-observation requirement.
    rejected_objects = (
        candidate_objects_after_merge
        - persistent_before_final_merge
    )

    # -----------------------------------------------------------------
    # Final strict duplicate consolidation
    #
    # Conditions are implemented inside
    # merge_persistent_duplicates():
    #
    #   compatible semantic class
    #   AND <= final_merge_distance
    #   AND zero shared frames
    #
    # Default = 5 cm.
    # -----------------------------------------------------------------

    (
        persistent_objects,
        final_id_remap,
    ) = merge_persistent_duplicates(
        persistent_objects,
        merge_distance=(
            args.final_merge_distance
        ),
    )

    persistent_after_final_merge = len(
        persistent_objects
    )

    # Apply final merge IDs to all historical observations.
    for row in association_rows:

        row["object_id"] = (
            resolve_object_id(
                int(
                    row["object_id"]
                ),
                final_id_remap,
            )
        )

    # -----------------------------------------------------------------
    # Final persistent IDs
    # -----------------------------------------------------------------

    persistent_ids = {
        int(
            obj.object_id
        )
        for obj in persistent_objects
    }

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Save final persistent object map
    # -----------------------------------------------------------------

    map_path = (
        output_dir
        / "object_map.csv"
    )

    map_fieldnames = [
        "object_id",
        "class_name",
        "observation_count",
        "mean_confidence",
        "x_world",
        "y_world",
        "z_world",
        "position_rmse_m",
    ]

    with map_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=map_fieldnames,
        )

        writer.writeheader()

        for obj in sorted(
            persistent_objects,
            key=lambda x:
                x.object_id,
        ):

            position = (
                obj.position
            )

            writer.writerow(
                {
                    "object_id":
                        obj.object_id,

                    "class_name":
                        obj.class_name,

                    "observation_count":
                        obj.observation_count,

                    "mean_confidence":
                        obj.mean_confidence,

                    "x_world":
                        position[0],

                    "y_world":
                        position[1],

                    "z_world":
                        position[2],

                    "position_rmse_m":
                        obj.position_rmse,
                }
            )

    # -----------------------------------------------------------------
    # Save complete observation history
    # -----------------------------------------------------------------

    observations_path = (
        output_dir
        / "object_observations.csv"
    )

    observation_fieldnames = [
        "frame_index",
        "timestamp",
        "class_name",
        "confidence",
        "object_id",
        "x_world",
        "y_world",
        "z_world",
        "persistent",
    ]

    with observations_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                observation_fieldnames
            ),
        )

        writer.writeheader()

        for row in association_rows:

            output_row = dict(
                row
            )

            output_row[
                "persistent"
            ] = (
                int(
                    output_row[
                        "object_id"
                    ]
                )
                in persistent_ids
            )

            writer.writerow(
                output_row
            )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    print()
    print(
        "Persistent object mapping v3"
    )

    print(
        "----------------------------"
    )

    print(
        f"3D observations:               "
        f"{total_observations}"
    )

    print(
        f"Candidates before merge:       "
        f"{candidate_objects_before_merge}"
    )

    print(
        f"Candidates after merge:        "
        f"{candidate_objects_after_merge}"
    )

    print(
        f"Initial duplicate tracks merged: "
        f"{len(first_id_remap)}"
    )

    print(
        f"Rejected non-persistent tracks: "
        f"{rejected_objects}"
    )

    print(
        f"Persistent before final merge: "
        f"{persistent_before_final_merge}"
    )

    print(
        f"Strict duplicates merged:      "
        f"{len(final_id_remap)}"
    )

    print(
        f"Persistent objects:            "
        f"{persistent_after_final_merge}"
    )

    # -----------------------------------------------------------------
    # Persistent-map statistics
    # -----------------------------------------------------------------

    if persistent_objects:

        counts = np.asarray(
            [
                obj.observation_count
                for obj
                in persistent_objects
            ],
            dtype=np.float64,
        )

        errors = np.asarray(
            [
                obj.position_rmse
                for obj
                in persistent_objects
            ],
            dtype=np.float64,
        )

        print(
            f"Mean observations/object:      "
            f"{counts.mean():.2f}"
        )

        print(
            f"Median observations/object:    "
            f"{np.median(counts):.1f}"
        )

        print(
            f"Mean position RMSE:             "
            f"{errors.mean():.3f} m"
        )

        print(
            f"Median position RMSE:           "
            f"{np.median(errors):.3f} m"
        )

        print(
            f"90th percentile position RMSE: "
            f"{np.percentile(errors, 90):.3f} m"
        )

    # -----------------------------------------------------------------
    # Final paths
    # -----------------------------------------------------------------

    print()
    print(
        f"Object map:       "
        f"{map_path}"
    )

    print(
        f"Observations:     "
        f"{observations_path}"
    )


if __name__ == "__main__":
    main()