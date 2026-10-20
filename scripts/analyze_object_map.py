import argparse
from itertools import combinations

import numpy as np
import pandas as pd


COMPATIBLE_CLASSES = {
    frozenset(("tv", "laptop")),
}


def are_classes_compatible(a, b):
    if a == b:
        return True

    return frozenset((a, b)) in COMPATIBLE_CLASSES


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--objects",
        required=True,
        help="object_map.csv",
    )

    parser.add_argument(
        "--observations",
        required=True,
        help="object_observations.csv",
    )

    args = parser.parse_args()

    objects = pd.read_csv(args.objects)
    observations = pd.read_csv(args.observations)

    # ------------------------------------------------------------
    # Keep only observations belonging to objects that actually
    # exist in the final persistent object map.
    #
    # This is more reliable than depending on the saved
    # "persistent" boolean/string representation.
    # ------------------------------------------------------------

    final_object_ids = set(
        objects["object_id"]
        .astype(int)
        .tolist()
    )

    observations["object_id"] = (
        observations["object_id"]
        .astype(int)
    )

    observations = observations[
        observations["object_id"].isin(
            final_object_ids
        )
    ].copy()

    print(
    f"Persistent observation rows loaded: "
    f"{len(observations)}"
    )

    print("\nOBJECT MAP NUMERICAL ANALYSIS")
    print("=" * 40)

    # ------------------------------------------------------------
    # Basic track statistics
    # ------------------------------------------------------------

    counts = objects["observation_count"].to_numpy()
    rmse = objects["position_rmse_m"].to_numpy()

    print("\n1. Track statistics")
    print("-------------------")

    print(f"Persistent objects:       {len(objects)}")
    print(f"Total observations:       {counts.sum()}")

    print(
        f"Mean observations/object: "
        f"{counts.mean():.2f}"
    )

    print(
        f"Median observations:      "
        f"{np.median(counts):.1f}"
    )

    print(
        f"Minimum observations:     "
        f"{counts.min()}"
    )

    print(
        f"Maximum observations:     "
        f"{counts.max()}"
    )

    # ------------------------------------------------------------
    # Position consistency
    # ------------------------------------------------------------

    print("\n2. Spatial consistency")
    print("----------------------")

    print(
        f"Mean position RMSE:       "
        f"{rmse.mean():.3f} m"
    )

    print(
        f"Median position RMSE:     "
        f"{np.median(rmse):.3f} m"
    )

    print(
        f"75th percentile RMSE:     "
        f"{np.percentile(rmse, 75):.3f} m"
    )

    print(
        f"90th percentile RMSE:     "
        f"{np.percentile(rmse, 90):.3f} m"
    )

    print(
        f"Maximum RMSE:             "
        f"{rmse.max():.3f} m"
    )

    print(
        f"Objects RMSE <= 5 cm:     "
        f"{np.sum(rmse <= 0.05)}"
    )

    print(
        f"Objects RMSE <= 10 cm:    "
        f"{np.sum(rmse <= 0.10)}"
    )

    print(
        f"Objects RMSE > 10 cm:     "
        f"{np.sum(rmse > 0.10)}"
    )

    # ------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------

    print("\n3. Persistent objects by class")
    print("------------------------------")

    class_counts = (
        objects["class_name"]
        .value_counts()
        .sort_index()
    )

    for class_name, count in class_counts.items():
        print(
            f"{class_name:<15} {count:>3}"
        )

    # ------------------------------------------------------------
    # Semantic purity
    #
    # An object receiving:
    # laptop, laptop, tv, laptop
    #
    # should have reasonably high dominant-class confidence.
    # ------------------------------------------------------------

    print("\n4. Semantic track purity")
    print("------------------------")

    purity_rows = []

    for object_id, group in observations.groupby(
        "object_id"
    ):

        scores = (
            group.groupby("class_name")["confidence"]
            .sum()
        )

        total_score = scores.sum()

        if total_score <= 0:
            continue

        dominant_class = scores.idxmax()

        purity = (
            scores.max()
            / total_score
        )

        purity_rows.append(
            {
                "object_id": object_id,
                "dominant_class": dominant_class,
                "purity": purity,
                "classes_seen": len(scores),
            }
        )

    purity_df = pd.DataFrame(purity_rows)

    if not purity_df.empty:

        print(
            f"Mean semantic purity:     "
            f"{purity_df['purity'].mean():.3f}"
        )

        print(
            f"Median semantic purity:   "
            f"{purity_df['purity'].median():.3f}"
        )

        print(
            f"Tracks purity < 0.70:     "
            f"{(purity_df['purity'] < 0.70).sum()}"
        )

        print(
            f"Tracks with >1 label:     "
            f"{(purity_df['classes_seen'] > 1).sum()}"
        )

    # ------------------------------------------------------------
    # Weak tracks
    # ------------------------------------------------------------

    print("\n5. Weak / unstable tracks")
    print("-------------------------")

    weak = objects[
        objects["observation_count"] <= 4
    ]

    unstable = objects[
        objects["position_rmse_m"] > 0.10
    ]

    print(
        f"Tracks with <=4 observations: "
        f"{len(weak)}"
    )

    print(
        f"Tracks with RMSE >10 cm:       "
        f"{len(unstable)}"
    )

    # ------------------------------------------------------------
    # Pairwise duplicate-track diagnostics
    # ------------------------------------------------------------

    print("\n6. Possible remaining duplicates")
    print("--------------------------------")

    object_frames = {}

    for object_id, group in observations.groupby(
        "object_id"
    ):
        object_frames[object_id] = set(
            group["frame_index"].astype(int)
        )

    possible_duplicates = []

    for (_, a), (_, b) in combinations(
        objects.iterrows(),
        2,
    ):

        if not are_classes_compatible(
            a["class_name"],
            b["class_name"],
        ):
            continue

        pa = np.array(
            [
                a["x_world"],
                a["y_world"],
                a["z_world"],
            ]
        )

        pb = np.array(
            [
                b["x_world"],
                b["y_world"],
                b["z_world"],
            ]
        )

        distance = float(
            np.linalg.norm(pa - pb)
        )

        if distance > 0.15:
            continue

        frames_a = object_frames.get(
            a["object_id"],
            set(),
        )

        frames_b = object_frames.get(
            b["object_id"],
            set(),
        )

        simultaneous = bool(
            frames_a.intersection(frames_b)
        )

        possible_duplicates.append(
            {
                "id_a": int(a["object_id"]),
                "class_a": a["class_name"],

                "id_b": int(b["object_id"]),
                "class_b": b["class_name"],

                "distance_m": distance,
                "simultaneous": simultaneous,
            }
        )

    possible_duplicates.sort(
        key=lambda x: x["distance_m"]
    )

    if not possible_duplicates:
        print(
            "No compatible object pairs "
            "within 15 cm."
        )

    else:

        print(
            "Pairs within 15 cm:"
        )

        for pair in possible_duplicates:

            flag = (
                "SEPARATE"
                if pair["simultaneous"]
                else "POSSIBLE DUPLICATE"
            )

            print(
                f"{pair['id_a']:>3} "
                f"{pair['class_a']:<10} <-> "
                f"{pair['id_b']:>3} "
                f"{pair['class_b']:<10} "
                f"{pair['distance_m']:.3f} m  "
                f"{flag}"
            )

    # ------------------------------------------------------------
    # Distance threshold summary
    # ------------------------------------------------------------

    if possible_duplicates:

        distances = np.array(
            [
                p["distance_m"]
                for p in possible_duplicates
                if not p["simultaneous"]
            ]
        )

        print("\n7. Duplicate-distance summary")
        print("-----------------------------")

        if len(distances):

            for threshold in [
                0.05,
                0.08,
                0.10,
                0.15,
            ]:

                print(
                    f"Non-simultaneous pairs "
                    f"within {threshold:.2f} m: "
                    f"{np.sum(distances <= threshold)}"
                )

        else:
            print(
                "No non-simultaneous compatible "
                "pairs detected."
            )


if __name__ == "__main__":
    main()