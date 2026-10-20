import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.scene_graph import (
    SceneGraphBuilder,
    SceneGraphNode,
)


# ---------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------

def find_connected_components(
    nodes,
    edges,
):
    """
    Find connected components without requiring NetworkX.
    """

    adjacency = defaultdict(set)

    for node in nodes:
        adjacency[node.node_id]

    for edge in edges:

        adjacency[
            edge.source
        ].add(
            edge.target
        )

        adjacency[
            edge.target
        ].add(
            edge.source
        )

    visited = set()
    components = []

    for node in adjacency:

        if node in visited:
            continue

        stack = [node]
        component = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            component.append(
                current
            )

            for neighbour in adjacency[
                current
            ]:

                if neighbour not in visited:
                    stack.append(
                        neighbour
                    )

        components.append(
            component
        )

    return components


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build an object-level spatial scene graph "
            "from the persistent SemanticNav3D object map."
        )
    )

    parser.add_argument(
        "--objects",
        required=True,
        help="Final object_map.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--very-near-distance",
        type=float,
        default=0.30,
        help="VERY_NEAR threshold in metres.",
    )

    parser.add_argument(
        "--near-distance",
        type=float,
        default=0.75,
        help="NEAR threshold in metres.",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Optional minimum mean object confidence. "
            "Default keeps all persistent objects."
        ),
    )

    parser.add_argument(
        "--max-position-rmse",
        type=float,
        default=None,
        help=(
            "Optional maximum object-position RMSE. "
            "Default keeps all persistent objects."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load persistent map
    # -----------------------------------------------------------------

    df = pd.read_csv(
        args.objects
    )

    objects_before_filtering = len(
        df
    )

    # Optional quality filters
    df = df[
        df["mean_confidence"]
        >= args.min_confidence
    ].copy()

    if (
        args.max_position_rmse
        is not None
    ):
        df = df[
            df["position_rmse_m"]
            <= args.max_position_rmse
        ].copy()

    # -----------------------------------------------------------------
    # Convert persistent objects to graph nodes
    # -----------------------------------------------------------------

    nodes = []

    for _, row in df.iterrows():

        node = SceneGraphNode(
            object_id=int(
                row["object_id"]
            ),

            class_name=str(
                row["class_name"]
            ),

            x_world=float(
                row["x_world"]
            ),

            y_world=float(
                row["y_world"]
            ),

            z_world=float(
                row["z_world"]
            ),

            observation_count=int(
                row["observation_count"]
            ),

            mean_confidence=float(
                row["mean_confidence"]
            ),

            position_rmse_m=float(
                row["position_rmse_m"]
            ),
        )

        nodes.append(
            node
        )

    # -----------------------------------------------------------------
    # Build graph
    # -----------------------------------------------------------------

    builder = SceneGraphBuilder(
        very_near_distance=(
            args.very_near_distance
        ),

        near_distance=(
            args.near_distance
        ),
    )

    edges = builder.build(
        nodes
    )

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
    # Save JSON graph
    # -----------------------------------------------------------------

    graph_path = (
        output_dir
        / "scene_graph.json"
    )

    graph_data = {
        "metadata": {
            "graph_type":
                "object_level_spatial_scene_graph",

            "coordinate_system":
                "ORB-SLAM3 world frame",

            "node_count":
                len(nodes),

            "edge_count":
                len(edges),

            "very_near_threshold_m":
                args.very_near_distance,

            "near_threshold_m":
                args.near_distance,

            "relations": [
                "VERY_NEAR",
                "NEAR",
            ],
        },

        "nodes": [
            node.to_dict()
            for node in nodes
        ],

        "edges": [
            edge.to_dict()
            for edge in edges
        ],
    }

    with graph_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            graph_data,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Save node table
    # -----------------------------------------------------------------

    nodes_path = (
        output_dir
        / "scene_graph_nodes.csv"
    )

    node_fields = [
        "node_id",
        "object_id",
        "class_name",
        "x_world",
        "y_world",
        "z_world",
        "observation_count",
        "mean_confidence",
        "position_rmse_m",
    ]

    with nodes_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=node_fields,
        )

        writer.writeheader()

        for node in nodes:
            writer.writerow(
                node.to_dict()
            )

    # -----------------------------------------------------------------
    # Save edge table
    # -----------------------------------------------------------------

    edges_path = (
        output_dir
        / "scene_graph_edges.csv"
    )

    edge_fields = [
        "source",
        "target",
        "source_object_id",
        "target_object_id",
        "relation",
        "distance_m",
    ]

    with edges_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=edge_fields,
        )

        writer.writeheader()

        for edge in edges:
            writer.writerow(
                edge.to_dict()
            )

    # -----------------------------------------------------------------
    # Numerical graph statistics
    # -----------------------------------------------------------------

    very_near_edges = [
        edge
        for edge in edges
        if edge.relation
        == "VERY_NEAR"
    ]

    near_edges = [
        edge
        for edge in edges
        if edge.relation
        == "NEAR"
    ]

    degrees = defaultdict(int)

    for node in nodes:
        degrees[
            node.node_id
        ] = 0

    for edge in edges:

        degrees[
            edge.source
        ] += 1

        degrees[
            edge.target
        ] += 1

    degree_values = np.array(
        list(
            degrees.values()
        ),
        dtype=np.float64,
    )

    isolated_nodes = [
        node_id
        for node_id, degree
        in degrees.items()
        if degree == 0
    ]

    components = (
        find_connected_components(
            nodes,
            edges,
        )
    )

    total_possible_pairs = (
        len(nodes)
        * (len(nodes) - 1)
        // 2
    )

    graph_density = (
        len(edges)
        / total_possible_pairs
        if total_possible_pairs > 0
        else 0.0
    )

    if edges:

        edge_distances = np.array(
            [
                edge.distance_m
                for edge in edges
            ],
            dtype=np.float64,
        )

    else:

        edge_distances = np.array(
            [],
            dtype=np.float64,
        )

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------

    print()
    print(
        "SemanticNav3D Scene Graph"
    )

    print(
        "========================="
    )

    print()
    print(
        "Node statistics"
    )

    print(
        "---------------"
    )

    print(
        f"Objects before filtering: "
        f"{objects_before_filtering}"
    )

    print(
        f"Graph nodes:              "
        f"{len(nodes)}"
    )

    print()
    print(
        "Edge statistics"
    )

    print(
        "---------------"
    )

    print(
        f"Possible object pairs:    "
        f"{total_possible_pairs}"
    )

    print(
        f"VERY_NEAR edges:          "
        f"{len(very_near_edges)}"
    )

    print(
        f"NEAR edges:               "
        f"{len(near_edges)}"
    )

    print(
        f"Total graph edges:        "
        f"{len(edges)}"
    )

    print(
        f"Graph density:            "
        f"{graph_density:.3f}"
    )

    if len(
        edge_distances
    ) > 0:

        print(
            f"Mean edge distance:       "
            f"{edge_distances.mean():.3f} m"
        )

        print(
            f"Median edge distance:     "
            f"{np.median(edge_distances):.3f} m"
        )

    print()
    print(
        "Connectivity"
    )

    print(
        "------------"
    )

    print(
        f"Connected components:     "
        f"{len(components)}"
    )

    print(
        f"Isolated nodes:           "
        f"{len(isolated_nodes)}"
    )

    if len(
        degree_values
    ) > 0:

        print(
            f"Mean node degree:         "
            f"{degree_values.mean():.2f}"
        )

        print(
            f"Median node degree:       "
            f"{np.median(degree_values):.1f}"
        )

        print(
            f"Maximum node degree:      "
            f"{int(degree_values.max())}"
        )

    # -----------------------------------------------------------------
    # Most connected objects
    # -----------------------------------------------------------------

    ranked_nodes = sorted(
        degrees.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    print()
    print(
        "Most connected objects"
    )

    print(
        "----------------------"
    )

    for node_id, degree in (
        ranked_nodes[:10]
    ):

        print(
            f"{node_id:<20} "
            f"degree={degree}"
        )

    # -----------------------------------------------------------------
    # Save numerical summary
    # -----------------------------------------------------------------

    summary_path = (
        output_dir
        / "scene_graph_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SemanticNav3D Scene Graph Summary\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Graph nodes: {len(nodes)}\n"
        )

        f.write(
            f"Possible pairs: {total_possible_pairs}\n"
        )

        f.write(
            f"VERY_NEAR edges: {len(very_near_edges)}\n"
        )

        f.write(
            f"NEAR edges: {len(near_edges)}\n"
        )

        f.write(
            f"Total edges: {len(edges)}\n"
        )

        f.write(
            f"Graph density: {graph_density:.6f}\n"
        )

        f.write(
            f"Connected components: {len(components)}\n"
        )

        f.write(
            f"Isolated nodes: {len(isolated_nodes)}\n"
        )

        if len(
            degree_values
        ) > 0:

            f.write(
                f"Mean degree: {degree_values.mean():.6f}\n"
            )

            f.write(
                f"Median degree: {np.median(degree_values):.6f}\n"
            )

            f.write(
                f"Maximum degree: {int(degree_values.max())}\n"
            )

    print()
    print(
        "Outputs"
    )

    print(
        "-------"
    )

    print(
        f"Graph JSON:    {graph_path}"
    )

    print(
        f"Nodes CSV:     {nodes_path}"
    )

    print(
        f"Edges CSV:     {edges_path}"
    )

    print(
        f"Summary:       {summary_path}"
    )


if __name__ == "__main__":
    main()