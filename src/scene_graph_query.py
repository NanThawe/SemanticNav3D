import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


class SceneGraphQuery:
    """
    Query interface for the SemanticNav3D object-level scene graph.

    The graph currently contains metric proximity relations:

        VERY_NEAR
        NEAR

    This class does not assume that graph edges represent
    collision-free robot navigation paths.
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

        self.metadata = graph.get(
            "metadata",
            {},
        )

        self.nodes = graph.get(
            "nodes",
            [],
        )

        self.edges = graph.get(
            "edges",
            [],
        )

        # -------------------------------------------------------------
        # Node lookup
        # -------------------------------------------------------------

        self.nodes_by_id = {
            node["node_id"]: node
            for node in self.nodes
        }

        # -------------------------------------------------------------
        # Undirected adjacency
        # -------------------------------------------------------------

        self.adjacency = defaultdict(
            list
        )

        for edge in self.edges:

            self.adjacency[
                edge["source"]
            ].append(
                {
                    "node_id":
                        edge["target"],

                    "relation":
                        edge["relation"],

                    "distance_m":
                        float(
                            edge[
                                "distance_m"
                            ]
                        ),
                }
            )

            self.adjacency[
                edge["target"]
            ].append(
                {
                    "node_id":
                        edge["source"],

                    "relation":
                        edge["relation"],

                    "distance_m":
                        float(
                            edge[
                                "distance_m"
                            ]
                        ),
                }
            )


    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _require_node(
        self,
        node_id: str,
    ) -> dict:

        if node_id not in self.nodes_by_id:
            raise ValueError(
                f"Unknown node: {node_id}"
            )

        return self.nodes_by_id[
            node_id
        ]


    @staticmethod
    def _position(
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
    # Find objects by class
    # -----------------------------------------------------------------

    def find_by_class(
        self,
        class_name: str,
    ) -> list[dict]:
        """
        Return all objects having the requested semantic class.
        """

        class_name = (
            class_name
            .strip()
            .lower()
        )

        return [
            node
            for node in self.nodes
            if (
                str(
                    node["class_name"]
                ).lower()
                == class_name
            )
        ]


    # -----------------------------------------------------------------
    # Graph neighbours
    # -----------------------------------------------------------------

    def neighbours(
        self,
        node_id: str,
        relation: str | None = None,
    ) -> list[dict]:
        """
        Return graph neighbours ordered by metric distance.
        """

        self._require_node(
            node_id
        )

        results = []

        for edge in self.adjacency[
            node_id
        ]:

            if (
                relation is not None
                and edge["relation"]
                != relation
            ):
                continue

            neighbour = dict(
                self.nodes_by_id[
                    edge["node_id"]
                ]
            )

            neighbour[
                "relation"
            ] = edge["relation"]

            neighbour[
                "distance_m"
            ] = edge["distance_m"]

            results.append(
                neighbour
            )

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
        node_id: str,
        class_name: str | None = None,
    ) -> dict | None:
        """
        Find the globally nearest mapped object.

        Unlike neighbours(), this searches all graph nodes,
        not only objects inside the 0.40 m NEAR threshold.
        """

        source = self._require_node(
            node_id
        )

        source_position = (
            self._position(
                source
            )
        )

        if class_name is not None:
            class_name = (
                class_name
                .strip()
                .lower()
            )

        candidates = []

        for node in self.nodes:

            if (
                node["node_id"]
                == node_id
            ):
                continue

            if (
                class_name is not None
                and str(
                    node[
                        "class_name"
                    ]
                ).lower()
                != class_name
            ):
                continue

            distance = float(
                np.linalg.norm(
                    source_position
                    - self._position(
                        node
                    )
                )
            )

            result = dict(
                node
            )

            result[
                "distance_m"
            ] = distance

            candidates.append(
                result
            )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item:
                item["distance_m"],
        )


    # -----------------------------------------------------------------
    # Local neighbourhood
    # -----------------------------------------------------------------

    def local_neighbourhood(
        self,
        node_id: str,
    ) -> dict:
        """
        Return one object's local scene-graph neighbourhood.
        """

        node = self._require_node(
            node_id
        )

        neighbours = self.neighbours(
            node_id
        )

        return {
            "centre": node,
            "degree": len(
                neighbours
            ),
            "neighbours": neighbours,
        }


    # -----------------------------------------------------------------
    # Connected spatial cluster
    # -----------------------------------------------------------------

    def connected_component(
        self,
        node_id: str,
    ) -> list[dict]:
        """
        Return all objects connected to the requested object through
        NEAR / VERY_NEAR proximity edges.

        This is a spatial cluster, not a navigation route.
        """

        self._require_node(
            node_id
        )

        visited = set()

        queue = deque(
            [node_id]
        )

        while queue:

            current = queue.popleft()

            if current in visited:
                continue

            visited.add(
                current
            )

            for neighbour in self.adjacency[
                current
            ]:

                neighbour_id = (
                    neighbour[
                        "node_id"
                    ]
                )

                if (
                    neighbour_id
                    not in visited
                ):
                    queue.append(
                        neighbour_id
                    )

        return [
            self.nodes_by_id[
                current_id
            ]
            for current_id
            in sorted(
                visited
            )
        ]


    # -----------------------------------------------------------------
    # Graph summary
    # -----------------------------------------------------------------

    def summary(self) -> dict:

        degrees = {
            node["node_id"]:
                len(
                    self.adjacency[
                        node[
                            "node_id"
                        ]
                    ]
                )
            for node in self.nodes
        }

        isolated = [
            node_id
            for node_id, degree
            in degrees.items()
            if degree == 0
        ]

        return {
            "nodes":
                len(
                    self.nodes
                ),

            "edges":
                len(
                    self.edges
                ),

            "isolated_nodes":
                isolated,

            "degrees":
                degrees,
        }