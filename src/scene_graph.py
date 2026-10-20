from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class SceneGraphNode:
    object_id: int
    class_name: str

    x_world: float
    y_world: float
    z_world: float

    observation_count: int
    mean_confidence: float
    position_rmse_m: float

    @property
    def node_id(self) -> str:
        return f"{self.class_name}_{self.object_id}"

    @property
    def position(self) -> np.ndarray:
        return np.array(
            [
                self.x_world,
                self.y_world,
                self.z_world,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["node_id"] = self.node_id
        return data


@dataclass
class SceneGraphEdge:
    source: str
    target: str

    source_object_id: int
    target_object_id: int

    relation: str
    distance_m: float

    def to_dict(self) -> dict:
        return asdict(self)


class SceneGraphBuilder:
    """
    Build a conservative metric scene graph.

    Relations:

        VERY_NEAR:
            distance <= very_near_distance

        NEAR:
            very_near_distance < distance <= near_distance

    One undirected edge is generated per qualifying object pair.
    """

    def __init__(
        self,
        very_near_distance: float = 0.30,
        near_distance: float = 0.75,
    ):
        if very_near_distance <= 0:
            raise ValueError(
                "very_near_distance must be positive."
            )

        if near_distance <= very_near_distance:
            raise ValueError(
                "near_distance must be greater than "
                "very_near_distance."
            )

        self.very_near_distance = float(
            very_near_distance
        )

        self.near_distance = float(
            near_distance
        )


    def classify_relation(
        self,
        distance: float,
    ) -> Optional[str]:

        if distance <= self.very_near_distance:
            return "VERY_NEAR"

        if distance <= self.near_distance:
            return "NEAR"

        return None


    def build(
        self,
        nodes: list[SceneGraphNode],
    ) -> list[SceneGraphEdge]:

        edges = []

        for i in range(len(nodes)):

            node_a = nodes[i]

            for j in range(
                i + 1,
                len(nodes),
            ):

                node_b = nodes[j]

                distance = float(
                    np.linalg.norm(
                        node_a.position
                        - node_b.position
                    )
                )

                relation = (
                    self.classify_relation(
                        distance
                    )
                )

                if relation is None:
                    continue

                edge = SceneGraphEdge(
                    source=node_a.node_id,
                    target=node_b.node_id,

                    source_object_id=(
                        node_a.object_id
                    ),

                    target_object_id=(
                        node_b.object_id
                    ),

                    relation=relation,

                    distance_m=distance,
                )

                edges.append(edge)

        return edges