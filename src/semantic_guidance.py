from dataclasses import dataclass

from src.robot_relative_query import (
    CameraTrajectoryPose,
    RobotRelativeQuery,
)


@dataclass
class GuidanceResult:
    target_id: str
    class_name: str

    distance_m: float
    bearing_deg: float

    x_right_m: float
    z_forward_m: float

    action: str

    heading_error_deg: float
    stop_distance_m: float

    side: str
    longitudinal: str


class SemanticGuidance:
    """
    Lightweight target-oriented semantic guidance.

    This module does NOT perform:
        - collision avoidance
        - free-space planning
        - trajectory generation
        - robot control

    It only converts semantic target location into
    simple directional guidance.
    """

    def __init__(
        self,
        query: RobotRelativeQuery,
        heading_tolerance_deg: float = 10.0,
        stop_distance_m: float = 0.50,
    ):
        if heading_tolerance_deg <= 0:
            raise ValueError(
                "heading_tolerance_deg must be positive."
            )

        if stop_distance_m <= 0:
            raise ValueError(
                "stop_distance_m must be positive."
            )

        self.query = query

        self.heading_tolerance_deg = float(
            heading_tolerance_deg
        )

        self.stop_distance_m = float(
            stop_distance_m
        )


    def classify_action(
        self,
        relative: dict,
    ) -> str:
        """
        Convert a camera-relative target position into
        one simple guidance action.
        """

        distance = float(
            relative["distance_m"]
        )

        bearing = float(
            relative["bearing_deg"]
        )

        z_forward = float(
            relative["z_forward_m"]
        )

        # ---------------------------------------------------------
        # Target reached
        # ---------------------------------------------------------

        if distance <= self.stop_distance_m:
            return "ARRIVED"

        # ---------------------------------------------------------
        # Target is behind camera
        # ---------------------------------------------------------

        if z_forward <= 0:
            return "TURN_AROUND"

        # ---------------------------------------------------------
        # Target is in front but not centred
        # ---------------------------------------------------------

        if (
            bearing
            < -self.heading_tolerance_deg
        ):
            return "TURN_LEFT"

        if (
            bearing
            > self.heading_tolerance_deg
        ):
            return "TURN_RIGHT"

        # ---------------------------------------------------------
        # Target approximately centred
        # ---------------------------------------------------------

        return "FORWARD"


    def guide_to_object(
        self,
        node_id: str,
        pose: CameraTrajectoryPose,
    ) -> GuidanceResult:

        relative = (
    self.query.relative_position(
        node_id,
        pose,
        centre_deadband_deg=(
            self.heading_tolerance_deg
        ),
    )
)

        action = self.classify_action(
            relative
        )

        return GuidanceResult(
            target_id=relative[
                "node_id"
            ],

            class_name=relative[
                "class_name"
            ],

            distance_m=float(
                relative[
                    "distance_m"
                ]
            ),

            bearing_deg=float(
                relative[
                    "bearing_deg"
                ]
            ),

            x_right_m=float(
                relative[
                    "x_right_m"
                ]
            ),

            z_forward_m=float(
                relative[
                    "z_forward_m"
                ]
            ),

            action=action,

            heading_error_deg=float(
                relative[
                    "bearing_deg"
                ]
            ),

            stop_distance_m=(
                self.stop_distance_m
            ),

            side=relative[
                "side"
            ],

            longitudinal=relative[
                "longitudinal"
            ],
        )


    def guide_to_class(
        self,
        class_name: str,
        pose: CameraTrajectoryPose,
    ) -> GuidanceResult | None:
        """
        Find the nearest object of a requested semantic class
        and generate guidance to it.
        """

        target = self.query.nearest(
            pose,
            class_name=class_name,
            front_only=False,
        )

        if target is None:
            return None

        return self.guide_to_object(
            target["node_id"],
            pose,
        )