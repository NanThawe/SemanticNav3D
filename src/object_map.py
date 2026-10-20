from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------
# Semantic compatibility
# ---------------------------------------------------------------------

# Some COCO classes are frequently confused in this indoor scene.
# In particular, computer monitors may alternate between "tv" and
# "laptop". Cross-class association is only allowed for explicitly
# listed compatible classes.
CLASS_COMPATIBILITY = {
    "tv": {"tv", "laptop"},
    "laptop": {"laptop", "tv"},
    "mouse": {"mouse"},
    "keyboard": {"keyboard"},
    "book": {"book"},
    "chair": {"chair"},
    "cup": {"cup"},
    "cell phone": {"cell phone"},
}

def merge_persistent_duplicates(
    objects: list[MapObject],
    merge_distance: float = 0.05,
):
    """
    Conservative final consolidation of persistent tracks.

    Merge only when:
        1. semantic classes are compatible
        2. final 3D centres are <= merge_distance apart
        3. the tracks were never observed in the same frame

    Returns:
        merged_objects
        id_remap
    """

    objects = list(objects)
    id_remap = {}

    changed = True

    while changed:
        changed = False

        for i in range(len(objects)):

            if changed:
                break

            obj_a = objects[i]

            for j in range(
                i + 1,
                len(objects),
            ):
                obj_b = objects[j]

                # ------------------------------------------
                # Semantic compatibility
                # ------------------------------------------

                if not classes_compatible(
                    obj_a.class_name,
                    obj_b.class_name,
                ):
                    continue

                # ------------------------------------------
                # Spatial distance
                # ------------------------------------------

                distance = float(
                    np.linalg.norm(
                        obj_a.position
                        - obj_b.position
                    )
                )

                if distance > merge_distance:
                    continue

                # ------------------------------------------
                # Never independently observed together
                # ------------------------------------------

                shared_frames = (
                    obj_a.frame_indices
                    & obj_b.frame_indices
                )

                if shared_frames:
                    continue

                # ------------------------------------------
                # Select survivor
                #
                # Keep the object with more observations.
                # ------------------------------------------

                if (
                    obj_b.observation_count
                    > obj_a.observation_count
                ):
                    obj_a, obj_b = (
                        obj_b,
                        obj_a,
                    )

                    survivor_index = j
                    removed_index = i

                else:
                    survivor_index = i
                    removed_index = j

                surviving_id = (
                    obj_a.object_id
                )

                removed_id = (
                    obj_b.object_id
                )

                # ------------------------------------------
                # Transfer observations
                # ------------------------------------------

                for observation in (
                    obj_b.observations
                ):
                    obj_a.add(
                        observation
                    )

                id_remap[
                    removed_id
                ] = surviving_id

                objects.pop(
                    removed_index
                )

                changed = True
                break

    # Resolve possible ID chains
    for old_id in list(
        id_remap.keys()
    ):
        target = id_remap[
            old_id
        ]

        while target in id_remap:
            target = id_remap[
                target
            ]

        id_remap[
            old_id
        ] = target

    return objects, id_remap


def classes_compatible(
    class_a: str,
    class_b: str,
) -> bool:
    """
    Return True when two detector labels are considered compatible.
    """

    if class_a == class_b:
        return True

    compatible = CLASS_COMPATIBILITY.get(
        class_a,
        {class_a},
    )

    return class_b in compatible


# ---------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------

def bbox_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    """
    Compute intersection-over-union between two bounding boxes.

    Boxes are expected in:
        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        ix2 - ix1,
    )

    intersection_height = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        (ax2 - ax1)
        * (ay2 - ay1),
    )

    area_b = max(
        0.0,
        (bx2 - bx1)
        * (by2 - by1),
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


# ---------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------

@dataclass
class ObjectObservation:
    """
    One object detection/localisation from one RGB-D frame.
    """

    frame_index: int
    timestamp: float

    class_name: str
    confidence: float

    # World-space XYZ
    position: np.ndarray

    # Detection bounding box:
    # [x1, y1, x2, y2]
    bbox: np.ndarray


# ---------------------------------------------------------------------
# Persistent map object
# ---------------------------------------------------------------------

@dataclass
class MapObject:
    """
    Persistent physical-object hypothesis.

    The final semantic class is not taken from one detection.
    Instead, detector confidences accumulate as class evidence.
    """

    object_id: int

    observations: list[ObjectObservation] = field(
        default_factory=list
    )

    class_scores: dict = field(
        default_factory=lambda: defaultdict(float)
    )


    def add(
        self,
        observation: ObjectObservation,
    ):
        """
        Add a new observation to this persistent object.
        """

        self.observations.append(
            observation
        )

        self.class_scores[
            observation.class_name
        ] += observation.confidence


    @property
    def class_name(self) -> str:
        """
        Current semantic class based on accumulated
        confidence-weighted votes.
        """

        if not self.class_scores:
            return "unknown"

        return max(
            self.class_scores,
            key=self.class_scores.get,
        )


    @property
    def position(self) -> np.ndarray:
        """
        Robust long-term object position.

        Median is used rather than the mean so occasional
        incorrect depth measurements have less influence.
        """

        if not self.observations:
            return np.zeros(
                3,
                dtype=np.float64,
            )

        positions = np.asarray(
            [
                observation.position
                for observation
                in self.observations
            ],
            dtype=np.float64,
        )

        return np.median(
            positions,
            axis=0,
        )


    @property
    def observation_count(self) -> int:
        return len(
            self.observations
        )


    @property
    def mean_confidence(self) -> float:

        if not self.observations:
            return 0.0

        return float(
            np.mean(
                [
                    observation.confidence
                    for observation
                    in self.observations
                ]
            )
        )


    @property
    def position_rmse(self) -> float:
        """
        Spatial consistency around the final robust
        object position.
        """

        if not self.observations:
            return 0.0

        positions = np.asarray(
            [
                observation.position
                for observation
                in self.observations
            ],
            dtype=np.float64,
        )

        distances = np.linalg.norm(
            positions - self.position,
            axis=1,
        )

        return float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        )


    @property
    def last_observation(
        self,
    ) -> ObjectObservation:
        """
        Most recent observation by timestamp.
        """

        return max(
            self.observations,
            key=lambda observation:
                observation.timestamp,
        )


    @property
    def last_position(
        self,
    ) -> np.ndarray:

        return (
            self.last_observation.position
        )


    @property
    def last_bbox(
        self,
    ) -> np.ndarray:

        return (
            self.last_observation.bbox
        )


    @property
    def last_timestamp(
        self,
    ) -> float:

        return float(
            self.last_observation.timestamp
        )


    @property
    def last_frame_index(
        self,
    ) -> int:

        return int(
            self.last_observation.frame_index
        )


    @property
    def frame_indices(
        self,
    ) -> set[int]:
        """
        Frames in which this object was observed.
        """

        return {
            observation.frame_index
            for observation
            in self.observations
        }


# ---------------------------------------------------------------------
# Persistent object map
# ---------------------------------------------------------------------

class PersistentObjectMap:
    """
    Lightweight object-level data association.

    Association uses:

        1. 3D spatial consistency
        2. recent 3D position
        3. semantic compatibility
        4. temporal continuity
        5. short-term 2D bounding-box IoU

    No neural ReID or appearance embeddings are required.
    """

    def __init__(
        self,
        association_distance: float = 0.25,
        compatible_class_distance: float = 0.15,
        merge_distance: float = 0.08,
    ):

        # Normal association radius for the same class.
        self.association_distance = float(
            association_distance
        )

        # Cross-class associations such as:
        # tv <-> laptop
        # use a stricter radius.
        self.compatible_class_distance = float(
            compatible_class_distance
        )

        # Used only during post-processing merge.
        self.merge_distance = float(
            merge_distance
        )

        self.objects: list[MapObject] = []

        self.next_object_id = 0


    # -----------------------------------------------------------------
    # Object creation
    # -----------------------------------------------------------------

    def _create_object(
        self,
        observation: ObjectObservation,
    ) -> MapObject:

        obj = MapObject(
            object_id=self.next_object_id,
        )

        obj.add(
            observation
        )

        self.objects.append(
            obj
        )

        self.next_object_id += 1

        return obj


    # -----------------------------------------------------------------
    # Frame association
    # -----------------------------------------------------------------

    def process_frame(
        self,
        observations: list[ObjectObservation],
    ):
        """
        Associate all observations from one frame.

        Each observation can match at most one map object.

        Each map object can also receive at most one observation
        from the same frame.

        Returns:
            [
                (observation, object_id),
                ...
            ]
        """

        candidates = []

        # -------------------------------------------------------------
        # Generate candidate observation-object associations
        # -------------------------------------------------------------

        for (
            observation_index,
            observation,
        ) in enumerate(observations):

            for (
                object_index,
                obj,
            ) in enumerate(self.objects):

                same_class = (
                    observation.class_name
                    == obj.class_name
                )

                compatible = classes_compatible(
                    observation.class_name,
                    obj.class_name,
                )

                if not compatible:
                    continue

                # -----------------------------------------------------
                # Spatial consistency
                # -----------------------------------------------------

                distance_to_map = float(
                    np.linalg.norm(
                        observation.position
                        - obj.position
                    )
                )

                distance_to_last = float(
                    np.linalg.norm(
                        observation.position
                        - obj.last_position
                    )
                )

                # Long-term centre gives map stability.
                # Last observation helps preserve temporal continuity.
                distance = min(
                    distance_to_map,
                    distance_to_last,
                )

                # -----------------------------------------------------
                # Distance gates
                # -----------------------------------------------------

                if same_class:

                    if (
                        distance
                        > self.association_distance
                    ):
                        continue

                else:

                    if (
                        distance
                        > self.compatible_class_distance
                    ):
                        continue

                # -----------------------------------------------------
                # Temporal consistency
                # -----------------------------------------------------

                time_gap = abs(
                    observation.timestamp
                    - obj.last_timestamp
                )

                # -----------------------------------------------------
                # Bounding-box consistency
                # -----------------------------------------------------

                iou = bbox_iou(
                    observation.bbox,
                    obj.last_bbox,
                )

                # -----------------------------------------------------
                # Association score
                #
                # Lower is better.
                # -----------------------------------------------------

                score = distance

                # Cross-class association is possible but should
                # require stronger spatial/temporal evidence.
                if not same_class:
                    score += 0.03

                # Bounding boxes are meaningful mainly over short
                # temporal intervals.
                if time_gap <= 1.0:
                    score -= (
                        0.06 * iou
                    )

                # Prefer recently seen objects when several candidates
                # have similar geometry.
                score += (
                    min(
                        time_gap,
                        3.0,
                    )
                    * 0.005
                )

                candidates.append(
                    (
                        score,
                        distance,
                        observation_index,
                        object_index,
                    )
                )

        # -------------------------------------------------------------
        # Greedy one-to-one assignment
        # -------------------------------------------------------------

        candidates.sort(
            key=lambda item:
                item[0]
        )

        used_observations = set()
        used_objects = set()

        associations = {}

        for (
            score,
            distance,
            observation_index,
            object_index,
        ) in candidates:

            if (
                observation_index
                in used_observations
            ):
                continue

            if (
                object_index
                in used_objects
            ):
                continue

            used_observations.add(
                observation_index
            )

            used_objects.add(
                object_index
            )

            associations[
                observation_index
            ] = object_index

        # -------------------------------------------------------------
        # Update existing objects or create new ones
        # -------------------------------------------------------------

        results = []

        for (
            observation_index,
            observation,
        ) in enumerate(observations):

            if (
                observation_index
                in associations
            ):

                object_index = associations[
                    observation_index
                ]

                obj = self.objects[
                    object_index
                ]

                obj.add(
                    observation
                )

            else:

                obj = self._create_object(
                    observation
                )

            results.append(
                (
                    observation,
                    obj.object_id,
                )
            )

        return results


    # -----------------------------------------------------------------
    # Duplicate consolidation
    # -----------------------------------------------------------------

    def merge_duplicates(self):
        """
        Merge very close track fragments after the full sequence
        has been processed.

        Two tracks may merge only when:

            1. classes are compatible
            2. centres are closer than merge_distance
            3. the two tracks were never independently detected
               in the same frame

        The third condition protects genuinely separate nearby
        objects from being collapsed.

        Returns:
            {
                removed_object_id: surviving_object_id,
                ...
            }
        """

        remap = {}

        changed = True

        while changed:

            changed = False

            for i in range(
                len(self.objects)
            ):

                if changed:
                    break

                obj_a = self.objects[i]

                for j in range(
                    i + 1,
                    len(self.objects),
                ):

                    obj_b = self.objects[j]

                    # -------------------------------------------------
                    # Semantic compatibility
                    # -------------------------------------------------

                    if not classes_compatible(
                        obj_a.class_name,
                        obj_b.class_name,
                    ):
                        continue

                    # -------------------------------------------------
                    # Spatial proximity
                    # -------------------------------------------------

                    distance = float(
                        np.linalg.norm(
                            obj_a.position
                            - obj_b.position
                        )
                    )

                    if (
                        distance
                        > self.merge_distance
                    ):
                        continue

                    # -------------------------------------------------
                    # Simultaneous observation test
                    #
                    # If they appeared independently in the same
                    # frame, preserve them as separate objects.
                    # -------------------------------------------------

                    if not (
                        obj_a.frame_indices
                        .isdisjoint(
                            obj_b.frame_indices
                        )
                    ):
                        continue

                    # -------------------------------------------------
                    # Merge B into A
                    # -------------------------------------------------

                    surviving_id = (
                        obj_a.object_id
                    )

                    removed_id = (
                        obj_b.object_id
                    )

                    for observation in (
                        obj_b.observations
                    ):

                        obj_a.add(
                            observation
                        )

                    remap[
                        removed_id
                    ] = surviving_id

                    self.objects.pop(
                        j
                    )

                    changed = True

                    break

        # -------------------------------------------------------------
        # Resolve remapping chains.
        #
        # Example:
        #
        # 12 -> 7
        #  7 -> 3
        #
        # becomes:
        #
        # 12 -> 3
        #  7 -> 3
        # -------------------------------------------------------------

        for object_id in list(
            remap.keys()
        ):

            target = remap[
                object_id
            ]

            while target in remap:

                target = remap[
                    target
                ]

            remap[
                object_id
            ] = target

        return remap