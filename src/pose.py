from dataclasses import dataclass

import numpy as np


@dataclass
class CameraPose:
    timestamp: float

    translation: np.ndarray
    quaternion: np.ndarray

    # 4x4 transformation:
    # camera coordinates -> world coordinates
    T_world_camera: np.ndarray