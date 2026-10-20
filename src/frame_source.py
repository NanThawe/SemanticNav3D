from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class RGBDFrame:
    index: int
    timestamp: float

    rgb: np.ndarray
    depth: np.ndarray

    intrinsics: CameraIntrinsics