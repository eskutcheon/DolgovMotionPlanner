

import math
import numpy as np

TAU = 2.0 * math.pi

def wrap_angle(theta: float) -> float:
    """ wrap angle back to interval [-pi, pi) """
    x = (theta + math.pi) % TAU - math.pi
    return x

def wrap_angle_2pi(theta: float) -> float:
    """ wrap angle back to interval [0, 2pi) """
    return theta % TAU

def rot2d(theta: float) -> np.ndarray:
    """ 2D rotation matrix for angle theta (in radians) """
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def clamp(x: float, lo: float, hi: float) -> float:
    """ general utility for other float clamping """
    return lo if x < lo else hi if x > hi else x