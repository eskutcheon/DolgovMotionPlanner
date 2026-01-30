
import math

from models import BicycleModel
from structs import Pose, VehicleParams


def test_bicycle_propagate_straight_line():
    model = BicycleModel(VehicleParams(wheelbase=2.5))
    p0 = Pose(0.0, 0.0, 0.0)
    p1 = model.propagate(p0, steer=0.0, direction=1, ds=3.0)
    assert abs(p1.x - 3.0) < 1e-12
    assert abs(p1.y - 0.0) < 1e-12
    assert abs(p1.theta - 0.0) < 1e-12


def test_bicycle_propagate_reverse_straight_line():
    model = BicycleModel(VehicleParams(wheelbase=2.5))
    p0 = Pose(0.0, 0.0, math.pi / 2.0)
    p1 = model.propagate(p0, steer=0.0, direction=-1, ds=2.0)
    # facing +y; reversing moves -y
    assert abs(p1.x - 0.0) < 1e-12
    assert abs(p1.y + 2.0) < 1e-12


def test_bicycle_propagate_arc_has_expected_heading_change():
    L = 2.5
    steer = math.radians(20.0)
    model = BicycleModel(VehicleParams(wheelbase=L))
    p0 = Pose(0.0, 0.0, 0.0)
    ds = 1.0
    p1 = model.propagate(p0, steer=steer, direction=1, ds=ds)

    kappa = math.tan(steer) / L
    expected_dtheta = ds * kappa
    assert abs((p1.theta - expected_dtheta)) < 1e-6

