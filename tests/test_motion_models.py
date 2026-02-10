
import math

from src.models import BicycleModel
from src.structs import Pose, VehicleParams


def test_bicycle_propagate_straight_line():
    model = BicycleModel(VehicleParams(wheelbase=2.5))
    p0 = Pose(0.0, 0.0, 0.0)
    p1 = model.propagate(p0, u=0.0, direction=1, ds=3.0, kappa_max=0.2)
    assert abs(p1.x - 3.0) < 1e-12
    assert abs(p1.y - 0.0) < 1e-12
    assert abs(p1.theta - 0.0) < 1e-12


def test_bicycle_propagate_reverse_straight_line():
    model = BicycleModel(VehicleParams(wheelbase=2.5))
    p0 = Pose(0.0, 0.0, math.pi / 2.0)
    # p1 = model.propagate(p0, steer=0.0, direction=-1, ds=2.0)
    p1 = model.propagate(p0, u=0.0, direction=-1, ds=2.0, kappa_max=0.2)
    # facing +y; reversing moves -y
    assert abs(p1.x - 0.0) < 1e-12
    assert abs(p1.y + 2.0) < 1e-12


def test_bicycle_propagate_arc_has_expected_heading_change():
    L = 2.5
    # steer = math.radians(20.0)
    u = 0.1  # curvature = 0.1 1/meters
    ds = 1.0
    k0 = 0.0
    model = BicycleModel(VehicleParams(wheelbase=L))
    p0 = Pose(0.0, 0.0, 0.0, kappa=k0)
    p1: Pose = model.propagate(p0, u=u, direction=+1, ds=ds, kappa_max=0.2)
    #!!! FIXME: this will probably always fail since kappa below is derived from steer and L rather than the new approach
    # kappa = math.tan(steer) / L # NOTE: this kappa definition gives a final theta error around 0.028945
    k1 = k0 + u * ds  # new curvature (without clamping) after applying curvature rate u over distance ds
    # yaw update now uses midpoint curvature for better accuracy
    expected_dtheta = 0.5 * (k0 + k1) * ds
    assert abs(p1.theta - expected_dtheta) < 1e-6, f"Expected dtheta near {expected_dtheta}, got {p1.theta}"
    assert abs(p1.kappa - k1) < 1e-9, f"Expected kappa near {k1}, got {p1.kappa}"

