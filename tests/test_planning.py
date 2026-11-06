import math

import numpy as np
import pytest

from boxer.planning import container_geometry, waypoints, world_state
from boxer.settings import SetupError


def test_workcell_requires_real_measurements(settings):
    settings.validate_workcell()
    settings.data["sources"]["source-1"]["container"] = None
    with pytest.raises(SetupError, match="source"):
        settings.validate_workcell()


def test_container_model_is_hollow(settings):
    g = container_geometry("source", settings.data["sources"]["source-1"]["container"])
    assert len(g) == 5
    assert g[0].center.z == -2.5
    assert g[0].box.dims_mm.z == 5
    assert all(geom.box.dims_mm.z == 100 for geom in g[1:])


def test_payload_attached_to_gripper_and_only_target_removed(settings, candidate):
    other = candidate.points_mm + [70, 0, 0]
    state = world_state(settings, [("blue", candidate.points_mm), ("blue", other)], candidate, True)
    assert state.transforms[0].pose_in_observer_frame.reference_frame == "gripper"
    labels = [g.label for g in state.obstacles[0].geometries]
    assert "boxer-object-0-blue" not in labels
    assert "boxer-object-1-blue" in labels
    assert state.transforms[0].physical_object.box.dims_mm.z > 0


def test_top_down_ov_theta_matches_finger_closing_yaw(settings, candidate):
    points = waypoints(candidate, settings, "bin-1")
    pose = points["pick"]
    theta = math.radians(pose["theta"])
    local_x = np.array([-math.cos(theta), math.sin(theta)])
    target_axis = np.array(
        [math.cos(math.radians(candidate.yaw_deg)), math.sin(math.radians(candidate.yaw_deg))]
    )
    assert abs(np.dot(local_x, target_axis)) == pytest.approx(1)
    assert pose["o_z"] == -1
    assert points["lift"]["z"] == settings.data["travel_z_mm"]


def test_low_drop_would_hit_bin_rim(settings):
    settings.data["bins"]["bin-4"]["drop_pose"]["z"] = 120
    with pytest.raises(SetupError, match="rim"):
        settings.validate_workcell()
