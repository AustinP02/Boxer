from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.proto.component.camera import GetPropertiesResponse, IntrinsicParameters

from boxer.hardware import Hardware
from boxer.planning import world_state


async def test_camera_transform_uses_viam_world_chain(settings):
    hw = Hardware.__new__(Hardware)
    hw.settings = settings
    matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    origin = np.array([200, 100, 400])

    async def transform(source, destination):
        assert source.reference_frame == "cam" and destination == "world"
        p = source.pose
        v = matrix @ [p.x, p.y, p.z] + origin
        return PoseInFrame(reference_frame="world", pose=Pose(x=v[0], y=v[1], z=v[2], o_z=1))

    hw.robot = SimpleNamespace(transform_pose=transform)
    actual, translation = await hw.camera_matrix("cam")
    np.testing.assert_allclose(actual, matrix)
    np.testing.assert_allclose(translation, origin)


async def test_motion_uses_tcp_world_state_and_linear_constraints(settings, candidate):
    hw = Hardware.__new__(Hardware)
    hw.settings = settings
    hw.motion = SimpleNamespace(move=AsyncMock(return_value=True))
    state = world_state(settings, target=candidate, carrying=True)
    await hw.move(settings.data["sources"]["source-1"]["observe_pose"], state, linear=True)
    kw = hw.motion.move.await_args.kwargs
    assert kw["component_name"] == "gripper"
    assert kw["destination"].reference_frame == "world"
    assert kw["world_state"] is state
    constraint = kw["constraints"]
    assert constraint.linear_constraint[0].line_tolerance_mm == 2
    allowed = constraint.collision_specification[0].allows[0]
    assert (allowed.frame1, allowed.frame2) == ("gripper", "boxer-payload")


async def test_failed_motion_result_is_an_error(settings):
    hw = Hardware.__new__(Hardware)
    hw.settings = settings
    hw.motion = SimpleNamespace(move=AsyncMock(return_value=False))
    with pytest.raises(RuntimeError, match="did not complete"):
        await hw.move(settings.data["sources"]["source-1"]["observe_pose"], world_state(settings))


async def test_full_rgbd_observation_uses_local_opencv_and_world_transform(settings):
    """Color detection and 3D filtering are local; no Viam vision client is required."""

    hw = Hardware.__new__(Hardware)
    hw.settings = settings
    hw.assert_ready = AsyncMock()
    hw.arm = SimpleNamespace(
        is_moving=AsyncMock(return_value=False),
        get_joint_positions=AsyncMock(return_value=JointPositions(values=[0] * 6)),
    )
    x, y = np.meshgrid(np.arange(-0.02, 0.021, 0.002), np.arange(-0.01, 0.011, 0.002))
    pcd = np.zeros(x.size, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    pcd["x"], pcd["y"], pcd["z"], pcd["rgb"] = x.ravel(), y.ravel(), 0.5, 0x0000FF
    header = (
        f"VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
        f"WIDTH {len(pcd)}\nHEIGHT 1\nPOINTS {len(pcd)}\nDATA binary\n"
    ).encode()
    props = GetPropertiesResponse(
        supports_pcd=True,
        default_reference_frame="cam",
        intrinsic_parameters=IntrinsicParameters(
            width_px=100, height_px=100, focal_x_px=500, focal_y_px=500, center_x_px=50, center_y_px=50
        ),
    )
    hw.cam = SimpleNamespace(
        get_properties=AsyncMock(return_value=props),
        get_point_cloud=AsyncMock(return_value=(header + pcd.tobytes(), "pointcloud/pcd")),
    )

    async def transform(source, destination):
        p = source.pose
        return PoseInFrame(reference_frame="world", pose=Pose(x=150 + p.x, y=150 - p.y, z=550 - p.z, o_z=1))

    hw.robot = SimpleNamespace(get_operations=AsyncMock(return_value=[]), transform_pose=transform)

    observation = await hw.observe()
    assert len(observation.candidates) == 1
    c = observation.candidates[0]
    assert c.color == "blue"
    np.testing.assert_allclose(c.center_mm, [150, 150, 50], atol=settings.data["perception"]["voxel_mm"])
    assert c.grasp_mm[2] == pytest.approx(38)
