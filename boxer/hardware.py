"""Viam SDK adapter. Only move/open/grab/stop/set_motion_limits actuate anything."""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import CollisionSpecification, Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from .pcd import decode_pcd
from .perception import detect_colors, extract_candidates, project_cloud
from .planning import pose_dict
from .settings import SetupError


@dataclass
class Observation:
    candidates: list
    rejected: list
    objects: list
    detections: dict
    rgb: np.ndarray
    captured_at: float
    joints: np.ndarray
    source_id: str | None = None

    def summary(self):
        return {
            "candidates": [c.summary() for c in self.candidates],
            "source_id": self.source_id,
            "rejected": self.rejected,
            "detection_counts": {k: len(v) for k, v in self.detections.items()},
        }


@asynccontextmanager
async def connection(settings):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    address = os.environ.get("VIAM_ADDRESS", settings.data["address"])
    if address != settings.data["address"]:
        raise SetupError("VIAM_ADDRESS differs from the configured armfarm04 address")
    key, key_id = os.environ.get("VIAM_API_KEY"), os.environ.get("VIAM_API_KEY_ID")
    if not key or not key_id:
        raise SetupError("Set VIAM_API_KEY and VIAM_API_KEY_ID in .env")
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    robot = await RobotClient.at_address(address, options)
    try:
        yield Hardware(robot, settings)
    finally:
        await robot.close()


class Hardware:
    def __init__(self, robot, settings):
        self.robot, self.settings = robot, settings
        d = settings.data
        self.arm = Arm.from_robot(robot, d["arm"])
        self.cam = Camera.from_robot(robot, d["camera"])
        self.gripper = Gripper.from_robot(robot, d["gripper"])
        self.motion = MotionClient.from_robot(robot, d["motion"])

    async def color_preview(self):
        """Color-only monitoring remains useful while another controller moves the arm."""
        images, _ = await self.cam.get_images(filter_source_names=["color"], timeout=15)
        image = next((i for i in images if i.name == "color"), None)
        if image is None:
            raise RuntimeError("Camera did not return its color stream")
        bgr = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Could not decode camera color image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detect_colors(rgb, self.settings)
        return Observation(
            [],
            [{"reason": "Color-only monitoring; no 3D pick coordinates computed"}],
            [],
            detections,
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            time.monotonic(),
            np.array([]),
        )

    async def inspect(self):
        props = await self.cam.get_properties(timeout=15)
        status = await self.robot.get_machine_status()
        from google.protobuf.json_format import MessageToDict

        return {
            "resources": [f"{n.namespace}:{n.type}:{n.subtype}/{n.name}" for n in self.robot.resource_names],
            "camera": MessageToDict(props, preserving_proto_field_name=True),
            "machine_status": MessageToDict(status, preserving_proto_field_name=True),
        }

    async def current_pose(self):
        result = await self.robot.get_pose(self.settings.data["gripper"], "world", timeout=15)
        return pose_dict(result.pose)

    async def layout_snapshot(self):
        """Stationary RGB plus registered scene depth for tabletop/rim-height localization."""
        await self.assert_idle()
        before = await self.joints()
        images, _ = await self.cam.get_images(filter_source_names=["color"], timeout=15)
        image = next((i for i in images if i.name == "color"), None)
        if image is None:
            raise RuntimeError("Camera did not return its color stream")
        bgr = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Could not decode layout image")
        props = await self.cam.get_properties(timeout=15)
        if not props.supports_pcd or any(abs(v) > 1e-8 for v in props.distortion_parameters.parameters):
            raise SetupError("Layout detection requires registered point clouds and zero distortion")
        rotation, origin = await self.camera_matrix(
            props.default_reference_frame or self.settings.data["camera"]
        )
        xyz, _ = decode_pcd((await self.cam.get_point_cloud(timeout=30))[0])
        final_images, _ = await self.cam.get_images(filter_source_names=["color"], timeout=15)
        final_image = next((i for i in final_images if i.name == "color"), None)
        final_bgr = (
            cv2.imdecode(np.frombuffer(final_image.data, np.uint8), cv2.IMREAD_COLOR)
            if final_image is not None
            else None
        )
        if final_bgr is None or final_bgr.shape != bgr.shape:
            raise RuntimeError("Could not verify that the layout scene remained stationary")
        difference = cv2.absdiff(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)
        )
        if np.mean(difference > 30) > 0.02:
            raise RuntimeError(
                "Scene changed during layout capture; keep containers/items stationary and retry"
            )
        after = await self.joints()
        await self.assert_idle()
        if before.shape != after.shape or np.max(np.abs(before - after)) > 0.15:
            raise RuntimeError("Arm moved during layout capture; discard this snapshot")
        return {
            "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            "world_mm": xyz @ rotation.T + origin,
            "rotation": rotation,
            "origin": origin,
            "intrinsics": props.intrinsic_parameters,
            "camera_pose": await self.current_pose(),
            "scene_consistent": True,
        }

    async def assert_ready(self):
        d = self.settings.data
        required = {
            ("component", "arm", d["arm"]),
            ("component", "camera", d["camera"]),
            ("component", "gripper", d["gripper"]),
            ("service", "motion", d["motion"]),
        }
        status = await self.robot.get_machine_status()
        ready = {(r.name.type, r.name.subtype, r.name.name) for r in status.resources if r.state == 3}
        missing = required - ready
        if missing:
            raise RuntimeError("Resources not ready: " + ", ".join("/".join(x) for x in sorted(missing)))

    async def assert_idle(self):
        if await self.arm.is_moving(timeout=10):
            raise RuntimeError("Arm is moving; stop other test scripts before starting Boxer")
        operations = await self.robot.get_operations()
        if any(any(k in o.method.lower() for k in ("move", "grab", "open", "docommand")) for o in operations):
            raise RuntimeError("Another control operation is active on this robot")

    async def joints(self):
        return np.array((await self.arm.get_joint_positions(timeout=10)).values)

    async def camera_matrix(self, camera_frame):
        # Ask Viam to apply the wrist-camera kinematic chain. Avoid hand-built extrinsics.
        probes = ([0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100])
        transformed = []
        for x, y, z in probes:
            result = await asyncio.wait_for(
                self.robot.transform_pose(
                    PoseInFrame(reference_frame=camera_frame, pose=Pose(x=x, y=y, z=z, o_z=1)),
                    "world",
                ),
                timeout=15,
            )
            transformed.append(np.array([result.pose.x, result.pose.y, result.pose.z]))
        origin = transformed[0]
        rotation = np.column_stack([(v - origin) / 100 for v in transformed[1:]])
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.005) or np.linalg.det(rotation) < 0.99:
            raise RuntimeError("Camera frame changed during observation; retry with the arm stationary")
        return rotation, origin

    async def observe(self, source_id=None):
        await self.assert_ready()
        await self.assert_idle()
        before = await self.joints()
        props = await self.cam.get_properties(timeout=15)
        if not props.supports_pcd:
            raise SetupError("Camera must support colored point clouds")
        distortion = props.distortion_parameters
        if any(abs(v) > 1e-8 for v in distortion.parameters):
            raise SetupError("Nonzero camera distortion requires a calibrated projection implementation")
        captured_at = time.monotonic()
        xyz, rgb = decode_pcd((await self.cam.get_point_cloud(timeout=30))[0])
        if len(xyz) < 100:
            raise RuntimeError("Depth capture is empty or too sparse; no pick will be attempted")
        cloud = project_cloud(xyz, rgb, props.intrinsic_parameters)
        rotation, origin = await self.camera_matrix(
            props.default_reference_frame or self.settings.data["camera"]
        )
        world = cloud.camera_mm @ rotation.T + origin
        detections = detect_colors(cloud.rgb, self.settings)
        after = await self.joints()
        await self.assert_idle()
        if before.shape != after.shape or np.max(np.abs(before - after)) > 0.15:
            raise RuntimeError("Arm moved during capture; discard observation")
        candidates, rejected, objects = extract_candidates(cloud, world, detections, self.settings, source_id)
        return Observation(
            candidates, rejected, objects, detections, cloud.rgb, captured_at, after, source_id
        )

    async def set_motion_limits(self):
        # UFactory's documented model-specific commands cap driver speed for planned moves too.
        await self.arm.do_command({"set_speed": 10.0, "set_acceleration": 20.0}, timeout=15)

    async def move(self, pose, state, linear=False):
        limits = self.settings.data["motion_limits"]
        constraints = Constraints()
        if linear:
            constraints = Constraints(
                linear_constraint=[
                    LinearConstraint(
                        line_tolerance_mm=limits["linear_tolerance_mm"], orientation_tolerance_degs=2.0
                    )
                ]
            )
        if state.transforms:
            constraints.collision_specification.append(
                CollisionSpecification(
                    allows=[
                        CollisionSpecification.AllowedFrameCollisions(
                            frame1=self.settings.data["gripper"], frame2="boxer-payload"
                        )
                    ]
                )
            )
        moved = await self.motion.move(
            component_name=self.settings.data["gripper"],
            destination=PoseInFrame(reference_frame="world", pose=Pose(**pose)),
            world_state=state,
            constraints=constraints,
            timeout=limits["timeout_s"],
            extra={"collision_buffer_mm": 3.0},
        )
        if not moved:
            raise RuntimeError("Viam motion planner did not complete the requested move")

    async def holding(self):
        result = await self.gripper.is_holding_something(timeout=10)
        return result.is_holding_something

    async def open(self):
        await self.gripper.open(timeout=15)

    async def grab(self):
        return await self.gripper.grab(timeout=15)

    async def stop(self):
        # Never open on failure or automatically return home while carrying an item.
        results = await asyncio.gather(
            self.arm.stop(timeout=10), self.gripper.stop(timeout=10), return_exceptions=True
        )
        errors = [type(r).__name__ for r in results if isinstance(r, BaseException)]
        if errors:
            raise RuntimeError("Could not confirm stop: " + ", ".join(errors))
