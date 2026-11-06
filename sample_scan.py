"""Small xArm6 calibration sweep, relative to its CURRENT world pose.

No arguments: inspect setup and print waypoints. --execute: physically move.
Keeps wrist orientation fixed. Does not perform a full top-to-side scan.
"""
import argparse
import asyncio
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame, GeometriesInFrame, WorldState
from live_feed import get_color_frame
from block_detector import detect_blocks, draw_detections

ROOT = Path(__file__).resolve().parent
OBSTACLES = ('table', 'wall-front', 'wall-side', 'ceiling')


def scan_poses(start, travel_y=30, drop_z=15, side_x=5, steps=7):
    if (not all(math.isfinite(v) for v in (travel_y, drop_z, side_x))
            or not 0 < abs(travel_y) <= 50 or not 0 <= drop_z <= 25
            or not 0 <= side_x <= 10 or not 3 <= steps <= 15):
        raise ValueError('Sample limits: 0<abs(Y)<=50, 0<=Z drop<=25, 0<=X<=10 mm, 3..15 views')
    result = []
    for i in range(steps):
        t = i/(steps-1)
        angle = math.pi*t/2
        goal = Pose()
        goal.CopyFrom(start)
        goal.x += side_x*math.sin(2*math.pi*t)
        goal.y += travel_y*math.sin(angle)
        goal.z -= drop_z*(1-math.cos(angle))
        result.append(goal)
    return result


async def inspect_setup(machine, motion):
    frames = await asyncio.wait_for(machine.get_frame_system_config(), timeout=20)
    transforms = {item.frame.reference_frame: item.frame for item in frames}
    for name in ('arm', 'cam', *OBSTACLES):
        if name not in transforms:
            raise RuntimeError(f'{name}: missing frame configuration; configure it in Viam first')
    # The camera must be on the arm's descendant chain, not fixed in world.
    parent = transforms['cam'].pose_in_observer_frame.reference_frame
    seen = set()
    while parent != 'arm':
        if parent in seen or parent not in transforms:
            raise RuntimeError('Camera frame is not attached beneath arm; correct its frame parent')
        seen.add(parent)
        parent = transforms[parent].pose_in_observer_frame.reference_frame
    obstacles = []
    info = {}
    for name in OBSTACLES:
        pose = await motion.get_pose(name, 'world', timeout=15)
        geometries = await Gripper.from_robot(machine, name).get_geometries(timeout=15)
        if not geometries:
            raise RuntimeError(f'{name}: no collision geometry returned')
        info[name] = dict(world_pose=MessageToDict(pose), geometry_count=len(geometries))
        obstacles.append(GeometriesInFrame(reference_frame=name, geometries=geometries))
    info['camera_frame'] = MessageToDict(transforms['cam'])
    info['world_z_note'] = 'Verify world +Z is physically up before execution'
    return WorldState(obstacles=obstacles), info


async def run(args):
    load_dotenv(ROOT / '.env')
    options = RobotClient.Options.with_api_key(
        api_key=os.environ['VIAM_API_KEY'], api_key_id=os.environ['VIAM_API_KEY_ID'])
    machine = await asyncio.wait_for(
        RobotClient.at_address(os.environ['VIAM_ADDRESS'], options), timeout=30)
    arm = None
    motion_attempted = False
    try:
        arm = Arm.from_robot(machine, 'arm')
        camera = Camera.from_robot(machine, 'cam')
        motion = MotionClient.from_robot(machine, 'builtin')
        world, setup = await inspect_setup(machine, motion)
        current = await motion.get_pose('arm', 'world', timeout=15)
        poses = scan_poses(current.pose, args.travel_y, args.drop_z, args.side_x)
        print(json.dumps(dict(setup=setup, start=MessageToDict(current),
                              goals=[MessageToDict(p) for p in poses]), indent=2))
        # Check camera availability before issuing any arm commands.
        await asyncio.wait_for(get_color_frame(camera), timeout=20)
        if not args.execute:
            print('INSPECTION ONLY. Add --execute to run the physical sample scan.')
            return
        if args.clearance_mm is None or not math.isfinite(args.clearance_mm) or args.clearance_mm < args.drop_z+50:
            raise ValueError('Supply measured --clearance-mm, at least Z drop + 50 mm, below the lowest tool/camera part')
        folder = ROOT / 'scan_captures' / datetime.now().strftime('sample_%Y%m%d_%H%M%S_%f')
        folder.mkdir(parents=True)
        (folder/'setup.json').write_text(json.dumps(setup, indent=2))
        # Documented xArm module command. Fail rather than silently using a fast default.
        await arm.do_command({'set_speed': 5.0}, timeout=15)
        print('xArm joint speed set to 5 deg/s (remains set after scan). Ctrl+C stops.')
        print('Starting sample; orientation held fixed; no gripper actuation or automatic return.')
        for index, goal in enumerate(poses):
            if index:
                print(f'Moving to view {index+1}/{len(poses)}', flush=True)
                motion_attempted = True
                success = await motion.move(
                    component_name='arm',
                    destination=PoseInFrame(reference_frame='world', pose=goal),
                    world_state=world,
                    extra={'motion_profile': 'linear'}, timeout=60)
                if not success:
                    raise RuntimeError('Motion service did not confirm success')
            await asyncio.sleep(0.5)
            actual = await motion.get_pose('arm', 'world', timeout=15)
            error = math.sqrt(sum((getattr(actual.pose, axis)-getattr(goal, axis))**2
                                  for axis in ('x', 'y', 'z')))
            if error > 5:
                raise RuntimeError(f'Arm is {error:.1f} mm from requested waypoint; stopping')
            frame = await asyncio.wait_for(get_color_frame(camera), timeout=20)
            detections, _ = detect_blocks(frame)
            stem = folder/f'view_{index:02d}'
            if not cv2.imwrite(str(stem.with_suffix('.jpg')), frame):
                raise RuntimeError('Failed to save frame')
            if not cv2.imwrite(str(folder/f'view_{index:02d}_overlay.jpg'), draw_detections(frame, detections)):
                raise RuntimeError('Failed to save annotated frame')
            stem.with_suffix('.json').write_text(json.dumps(dict(
                world_tool_pose=MessageToDict(actual), detections=detections,
                note='Pixel detections; no calibrated 3D reconstruction'), indent=2))
            print(f'Saved view {index+1}; position error {error:.2f} mm', flush=True)
        print(f'Scan complete. Arm remains at final view. Images: {folder}')
    finally:
        try:
            if arm is not None and motion_attempted:
                await arm.stop(timeout=10)
        finally:
            await machine.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--travel-y', type=float, default=30, help='Signed world Y travel, mm')
    parser.add_argument('--drop-z', type=float, default=15)
    parser.add_argument('--side-x', type=float, default=5)
    parser.add_argument('--clearance-mm', type=float,
                        help='Measured free space below lowest tool/camera part at start')
    args = parser.parse_args()
    # Reject invalid paths before any connection.
    scan_poses(Pose(), args.travel_y, args.drop_z, args.side_x)
    return args


if __name__ == '__main__':
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print('Stopped by operator. No return motion commanded.')
    except Exception as error:
        raise SystemExit(f'Scan stopped: {error}') from None
