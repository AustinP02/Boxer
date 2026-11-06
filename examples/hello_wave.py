import asyncio
import os
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.component.arm import JointPositions

from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

ADDRESS = os.environ["VIAM_ADDRESS"]
API_KEY = os.environ["VIAM_API_KEY"]
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]

# Joint angles in degrees: [J1 base, J2 shoulder, J3 elbow, J4 wrist roll, J5 wrist pitch, J6 tool roll]
UP = [0, -20, -60, 0, -30, 0]
WRIST_AMPLITUDE = 25   # J5 up/down wave
SIDE_AMPLITUDE = 20    # J1 base left/right wave
WAVES = 3
PAUSE_S = 0.5

def jp(values):
    return JointPositions(values=values)

def offset(base, joint, delta):
    p = list(base)
    p[joint] += delta
    return p

async def wave_with_gripper(arm, gripper, joint, amplitude):
    for _ in range(WAVES):
        await asyncio.gather(
            arm.move_to_joint_positions(jp(offset(UP, joint, amplitude))),
            gripper.open(),
        )
        await asyncio.gather(
            arm.move_to_joint_positions(jp(offset(UP, joint, -amplitude))),
            gripper.grab(),
        )

async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    machine = await RobotClient.at_address(ADDRESS, opts)
    arm = Arm.from_robot(machine, "arm")
    gripper = Gripper.from_robot(machine, "gripper")

    down = list((await arm.get_joint_positions()).values)
    print("down pose:", [round(v, 1) for v in down])
    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"cycle {cycle}: hands up")
            await arm.move_to_joint_positions(jp(UP))

            print(f"cycle {cycle}: wrist wave")
            await wave_with_gripper(arm, gripper, 4, WRIST_AMPLITUDE)
            await arm.move_to_joint_positions(jp(UP))

            print(f"cycle {cycle}: left-right wave")
            await wave_with_gripper(arm, gripper, 0, SIDE_AMPLITUDE)
            await arm.move_to_joint_positions(jp(UP))

            print(f"cycle {cycle}: down")
            await gripper.open()
            await arm.move_to_joint_positions(jp(down))
            await asyncio.sleep(PAUSE_S)
    finally:
        print("returning to down pose")
        try:
            await gripper.stop()
            await arm.move_to_joint_positions(jp(down))
        finally:
            await arm.stop()
            await machine.close()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
