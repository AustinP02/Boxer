import asyncio
import os
import random
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

# Same raised pose as hello_wave; random moves stay within small offsets of it
CENTER = [0, -20, -60, 0, -30, 0]
# Max random offset per joint (degrees): base, shoulder, elbow, wrist roll, wrist pitch, tool roll
LIMITS = [20, 10, 15, 45, 30, 90]

def jp(values):
    return JointPositions(values=values)

def random_pose():
    return [c + random.uniform(-l, l) for c, l in zip(CENTER, LIMITS)]

async def gripper_move(gripper):
    if random.random() < 0.5:
        await gripper.open()
    else:
        await gripper.grab()

async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    machine = await RobotClient.at_address(ADDRESS, opts)
    arm = Arm.from_robot(machine, "arm")
    gripper = Gripper.from_robot(machine, "gripper")

    home = list((await arm.get_joint_positions()).values)
    print("home pose:", [round(v, 1) for v in home])
    step = 0
    try:
        await arm.move_to_joint_positions(jp(CENTER))
        while True:
            step += 1
            pose = random_pose()
            print(f"move {step}:", [round(v) for v in pose])
            await asyncio.gather(arm.move_to_joint_positions(jp(pose)), gripper_move(gripper))
            if random.random() < 0.2:
                await arm.move_to_joint_positions(jp(CENTER))  # occasional reset to center
            await asyncio.sleep(random.uniform(0.1, 0.6))
    finally:
        print("dance over, going home")
        try:
            await gripper.stop()
            await arm.move_to_joint_positions(jp(home))
        finally:
            await arm.stop()
            await machine.close()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
