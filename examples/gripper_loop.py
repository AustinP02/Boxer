import asyncio
import os
from viam.robot.client import RobotClient
from viam.components.gripper import Gripper

ADDRESS = os.environ["VIAM_ADDRESS"]
API_KEY = os.environ["VIAM_API_KEY"]        # the long random string (e.g. rtr99...)
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]  # the UUID with dashes

PAUSE_S = 1.0

async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    machine = await RobotClient.at_address(ADDRESS, opts)
    gripper = Gripper.from_robot(machine, "gripper")
    cycle = 0
    try:
        while True:
            cycle += 1
            await gripper.open()
            print(f"cycle {cycle}: opened")
            await asyncio.sleep(PAUSE_S)
            grabbed = await gripper.grab()
            #print(f"cycle {cycle}: closed (holding: {grabbed})")
            await asyncio.sleep(PAUSE_S)
    finally:
        await gripper.stop()
        await machine.close()
        print("stopped")

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
