import asyncio
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.camera import Camera

load_dotenv()

ADDRESS = os.environ["VIAM_ADDRESS"]
API_KEY = os.environ["VIAM_API_KEY"]
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]

async def get_color_frame(cam):
    images, _ = await cam.get_images()
    for img in images:
        if img.name == "color" or img.mime_type == "image/jpeg":
            return cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
    return None

async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    machine = await RobotClient.at_address(ADDRESS, opts)
    cam = Camera.from_robot(machine, "cam")
    try:
        while True:
            frame = await get_color_frame(cam)
            if frame is not None:
                # process the frame here (e.g. detect objects)
                cv2.imshow("cam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        await machine.close()

asyncio.run(main())
