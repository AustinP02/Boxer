"""Viam camera with local color detection. Q quits; R selects workspace ROI."""
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from block_detector import detect_blocks, draw_detections

from project_paths import PROJECT_ROOT, OUTPUT_DIR
load_dotenv(PROJECT_ROOT / '.env')


async def get_color_frame(cam):
    images, _ = await cam.get_images()
    # Prefer the actual color stream before considering other encoded images.
    ordered = sorted(images, key=lambda image: image.name != 'color')
    for image in ordered:
        mime = str(image.mime_type).lower()
        if 'jpeg' not in mime and 'png' not in mime:
            continue
        frame = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    raise RuntimeError('Camera returned no decodable JPEG/PNG color image.')


async def main():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ['VIAM_API_KEY'], api_key_id=os.environ['VIAM_API_KEY_ID'])
    machine = await RobotClient.at_address(os.environ['VIAM_ADDRESS'], opts)
    roi = None
    show_mask = False
    capture_index = 0
    try:
        cam = Camera.from_robot(machine, 'cam')
        print('Q: quit | R: table region | C: clear | M: mask | S: save inspection view')
        print('Shape labels are silhouette hypotheses. No robot movement is issued.')
        while True:
            frame = await get_color_frame(cam)
            detections, mask = detect_blocks(frame, roi=roi)
            display = draw_detections(frame, detections, roi)
            cv2.putText(display, f'Shape/color hypotheses: {len(detections)}', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.imshow('cam', display)
            if show_mask:
                cv2.imshow('color mask', mask)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or key == 27:
                break
            if key == ord('r'):
                selected = cv2.selectROI('cam', frame, showCrosshair=True)
                if selected[2] and selected[3]:
                    roi = tuple(map(int, selected))
            if key == ord('c'):
                roi = None
            if key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow('color mask')
            if key == ord('s'):
                folder = OUTPUT_DIR / 'scan_captures'
                folder.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                stem = folder / stamp
                if not cv2.imwrite(str(stem.with_suffix('.jpg')), frame):
                    raise RuntimeError('Failed to save inspection image')
                record = dict(view=capture_index, detections=detections,
                              roi=roi, coordinates='image pixels; no depth',
                              note='Independent view; no cross-view object association')
                stem.with_suffix('.json').write_text(json.dumps(record, indent=2))
                capture_index += 1
                print(f'Saved view {capture_index}: {stem.name}')
            await asyncio.sleep(0.02)
    finally:
        cv2.destroyAllWindows()
        await machine.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
