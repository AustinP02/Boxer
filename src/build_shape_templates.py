"""Build silhouette references from the user's labeled example objects."""
import json
import sys
from pathlib import Path
import cv2
from block_detector import detect_blocks
from project_paths import MODELS_DIR


def collect(frame, labels, output, source):
    detections, mask = detect_blocks(frame, min_area=500)
    # In these specific examples: red cuboid; blue/yellow cylinders; green notch.
    # Color selects the labeled TRAINING instance only, never inference class.
    for color, shape in labels.items():
        candidates = [d for d in detections if d['color'] == color]
        if not candidates:
            continue
        d = max(candidates, key=lambda d: d['area'])
        x, y, w, h = d['bbox']
        contours, _ = cv2.findContours(mask[y:y+h, x:x+w], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        output.append(dict(shape=shape, source=source, contour=contour.tolist()))


if __name__ == '__main__':
    video, photo = sys.argv[1:3]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('Cannot open reference video')
    templates = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in range(0, count, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            collect(frame, {'red': 'cuboid', 'blue': 'cylinder', 'yellow': 'cylinder'},
                    templates, f'example-video-frame-{index}')
    cap.release()
    frame = cv2.imread(photo)
    if frame is None:
        raise SystemExit('Cannot open reference photo')
    collect(frame, {'green': 'notched'}, templates, 'example-photo')
    (MODELS_DIR / 'shape_templates.json').write_text(json.dumps(templates))
    print(f'Saved {len(templates)} shape references')
