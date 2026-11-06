"""Review shape/color detection offline, without Viam or robot movement."""
import argparse
import json
from pathlib import Path
import cv2
from block_detector import detect_blocks, draw_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input and output must be different files')
    if args.input.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        frame = cv2.imread(str(args.input))
        if frame is None:
            raise ValueError('Cannot read input image')
        detections, _ = detect_blocks(frame)
        if not cv2.imwrite(str(args.output), draw_detections(frame, detections)):
            raise ValueError('Cannot write output image')
        print(json.dumps(detections, indent=2))
        return
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError('Cannot open input video')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        capture.release()
        raise ValueError('Cannot open output video writer')
    count = 0
    try:
        with args.output.with_suffix('.jsonl').open('w') as report:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections, _ = detect_blocks(frame)
                writer.write(draw_detections(frame, detections))
                report.write(json.dumps(dict(frame=count, detections=detections))+'\n')
                count += 1
    finally:
        capture.release()
        writer.release()
    print(f'Processed {count} frames')


if __name__ == '__main__':
    main()
