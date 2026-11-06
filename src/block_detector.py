"""Color-based object proposals; centers are image pixels, not robot coordinates."""
import cv2
import numpy as np
import json
from pathlib import Path
from functools import lru_cache

COLORS = {
    'red': ([(0, 12), (170, 179)], (0, 0, 255)),
    'yellow': ([(18, 38)], (0, 255, 255)),
    'blue': ([(90, 130)], (255, 120, 0)),
    'green': ([(39, 89)], (0, 200, 0)),
}


@lru_cache(maxsize=1)
def shape_templates():
    from project_paths import MODELS_DIR
    path = MODELS_DIR / 'shape_templates.json'
    return json.loads(path.read_text()) if path.exists() else []


def shape_features(contour):
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return dict(solidity=area / max(hull_area, 1),
                aspect=max(w, h) / max(min(w, h), 1))


def classify_shape(contour):
    """Compare silhouette geometry across ALL shape classes, independent of color.

    Unknown is intentional for overlapping or unseen silhouettes. A match is
    a 2D shape hypothesis, not proof of an object's 3D geometry.
    """
    features = shape_features(contour)
    scores = {}
    for template in shape_templates():
        reference = np.array(template['contour'], dtype=np.int32)
        other = shape_features(reference)
        score = cv2.matchShapes(contour, reference, cv2.CONTOURS_MATCH_I1, 0)
        score += 0.2 * abs(np.log(features['aspect'] / other['aspect']))
        score += abs(features['solidity'] - other['solidity'])
        label = template['shape']
        scores[label] = min(scores.get(label, float('inf')), float(score))
    ranked = sorted(scores.items(), key=lambda pair: pair[1])
    if not ranked:
        return 'unknown', None
    label, score = ranked[0]
    margin = ranked[1][1] - score if len(ranked) > 1 else 0
    if score > 0.22 or margin < 0.025:
        label = 'ambiguous'
    return label, round(score, 4)


def detect_blocks(frame, min_area=800, roi=None):
    """Return color, bbox, centroid and area for separated colored regions.

    ROI is (x, y, width, height). Same-color touching objects may merge.
    These are color candidates, not a learned semantic block classifier.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), dtype=np.uint8)
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    detections = []
    for color, (ranges, _) in COLORS.items():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, (low, 90, 60), (high, 255, 255))
        if roi is not None:
            x, y, w, h = roi
            region = np.zeros_like(mask)
            region[y:y+h, x:x+w] = 255
            mask &= region
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        combined |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # Reject thin streaks and sparse, irregular color noise.
            if min(w, h) < 15 or area / (w * h) < 0.45:
                continue
            moments = cv2.moments(contour)
            if not moments['m00']:
                continue
            center = (int(moments['m10'] / moments['m00']),
                      int(moments['m01'] / moments['m00']))
            shape, score = classify_shape(contour)
            clipped = (x <= 1 or y <= 1 or x+w >= frame.shape[1]-1
                       or y+h >= frame.shape[0]-1)
            if roi is not None:
                rx, ry, rw, rh = roi
                clipped |= x <= rx+1 or y <= ry+1 or x+w >= rx+rw-1 or y+h >= ry+rh-1
            detections.append(dict(color=color, bbox=(x, y, w, h),
                                   center=center, area=area,
                                   shape='partial' if clipped else shape,
                                   shape_distance=score))
    return sorted(detections, key=lambda d: d['area'], reverse=True), combined


def draw_detections(frame, detections, roi=None):
    output = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(output, (x, y), (x+w, y+h), (180, 180, 180), 1)
    for item in detections:
        x, y, w, h = item['bbox']
        cx, cy = item['center']
        ink = COLORS[item['color']][1]
        cv2.rectangle(output, (x, y), (x+w, y+h), ink, 2)
        cv2.drawMarker(output, (cx, cy), ink, cv2.MARKER_CROSS, 18, 2)
        text = f"{item['color']}: {item['shape']}?"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = max(0, min(x, output.shape[1]-tw-5))
        ty = max(th+5, y-8)
        cv2.rectangle(output, (tx, ty-th-4), (tx+tw+4, ty+baseline+2), (20, 20, 20), -1)
        cv2.putText(output, text, (tx+2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(output, f'{cx},{cy} px', (x, min(output.shape[0]-5, y+h+18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return output
