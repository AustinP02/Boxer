"""Preview camera positions for a top-to-side arc. Does NOT move hardware.

Coordinates: user's chosen world frame, mm. x left/right, y back/front,
z up. Camera optical axis must separately be pointed at the target and its
mount offset transformed into an arm tool goal before execution.
"""
import argparse
import json
import math


def inspection_arc(target, far_radius, near_radius, side_offset, count=9):
    if count < 3 or not all(math.isfinite(v) for v in
                           (*target, far_radius, near_radius, side_offset)):
        raise ValueError('Use finite values and at least three viewpoints')
    if near_radius <= 0 or far_radius < near_radius or side_offset < 0:
        raise ValueError('Require far >= near > 0 and side >= 0')
    points = []
    for i in range(count):
        t = i / (count-1)
        # Stop 15 degrees above horizontal; target is still below the camera.
        angle = math.radians(75) * t
        radius = far_radius + (near_radius-far_radius) * t
        x = target[0] + side_offset * math.sin(2*math.pi*t)
        y = target[1] - radius * math.sin(angle)
        z = target[2] + radius * math.cos(angle)
        points.append(dict(view=i, camera_mm=[round(x, 2), round(y, 2), round(z, 2)],
                           look_at_mm=list(target)))
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=float, nargs=3, required=True)
    parser.add_argument('--far', type=float, required=True)
    parser.add_argument('--near', type=float, required=True)
    parser.add_argument('--side', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(dict(status='PREVIEW ONLY: unvalidated camera positions',
                          views=inspection_arc(args.target, args.far, args.near, args.side)), indent=2))
