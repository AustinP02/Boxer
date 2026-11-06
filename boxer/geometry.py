"""Rectangular container geometry, including yaw about the world Z axis."""

import math

import numpy as np


def rotation_xy(box):
    angle = math.radians(box.get("yaw_deg", 0))
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def local_points(points, box):
    points = np.asarray(points, dtype=float)
    center = (np.array(box["min_mm"][:2]) + box["max_mm"][:2]) / 2
    result = points.copy()
    result[..., :2] = (points[..., :2] - center) @ rotation_xy(box) + center
    return result


def inside_container(points, box):
    points = local_points(points, box)
    return ((points > np.array(box["min_mm"])) & (points < np.array(box["max_mm"]))).all(axis=-1)


def xy_corners(box, outer=False):
    lo, hi = np.array(box["min_mm"][:2]), np.array(box["max_mm"][:2])
    center = (lo + hi) / 2
    if outer:
        lo, hi = lo - box["wall_mm"], hi + box["wall_mm"]
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return (corners - center) @ rotation_xy(box).T + center


def footprints_overlap(a, b):
    p, q = xy_corners(a, outer=True), xy_corners(b, outer=True)
    for polygon in (p, q):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.array([-edge[1], edge[0]])
            if max(p @ axis) <= min(q @ axis) + 1e-7 or max(q @ axis) <= min(p @ axis) + 1e-7:
                return False
    return True
