"""Shape-independent, colored 3D clustering and conservative top-down grasp estimates."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import inside_container, local_points

# OpenCV floating-point HSV uses hue in degrees. Classification runs here in Python.
# Orange/purple and low-saturation pixels remain unknown instead of forcing a sorting color.
COLOR_HUE_RANGES = {
    "red": ((0, 25), (335, 360)),
    "yellow": ((40, 70),),
    "green": ((70, 170),),
    "blue": ((170, 270),),
}


@dataclass
class ColorDetection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_name: str
    confidence: float = 1.0


@dataclass
class CloudImage:
    rgb: np.ndarray
    camera_mm: np.ndarray
    uv: np.ndarray
    colors: np.ndarray


@dataclass
class Candidate:
    color: str
    points_mm: np.ndarray
    center_mm: np.ndarray
    size_mm: np.ndarray
    grasp_mm: np.ndarray
    yaw_deg: float
    width_mm: float
    source_id: str = "source-1"

    def summary(self):
        return {
            "color": self.color,
            "source_id": self.source_id,
            "points": len(self.points_mm),
            "center_mm": self.center_mm.round(2).tolist(),
            "size_mm": self.size_mm.round(2).tolist(),
            "grasp_mm": self.grasp_mm.round(2).tolist(),
            "yaw_deg": round(self.yaw_deg, 2),
            "width_mm": round(self.width_mm, 2),
        }


def project_cloud(xyz, rgb, intrinsics):
    """One registered PCD capture -> RGB image and matching, z-buffered 3D pixels.

    Detectors receive this exact image, avoiding separate RGB/depth capture races.
    RealSense 0.22.6 reports color intrinsics and color-frame PCD coordinates.
    """
    w, h = intrinsics.width_px, intrinsics.height_px
    if not (0 < w <= 4096 and 0 < h <= 4096 and intrinsics.focal_x_px > 0 and intrinsics.focal_y_px > 0):
        raise ValueError("Camera must supply valid color intrinsics")
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz, rgb = xyz[valid], rgb[valid]
    uv = np.rint(
        np.column_stack(
            (
                xyz[:, 0] / xyz[:, 2] * intrinsics.focal_x_px + intrinsics.center_x_px,
                xyz[:, 1] / xyz[:, 2] * intrinsics.focal_y_px + intrinsics.center_y_px,
            )
        )
    )
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    xyz, rgb, uv = xyz[valid], rgb[valid], uv[valid].astype(int)
    pixel_ids = uv[:, 1] * w + uv[:, 0]
    # Sort by pixel, then depth: np.unique takes the nearest sample for each pixel.
    order = np.lexsort((xyz[:, 2], pixel_ids))
    _, unique = np.unique(pixel_ids[order], return_index=True)
    keep = order[unique]
    xyz, rgb, uv = xyz[keep], rgb[keep], uv[keep]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[uv[:, 1], uv[:, 0]] = rgb
    return CloudImage(image, xyz, uv, rgb)


def color_mask(rgb, color, params):
    if len(rgb) == 0:
        return np.zeros(0, dtype=bool)
    hsv = cv2.cvtColor(rgb.reshape(1, -1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2HSV)[0]
    hue_mask = np.zeros(len(rgb), bool)
    for lower, upper in COLOR_HUE_RANGES[color]:
        hue_mask |= (hsv[:, 0] >= lower) & (hsv[:, 0] < upper)
    return hue_mask & (hsv[:, 1] >= params["min_saturation"]) & (hsv[:, 2] >= params["min_value"])


def detect_colors(rgb, settings):
    """Local OpenCV connected-color regions; no Viam vision service or ML inference."""
    height, width = rgb.shape[:2]
    params = settings.data["perception"]
    result = {}
    for color in COLOR_HUE_RANGES:
        mask = color_mask(rgb.reshape(-1, 3), color, params).reshape(height, width).astype(np.uint8)
        # Close small holes in the rendered depth image; 3D clustering still separates objects.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area >= params.get("min_color_pixels", 60):
                boxes.append(ColorDetection(int(x), int(y), int(x + w - 1), int(y + h - 1), color))
        result[color] = boxes
    return result


def cluster_points(points, voxel_mm, radius_mm, min_points):
    """Voxel downsampling bounds neighbor-search cost; split separated same-color items."""
    if not len(points):
        return []
    _, indices = np.unique(np.floor(points / voxel_mm).astype(np.int64), axis=0, return_index=True)
    points = points[indices]
    tree = cKDTree(points)
    seen = np.zeros(len(points), bool)
    result = []
    for start in range(len(points)):
        if seen[start]:
            continue
        seen[start] = True
        stack, component = [start], []
        while stack:
            i = stack.pop()
            component.append(i)
            neighbors = tree.query_ball_point(points[i], radius_mm)
            new = [j for j in neighbors if not seen[j]]
            seen[new] = True
            stack.extend(new)
        if len(component) >= min_points:
            result.append(points[component])
    return result


def grasp_candidate(color, points, settings, source_id="source-1"):
    """Estimate a pinch across the short axis; reject unsupported centers and wide clusters.

    This is a geometric heuristic, not a general bin-picking/grasp neural model.
    """
    d, p = settings.data, settings.data["perception"]
    g = d["grasp"]
    lo, hi = np.quantile(points, (0.02, 0.98), axis=0)
    size = hi - lo
    if max(size[:2]) < p["min_object_mm"] or max(size) > p["max_object_mm"]:
        return None, "object size outside configured limits (possibly touching items)"
    # A robust top surface prevents floor/background points from defining the grasp height.
    top = float(np.quantile(points[:, 2], 0.90))
    surface = points[points[:, 2] >= top - g["insertion_mm"]]
    if len(surface) < 5:
        return None, "insufficient top-surface depth"
    rect = cv2.minAreaRect(surface[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rect).astype(float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmin(lengths))]
    width = float(min(lengths))
    if width < p["min_object_mm"] or width + 2 * g["finger_margin_mm"] > g["max_width_mm"]:
        return None, "grasp width outside configured jaw limits"
    center = np.array(rect[0], dtype=float)
    # A ring, deep concavity, or merged cluster must not send the TCP into an empty center.
    distance, nearest = cKDTree(surface[:, :2]).query(center)
    if distance > g["center_support_mm"]:
        return None, "center has no observed surface; needs another grasp strategy"
    center = surface[nearest, :2]
    source = d["sources"][source_id].get("container")
    if source is None:
        return None, "source container is not measured"
    local_center = local_points([center[0], center[1], top], source)
    if any(
        not source["min_mm"][i] + g["wall_clearance_mm"]
        < local_center[i]
        < source["max_mm"][i] - g["wall_clearance_mm"]
        for i in (0, 1)
    ):
        return None, "target too close to source walls"
    grasp_z = top - g["insertion_mm"]
    if grasp_z < source["min_mm"][2] + g["floor_clearance_mm"]:
        return None, "object too shallow for floor clearance"
    yaw = math.degrees(math.atan2(edge[1], edge[0]))
    return Candidate(
        color, points, (lo + hi) / 2, size, np.array([center[0], center[1], grasp_z]), yaw, width, source_id
    ), None


def extract_candidates(cloud, world_mm, detections, settings, source_id=None):
    """Segment each source independently, even if both share a single camera image."""
    if source_id is not None and source_id not in settings.data["sources"]:
        raise ValueError("Unknown source ID")
    selected = [source_id] if source_id else list(settings.data["sources"])
    accepted, report, objects = [], [], []
    for sid in selected:
        source = settings.data["sources"][sid].get("container")
        if source is None:
            report.append({"source_id": sid, "reason": "source container is not measured"})
            continue
        candidates, rejected, scene_objects = _extract_source(cloud, world_mm, detections, settings, sid)
        accepted.extend(candidates)
        report.extend({"source_id": sid, **r} for r in rejected)
        objects.extend(scene_objects)
    # Ordering here carries no preference: the controller chooses randomly from the eligible pool.
    return accepted, report, objects


def _extract_source(cloud, world_mm, detections, settings, source_id):
    d, p = settings.data, settings.data["perception"]
    source = d["sources"][source_id]["container"]
    report, candidates, objects = [], [], []
    claimed = np.zeros(len(world_mm), bool)
    roi = inside_container(world_mm, source)
    for color in d["colors"]:
        in_box = np.zeros(len(world_mm), bool)
        for box in detections[color]:
            in_box |= (
                (cloud.uv[:, 0] >= box.x_min)
                & (cloud.uv[:, 0] <= box.x_max)
                & (cloud.uv[:, 1] >= box.y_min)
                & (cloud.uv[:, 1] <= box.y_max)
            )
        mask = roi & in_box & color_mask(cloud.colors, color, p)
        # Only claim retained clusters: small/sparse colored material remains unknown occupancy.
        for points in cluster_points(world_mm[mask], p["voxel_mm"], p["cluster_radius_mm"], p["min_points"]):
            objects.append((color, points))
            distances, _ = cKDTree(points).query(world_mm[mask])
            selected_indices = np.flatnonzero(mask)
            claimed[selected_indices[distances <= p["voxel_mm"] * 2]] = True
            candidate, reason = grasp_candidate(color, points, settings, source_id)
            if candidate:
                candidates.append(candidate)
            else:
                report.append({"color": color, "reason": reason, "points": len(points)})
    unknown = roi & ~claimed & (world_mm[:, 2] > source["min_mm"][2] + 5)
    for points in cluster_points(world_mm[unknown], p["voxel_mm"], p["cluster_radius_mm"], p["min_points"]):
        objects.append(("unknown", points))
    if len(objects) > 100:
        raise RuntimeError("Source scene is too fragmented for reliable planning; inspect depth/lighting")
    scene = world_mm[roi]
    accepted = []
    for c in candidates:
        distance, _ = cKDTree(c.points_mm).query(scene)
        other = scene[distance > p["voxel_mm"] * 2]
        same_height = other[other[:, 2] >= c.grasp_mm[2] - 2]
        if len(same_height):
            gaps, _ = cKDTree(c.points_mm).query(same_height)
            if np.min(gaps) < d["grasp"]["neighbor_clearance_mm"]:
                report.append({"color": c.color, "reason": "neighbor blocks finger clearance"})
                continue
        accepted.append(c)
    return accepted, report, objects
