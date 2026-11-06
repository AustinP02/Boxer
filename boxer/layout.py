"""Height-constrained detection of the six transparent rectangular containers.

RGB rim lines locate the containers. Table depth anchors metric coordinates; clear
wall depth is never used as a rim measurement. Results are proposals, not motion approval.
"""

import copy
import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .settings import BIN_IDS, SOURCE_IDS, Settings, SetupError


@dataclass
class TablePlane:
    normal: np.ndarray
    offset: float
    rms_mm: float
    support: float

    def z_at(self, x, y):
        return float(-(self.offset + self.normal[0] * x + self.normal[1] * y) / self.normal[2])


def fit_table_plane(points):
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 300:
        raise SetupError("Insufficient tabletop depth; show more exposed table")
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(12000, len(points)), replace=False)]
    best = np.zeros(len(sample), bool)
    for _ in range(180):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        if abs(n[2]) < math.cos(math.radians(3)):
            continue
        mask = np.abs((sample - a) @ n) < 2.5
        if mask.sum() > best.sum():
            best = mask
    support = float(best.mean())
    if support < 0.25:
        raise SetupError("No reliable horizontal tabletop plane; check camera frame and visible table")
    inliers = sample[best]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(-normal @ center)
    rms = float(np.sqrt(np.mean((inliers @ normal + offset) ** 2)))
    if normal[2] < math.cos(math.radians(3)) or rms > 2:
        raise SetupError("Table plane is too tilted/noisy for the current container model")
    return TablePlane(normal, offset, rms, support)


def pixels_on_rim(pixels, intrinsics, rotation, origin, table, height_mm):
    pixels = np.asarray(pixels, dtype=float)
    rays = (
        np.column_stack(
            (
                (pixels[:, 0] - intrinsics.center_x_px) / intrinsics.focal_x_px,
                (pixels[:, 1] - intrinsics.center_y_px) / intrinsics.focal_y_px,
                np.ones(len(pixels)),
            )
        )
        @ rotation.T
    )
    denominator = rays @ table.normal
    if np.any(np.abs(denominator) < 1e-6):
        raise SetupError("Camera view is parallel to container rims")
    # The supplied heights are vertical distance from table to top rim.
    shifted_offset = table.offset - table.normal[2] * height_mm
    distance = -(table.normal @ origin + shifted_offset) / denominator
    if np.any(distance <= 0):
        raise SetupError("Container rim plane is behind the camera")
    return origin + rays * distance[:, None]


def world_to_pixels(points, intrinsics, rotation, origin):
    camera = (points - origin) @ rotation
    if np.any(camera[:, 2] <= 0):
        raise SetupError("Detected rectangle is behind the camera")
    return np.column_stack(
        (
            camera[:, 0] / camera[:, 2] * intrinsics.focal_x_px + intrinsics.center_x_px,
            camera[:, 1] / camera[:, 2] * intrinsics.focal_y_px + intrinsics.center_y_px,
        )
    )


def rim_lines(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 45)
    # Colored block edges must not be mistaken for the clear container boundaries.
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    colored = cv2.dilate((saturation > 130).astype(np.uint8), np.ones((7, 7), np.uint8))
    edges[colored > 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=30, minLineLength=45, maxLineGap=25)
    if lines is None:
        return np.empty((0, 2, 2))
    return lines.reshape(-1, 2, 2).astype(float)


def _groups(segments, vertical):
    direction, along = (0, 1) if vertical else (1, 0)
    rows = []
    for segment in segments:
        delta = segment[1] - segment[0]
        if abs(delta[direction]) > abs(delta[along]) * math.tan(math.radians(5)):
            continue
        rows.append((float(segment[:, direction].mean()), tuple(sorted(segment[:, along]))))
    rows.sort()
    groups = []
    for position, interval in rows:
        if groups and abs(position - groups[-1][0]) <= 2:
            old, spans = groups[-1]
            groups[-1] = ((old * len(spans) + position) / (len(spans) + 1), [*spans, interval])
        else:
            groups.append((position, [interval]))
    return groups


def _coverage(spans, lower, upper):
    clipped = sorted((max(a, lower), min(b, upper)) for a, b in spans if b > lower and a < upper)
    end, total = lower, 0.0
    for a, b in clipped:
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total / (upper - lower)


def _overlap(a, b):
    # Double-precision convex clipping also handles coincident/shared polygon edges.
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def signed_area(poly):
        return sum(cross(p, q) for p, q in zip(poly, np.roll(poly, -1, axis=0))) / 2

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if signed_area(a) < 0:
        a = a[::-1]
    if signed_area(b) < 0:
        b = b[::-1]
    output = list(a)
    for start, end in zip(b, np.roll(b, -1, axis=0)):
        if not output:
            break
        edge = end - start
        incoming, output = output, []
        previous = incoming[-1]
        for current in incoming:
            old_inside = cross(edge, previous - start) >= -1e-8
            new_inside = cross(edge, current - start) >= -1e-8
            if old_inside != new_inside:
                denominator = cross(edge, current - previous)
                if abs(denominator) > 1e-12:
                    t = cross(edge, start - previous) / denominator
                    output.append(previous + t * (current - previous))
            if new_inside:
                output.append(current)
            previous = current
    area = abs(signed_area(np.array(output))) if len(output) >= 3 else 0
    return area / max(1e-6, min(abs(signed_area(a)), abs(signed_area(b))))


def detect_rectangles(rgb, intrinsics, rotation, origin, table, dimensions):
    """Find complete rim rectangles matching the reported dimensions on each height plane."""
    image_lines = rim_lines(rgb)
    if len(image_lines) < 4:
        return []
    candidates = []
    for kind, spec in dimensions.items():
        height = spec["height"]
        world = pixels_on_rim(image_lines.reshape(-1, 2), intrinsics, rotation, origin, table, height)
        segments = world[:, :2].reshape(-1, 2, 2)
        delta = segments[:, 1] - segments[:, 0]
        angles = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), np.pi / 2)
        lengths = np.linalg.norm(delta, axis=1)
        hist, edges = np.histogram(angles, bins=90, range=(0, np.pi / 2), weights=np.minimum(lengths, 150))
        peaks = np.argsort(hist)[-3:]
        for peak in peaks:
            angle = (edges[peak] + edges[peak + 1]) / 2
            axes = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            local = segments @ axes
            xs, ys = _groups(local, True), _groups(local, False)
            for (left, ls), (right, rs) in itertools.combinations(xs, 2):
                width = right - left
                for expected_w, expected_h in (
                    (spec["width"], spec["length"]),
                    (spec["length"], spec["width"]),
                ):
                    tolerance = 6.0
                    if abs(width - expected_w) > tolerance:
                        continue
                    for (top, ts), (bottom, bs) in itertools.combinations(ys, 2):
                        depth = bottom - top
                        if abs(depth - expected_h) > tolerance:
                            continue
                        coverage = [
                            _coverage(ls, top, bottom),
                            _coverage(rs, top, bottom),
                            _coverage(ts, left, right),
                            _coverage(bs, left, right),
                        ]
                        if min(coverage) < 0.45 or np.mean(coverage) < 0.65:
                            continue
                        local_quad = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
                        xy = local_quad @ axes.T
                        z = np.array([table.z_at(x, y) + height for x, y in xy])
                        pixel_quad = world_to_pixels(np.column_stack((xy, z)), intrinsics, rotation, origin)
                        h, w = rgb.shape[:2]
                        if (
                            np.any(pixel_quad < 4)
                            or np.any(pixel_quad[:, 0] >= w - 4)
                            or np.any(pixel_quad[:, 1] >= h - 4)
                        ):
                            continue
                        mask = np.zeros((h, w), np.uint8)
                        cv2.fillConvexPoly(mask, np.rint(pixel_quad).astype(np.int32), 1)
                        saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
                        colored_fraction = float(np.mean(saturation[mask > 0] > 130))
                        error = (abs(width - expected_w) + abs(depth - expected_h)) / (2 * tolerance)
                        score = float(np.mean(coverage) - 0.2 * error)
                        candidates.append(
                            {
                                "kind": kind,
                                "quad_world_mm": np.column_stack((xy, z)).tolist(),
                                "quad_px": pixel_quad.tolist(),
                                "center_xy_mm": xy.mean(axis=0).tolist(),
                                "size_mm": [width, depth],
                                "yaw_deg": math.degrees(angle),
                                "score": score,
                                "edge_coverage": coverage,
                                "colored_fraction": colored_fraction,
                            }
                        )
    retained = []
    for candidate in sorted(candidates, key=lambda c: -c["score"]):
        quad = np.array(candidate["quad_world_mm"])[:, :2]
        if any(
            c["kind"] == candidate["kind"] and _overlap(quad, np.array(c["quad_world_mm"])[:, :2]) > 0.85
            for c in retained
        ):
            continue
        retained.append(candidate)
    return retained


def _container_group(candidates, kind, count, width):
    candidates = [c for c in candidates if c["kind"] == kind]
    groups, seen = [], set()
    for anchor in candidates:
        quad = np.array(anchor["quad_world_mm"])[:, :2]
        delta = np.roll(quad, -1, axis=0) - quad
        long = delta[np.argmax(np.linalg.norm(delta, axis=1))]
        long /= np.linalg.norm(long)
        short = np.array([-long[1], long[0]])
        center = np.array(anchor["center_xy_mm"])
        compatible = []
        for candidate in candidates:
            q = np.array(candidate["quad_world_mm"])[:, :2]
            edges = np.roll(q, -1, axis=0) - q
            axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
            axis /= np.linalg.norm(axis)
            displacement = np.array(candidate["center_xy_mm"]) - center
            if abs(axis @ long) < math.cos(math.radians(5)) or abs(displacement @ long) > 20:
                continue
            compatible.append((float(displacement @ short), candidate))
        compatible.sort(key=lambda item: item[0])

        def extend(chosen, start, compatible=compatible, long=long, short=short):
            if len(chosen) == count:
                group = [candidate for _, candidate in chosen]
                key = tuple(sorted(id(c) for c in group))
                if key not in seen:
                    seen.add(key)
                    groups.append((group, long, short))
                return
            for i in range(start, len(compatible)):
                position, candidate = compatible[i]
                if chosen:
                    gap = position - chosen[-1][0]
                    if gap < width - 6:
                        continue
                    if gap > width + 20:
                        break
                quad = np.array(candidate["quad_world_mm"])[:, :2]
                if any(_overlap(quad, np.array(other["quad_world_mm"])[:, :2]) > 1e-8 for _, other in chosen):
                    continue
                extend([*chosen, (position, candidate)], i + 1)

        extend([], 0)
    return groups


def select_layout(candidates, dimensions):
    """Match the photo's two-source row beside a four-bin row; reject isolated cables/edges."""
    sources = _container_group(candidates, "source", 2, dimensions["source"]["width"])
    outputs = _container_group(candidates, "destination", 4, dimensions["destination"]["width"])
    choices = []
    for (sg, long, short), (bg, bin_long, _) in itertools.product(sources, outputs):
        if abs(long @ bin_long) < math.cos(math.radians(5)):
            continue
        source_center = np.mean([c["center_xy_mm"] for c in sg], axis=0)
        bin_center = np.mean([c["center_xy_mm"] for c in bg], axis=0)
        offset = bin_center - source_center
        nominal_length = (dimensions["source"]["length"] + dimensions["destination"]["length"]) / 2
        if not nominal_length - 12 < abs(offset @ long) < nominal_length + 100:
            continue
        if abs(offset @ short) > dimensions["source"]["width"] / 2:
            continue
        if any(
            _overlap(np.array(s["quad_world_mm"])[:, :2], np.array(b["quad_world_mm"])[:, :2]) > 1e-8
            for s in sg
            for b in bg
        ):
            continue
        score = np.mean([c["score"] for c in sg + bg]) + 0.1 * np.mean([c["colored_fraction"] for c in sg])
        choices.append((score, sg + bg))
    if not choices:
        return [], "Could not match the complete two-source/four-bin arrangement"
    choices.sort(key=lambda choice: -choice[0])
    return choices[0][1], None


def layout_proposal(
    rgb, intrinsics, rotation, origin, points, dimensions, camera_pose, scene_consistent=False
):
    for name in ("source", "destination"):
        if name not in dimensions:
            raise SetupError(f"Missing {name} dimensions")
        if any(
            not isinstance(dimensions[name].get(k), (int, float)) or dimensions[name][k] <= 0
            for k in ("width", "length", "height")
        ):
            raise SetupError(f"Missing positive {name} width/length/height in millimeters")
        if dimensions[name].get("height_reference") != "tabletop":
            raise SetupError("Container heights must be measured from tabletop to top rim")
    table = fit_table_plane(points)
    candidates = detect_rectangles(rgb, intrinsics, rotation, origin, table, dimensions)
    detected, reason = select_layout(candidates, dimensions)
    counts = {kind: sum(c["kind"] == kind for c in detected) for kind in dimensions}
    complete = counts.get("source") == 2 and counts.get("destination") == 4 and scene_consistent
    containers = []
    for kind, ids in (("source", SOURCE_IDS), ("destination", BIN_IDS)):
        selected = [c for c in detected if c["kind"] == kind]
        # Image order only names a new, unassigned layout. It must never overwrite owned-bin identities.
        if selected:
            centers = np.array([np.mean(c["quad_px"], axis=0) for c in selected])
            sort_axis = int(np.argmax(np.ptp(centers, axis=0)))
            selected.sort(key=lambda c: np.mean(c["quad_px"], axis=0)[sort_axis])
        for i, c in enumerate(selected):
            containers.append({"id": ids[i] if i < len(ids) else f"extra-{kind}-{i}", **c})
    return {
        "schema_version": 1,
        "complete": complete,
        "motion_enabled": False,
        "scene_consistent": scene_consistent,
        "counts": counts,
        "reason": reason,
        "raw_candidate_count": len(candidates),
        "containers": containers,
        "camera_tool_pose": camera_pose,
        "table_plane": {
            "normal": table.normal.tolist(),
            "offset": table.offset,
            "rms_mm": table.rms_mm,
            "support": table.support,
        },
        "dimensions": dimensions,
        "review_required": [
            "Check all six rim outlines and IDs on the overlay",
            "Confirm inside/outside width/length and bottom/wall thickness",
            "Verify camera-to-world alignment before activating motion",
        ],
        "note": "Read-only layout proposal. No operating coordinates or bin-color assignments were changed.",
    }


def draw_layout(rgb, proposal):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for c in proposal["containers"]:
        quad = np.rint(c["quad_px"]).astype(np.int32)
        color = (0, 180, 255) if c["kind"] == "source" else (0, 255, 0)
        cv2.polylines(image, [quad], True, color, 2)
        center = tuple(np.mean(quad, axis=0).astype(int))
        cv2.putText(image, c["id"], center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def settings_from_proposal(settings, proposal):
    """Compute container coordinates and operating poses; preserve the physical-review gate."""
    if (
        not proposal.get("complete")
        or not proposal.get("scene_consistent")
        or proposal.get("counts") != {"source": 2, "destination": 4}
    ):
        raise SetupError("Only a complete six-container layout can be saved")
    if {c.get("id") for c in proposal["containers"]} != {*SOURCE_IDS, *BIN_IDS}:
        raise SetupError("Layout IDs are incomplete or duplicated")
    d = copy.deepcopy(settings.data)
    dimensions = d.get("layout_detection", {})
    if dimensions != proposal["dimensions"]:
        raise SetupError("Measurements changed after detection; capture a fresh layout proposal")
    for kind in ("source", "destination"):
        spec = dimensions[kind]
        if spec.get("width_length_reference") not in ("inside", "outside"):
            raise SetupError(f"Confirm whether {kind} width/length are inside or outside measurements")
        for field in ("wall_thickness", "bottom_thickness"):
            value = spec.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise SetupError(f"Measure {kind} {field} in millimeters before saving operating geometry")
    plane_data = proposal["table_plane"]
    table = TablePlane(
        np.array(plane_data["normal"]), plane_data["offset"], plane_data["rms_mm"], plane_data["support"]
    )
    rims = []
    for c in proposal["containers"]:
        container_id, kind = c["id"], c["kind"]
        if kind != ("source" if container_id in SOURCE_IDS else "destination"):
            raise SetupError("Container role does not match its ID")
        spec = dimensions[kind]
        width, length = c["size_mm"]
        wall = spec["wall_thickness"]
        if spec["width_length_reference"] == "outside":
            width, length = width - 2 * wall, length - 2 * wall
        if min(width, length) <= 0 or spec["bottom_thickness"] >= spec["height"]:
            raise SetupError("Container dimensions do not leave a valid interior")
        x, y = c["center_xy_mm"]
        corners = np.asarray(c["quad_world_mm"], dtype=float)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise SetupError("Invalid detected container corners")
        support_z = max(table.z_at(px, py) for px, py in corners[:, :2])
        floor_z = support_z + spec["bottom_thickness"]
        rim_z = support_z + spec["height"]
        box = {
            "min_mm": [x - width / 2, y - length / 2, floor_z],
            "max_mm": [x + width / 2, y + length / 2, rim_z],
            "wall_mm": wall,
            "yaw_deg": c["yaw_deg"],
        }
        rims.append(rim_z)
        if kind == "source":
            d["sources"][container_id]["container"] = box
            d["sources"][container_id]["observe_pose"] = copy.deepcopy(proposal["camera_tool_pose"])
        else:
            d["bins"][container_id]["container"] = box
            drop_z = rim_z + d["grasp"]["payload_below_tcp_mm"] + d["motion_limits"]["clearance_mm"]
            d["bins"][container_id]["drop_pose"] = {
                "x": x,
                "y": y,
                "z": drop_z,
                "o_x": 0,
                "o_y": 0,
                "o_z": -1,
                "theta": 0,
            }
    d["travel_z_mm"] = (
        max(rims) + d["grasp"]["payload_below_tcp_mm"] + d["motion_limits"]["clearance_mm"] + 10
    )
    if proposal["camera_tool_pose"]["z"] < d["travel_z_mm"]:
        raise SetupError(
            "Current observation pose is below required travel height; move the camera higher and recapture"
        )
    d["calibration_reviewed"], d["camera_frame_verified"] = False, False
    result = Settings(d, path=settings.path)
    result.validate_workcell(require_review=False)
    return result
