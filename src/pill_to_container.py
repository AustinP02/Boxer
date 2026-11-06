"""Pick up a pill box (cylinder with a WHITE cap) and put it into the big container.

--stage scan (default): look once, save outputs/pill_test/pill_overlay.jpg, print what was found.
                        The arm is NOT moved (it observes from wherever it is, so park it at the scan pose
                        or just look at the overlay). Check the overlay before running --stage go.
--stage go:             move to the scan pose, look, pick the pill, carry it over the big container,
                        re-look at the container, release, lift.

The big container is found, in this order, from --container-xy (world mm at table level, no vision),
--container-px (a pixel in the overlay image), or the YOLOE container detector (largest outline that is
wider than a colour bin). The pill is the one raised blob (white cap, plus the amber body touching it)
that passes size/depth checks; white-on-white is separated by height above the table, not by colour.
This stage does not use the 2-source / 4-destination layout check of the colour sorter.
"""
import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

import color_sorter as sorter
import green_to_container as g
from sample_scan import inspect_setup, motion_world_state
from sort_vision import decode_depth

FOLDER = sorter.OUTPUT_DIR / 'pill_test'

# ---- pill detection / sanity limits (white cap + amber body, both must stand above the table) ----
WHITE_S_MAX = 60  # HSV saturation at most this (0-255) ...
WHITE_V_MIN = 110  # ... and value at least this is "white" (the real frame is dim: table V~140, cap ~125-220); height does the real work
MIN_RAISED_MM = 12  # white pixels must stand this far above the table (the white table / paper do not)
MIN_AREA_PX = 400
MIN_SHORT_MM = 25  # the bottles in the real frame are 35-42 mm across; glare specks on the bins are 13-20 mm
MAX_HEIGHT_MM = 220
MIN_FILL = 0.55  # blob area / bounding rectangle area (a disc is 0.79, a thin white rim is far lower)
MIN_DEPTH_COVERAGE = 0.5
LYING_ASPECT = 1.35  # blob (cap + body) long / short edge at or above this: the bottle lies down
MIN_STANDING_WHITE = 0.45  # standing: the white cap is most of the blob
MIN_LYING_AMBER = 0.15  # lying: some amber body must show next to the cap
MIN_CAP_PX = 150  # white raised pixels a blob needs to count as the pill (blocks have none)
AMBER_H = (6, 26)  # the bottle body, OpenCV hue 10-20 in the real frame; the red blocks sit at hue 2-3, the yellow ones at 21 (see amber_mask)
AMBER_S_MIN = 40  # the translucent body is weakly saturated (~56 in the real frame)
AMBER_V_MIN = 70

# ---- motion ----
CLAW_REACH_MM = 53  # claw tips hang this far below the tool point (see green_to_container.MAX_RAISE_MM)
APPROACH_CLEAR_MM = 40  # tool-point clearance of the claw tips above the pill's top on approach
GRASP_Z_MM = g.MIN_GRASP_Z_MM - g.PICK_LOWER_MM  # lowest grasp: claws wrap the body's bottom, not only the cap
DROP_ABOVE_RIM_MM = 70  # tool point above the container rim at release (a held cylinder hangs below it)
DEFAULT_CONTAINER_HEIGHT_MM = 70  # rim height above the table; set --container-height to the real value
CONTAINER_MIN_SIDE_MM = 80  # YOLOE outline must be wider than this on its short side (colour bins are ~52)
MAX_ATTEMPTS = 3
JAW_WHITE_FRACTION = 0.06


# =====================================================================================================
# Pure helpers (no robot): projection, white detection, orientation. Covered by tests/test_pill_to_container.py
# =====================================================================================================

def world_to_pixel(xyz, intrinsic, origin, basis):
    """Inverse of sorter.ray_plane: pixel (u, v) of a world point."""
    p = basis.T @ (np.asarray(xyz, float) - origin)
    if p[2] <= 0:
        raise ValueError('Point is behind the camera')
    fx, fy, cx, cy = intrinsic
    return float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy)


def height_map(depth, intrinsic, origin, basis, table_z):
    """Height above the table (mm) of every pixel; NaN where depth is missing."""
    h, w = depth.shape
    fx, fy, cx, cy = intrinsic
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.where(np.isfinite(depth) & (depth > 100) & (depth < 3000), depth, np.nan).astype(np.float32)
    z_world = origin[2] + (basis[2, 0] * (u - cx) * d / fx + basis[2, 1] * (v - cy) * d / fy + basis[2, 2] * d)
    return z_world - table_z


def estimate_table_offset(heights, search_mm=70.0, bin_mm=2.0):
    """How far (mm) the real table surface is above the nominal table height used for `heights`.

    The Viam table model is not exactly the physical surface (in practice ~25 mm lower), and the table
    fills most of the picture, so the most common height is the table. 0 if there is too little depth."""
    values = heights[np.isfinite(heights)]
    values = values[(values > -search_mm) & (values < search_mm)]
    if len(values) < 5000:
        return 0.0
    hist, edges = np.histogram(values, bins=np.arange(-search_mm, search_mm + bin_mm, bin_mm))
    peak = edges[int(np.argmax(hist))] + bin_mm / 2
    near = values[np.abs(values - peak) <= 3.0]
    return float(np.median(near)) if len(near) else float(peak)


def measure_table(depth, intrinsic, origin, basis, nominal_z):
    """(table_z, heights above the MEASURED table, offset): corrects the model's table height from depth."""
    heights = height_map(depth, intrinsic, origin, basis, nominal_z)
    offset = estimate_table_offset(heights)
    return nominal_z + offset, heights - offset, offset


def white_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (179, WHITE_S_MAX, 255))


def amber_mask(frame_bgr):
    """The bottle's orange/amber body. Overlaps the red and yellow blocks' hues, so it is only ever used
    together with a white cap touching it (find_pill drops every blob without one)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (AMBER_H[0], AMBER_S_MIN, AMBER_V_MIN), (AMBER_H[1], 255, 255))


def find_pill(frame, heights, intrinsic, origin, basis, table_z, max_width_mm=60.0, max_long_mm=110.0,
              exclude=()):
    """All pill-like blobs as candidate dicts; a candidate with an empty 'rejected' list is eligible.

    A blob is the raised white cap plus any raised amber body touching it (a white cap on a white table
    is told apart by height, not colour). Blobs with no white cap (coloured blocks) are ignored.
    exclude: pixel polygons (container outlines) blanked out before the search.
    """
    white = white_mask(frame)
    raised = heights >= MIN_RAISED_MM  # NaN compares False: no depth, no pill
    white = np.where(raised, white, 0).astype(np.uint8)
    amber = np.where(raised, amber_mask(frame), 0).astype(np.uint8)
    mask = white | amber
    for poly in exclude:
        for layer in (mask, white, amber):
            cv2.fillPoly(layer, [np.asarray(poly, np.int32)], 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = mask.shape
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA_PX:
            continue
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        in_blob = blob > 0
        white_px = int(np.count_nonzero(in_blob & (white > 0)))
        if white_px < MIN_CAP_PX:
            continue  # no white cap: a coloured block, not the pill
        white_frac = white_px / max(int(np.count_nonzero(in_blob)), 1)
        amber_frac = int(np.count_nonzero(in_blob & (amber > 0))) / max(int(np.count_nonzero(in_blob)), 1)
        rejected = []
        (rcx, rcy), (rw, rh), _ = rect = cv2.minAreaRect(contour)
        values = heights[blob > 0]
        covered = values[np.isfinite(values)]
        if len(covered) < MIN_DEPTH_COVERAGE * len(values) or len(covered) < 25:
            rejected.append('missing depth on the pill')
            top_h = float('nan')
        else:
            top_h = float(np.percentile(covered, 75))  # near the top, not the table-level fringe
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 2 or y <= 2 or x + w >= fw - 2 or y + h >= fh - 2:
            rejected.append('touches the image edge (wall/table?)')
        if area / max(rw * rh, 1) < MIN_FILL:
            rejected.append('not a compact shape (thin rim/wall?)')
        cand = dict(center_px=[float(rcx), float(rcy)], area_px=float(area), contour=contour.reshape(-1, 2).tolist(),
                    height_mm=top_h, rejected=rejected)
        if math.isfinite(top_h):
            top_z = table_z + top_h
            corners = cv2.boxPoints(rect)
            world = np.array([sorter.ray_plane(p, top_z, intrinsic, origin, basis)[:2] for p in corners])
            centre = np.array(sorter.ray_plane((rcx, rcy), top_z, intrinsic, origin, basis)[:2])
            shape = g.block_shape((world - centre).tolist())
            l1, l2 = float(np.linalg.norm(world[1] - world[0])), float(np.linalg.norm(world[2] - world[1]))
            short, long_ = sorted((l1, l2))
            px_edges = (corners[1] - corners[0], corners[2] - corners[1])
            long_px = px_edges[0] if np.linalg.norm(px_edges[0]) >= np.linalg.norm(px_edges[1]) else px_edges[1]
            aspect = long_ / max(short, 1e-6)
            lying = aspect >= LYING_ASPECT
            if short < MIN_SHORT_MM:
                rejected.append(f'too small ({short:.0f} mm)')
            if long_ > max_long_mm:
                rejected.append(f'too big ({long_:.0f} mm: table/wall/box?)')
            if short > max_width_mm:
                rejected.append(f'{short:.0f} mm across: wider than the jaws open')
            if top_h > MAX_HEIGHT_MM:
                rejected.append(f'{top_h:.0f} mm tall: too high')
            if not lying and white_frac < MIN_STANDING_WHITE:
                rejected.append(f'only {white_frac:.0%} white: not a white cap (a block with glare?)')
            if lying and amber_frac < MIN_LYING_AMBER:
                rejected.append(f'no amber body ({amber_frac:.0%}): glare on a bin or the table?')
            cand.update(top_z=top_z, world_xy=centre.tolist(), size_mm=[short, long_], aspect=aspect,
                        shape='lying' if lying else 'standing', width_mm=short,
                        long_yaw=shape[0] if shape else None,
                        long_dir_px=(long_px / max(np.linalg.norm(long_px), 1e-6)).tolist())
        found.append(cand)
    return found


def eligible(cands):
    return [c for c in cands if not c['rejected']]


def grasp_point(cand):
    """(x, y, note): the blob centre. Standing: the cylinder's axis. Lying: the middle of the bottle."""
    x, y = cand['world_xy']
    return x, y, 'blob centre' if cand['shape'] == 'standing' else 'middle of the lying bottle'


def grasp_orientation(orient, jaw_offset, cand, swing=0.0):
    """Wrist orientation. Standing: any angle (a cylinder has no orientation), plus the retry swing.
    Lying: the jaws close ACROSS the bottle (perpendicular to its long axis), plus the retry swing."""
    if cand['shape'] == 'lying' and cand.get('long_yaw') is not None:
        turned = g.turn_to(orient, jaw_offset, (cand['long_yaw'] + 90) % 180, 180)
    else:
        turned = dict(orient)
    return dict(turned, theta=turned['theta'] + swing)


def retry_swing(cand, attempt):
    """Wrist change for a retry. Standing: +/-90 deg. Lying: a quarter turn would span the cylinder's
    length, so only a half turn (same grip, other wrist configuration)."""
    if attempt <= 1:
        return 0
    if cand['shape'] == 'lying':
        return 180 if attempt % 2 == 0 else -180
    return 90 if attempt % 2 == 0 else -90


def container_from_polygon(polygon, rim_z, intrinsic, origin, basis):
    """(centre xy, (short, long) extent in mm) of an outline in pixels, projected at the RIM height."""
    corners = np.array([sorter.ray_plane(p, rim_z, intrinsic, origin, basis)[:2] for p in polygon], float)
    centre = corners.mean(axis=0)
    (_, _), (w, h), _ = cv2.minAreaRect(corners.astype(np.float32))
    return centre, tuple(sorted((float(w), float(h))))


def container_from_pixel(u, v, rim_z, intrinsic, origin, basis):
    return np.array(sorter.ray_plane((u, v), rim_z, intrinsic, origin, basis)[:2], float)


def pick_big_container(detections, rim_z, intrinsic, origin, basis, min_side_mm=CONTAINER_MIN_SIDE_MM):
    """The detection with the largest footprint whose short side is at least min_side_mm, else None.
    Returns dict(centre, extent_mm, polygon, confidence) and the list of all (extent) for printing."""
    best, seen = None, []
    for d in detections:
        poly = d.get('polygon')
        if not poly:
            x, y, w, h = d['roi']
            poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        try:
            centre, extent = container_from_polygon(poly, rim_z, intrinsic, origin, basis)
        except ValueError:
            continue
        seen.append(extent)
        if extent[0] >= min_side_mm and (best is None or extent[0] * extent[1] > best['extent_mm'][0] * best['extent_mm'][1]):
            best = dict(centre=centre, extent_mm=extent, polygon=poly, confidence=d.get('confidence'))
    return best, seen


def white_in_jaw_view(frame, depth=None):
    """Fraction of the jaw-check window (bottom centre of the wrist image) that is white and near the camera."""
    h, w = frame.shape[:2]
    x0, x1, y0 = int(w * g.JAW_VIEW[0]), int(w * g.JAW_VIEW[1]), int(h * g.JAW_VIEW[2])
    mask = white_mask(frame)[y0:h, x0:x1] > 0
    if depth is not None and depth.shape[:2] == (h, w):
        d = depth[y0:h, x0:x1]
        far = np.isfinite(d) & (d > g.TABLE_MIN_DEPTH_MM) & (d < 3000)
        mask = mask & ~far  # white table far below is not in the jaws
    return float(np.mean(mask))


# =====================================================================================================
# Overlay
# =====================================================================================================

def draw_overlay(frame, cands, chosen, container, grasp_px, status):
    img = frame.copy()
    for c in cands:
        ok = not c['rejected']
        colour = (0, 255, 0) if c is chosen else ((0, 200, 255) if ok else (0, 0, 255))
        cv2.polylines(img, [np.array(c['contour'], np.int32)], True, colour, 2)
        label = ('PILL ' if c is chosen else '') + (f"{c['size_mm'][0]:.0f}x{c['size_mm'][1]:.0f}mm {c['shape']}" if 'size_mm' in c else '?')
        cv2.putText(img, label, (int(c['center_px'][0]) - 40, int(c['center_px'][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if c['rejected']:
            cv2.putText(img, c['rejected'][0][:48], (int(c['center_px'][0]) - 40, int(c['center_px'][1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    if chosen is not None and grasp_px is not None:
        u, v = int(grasp_px[0]), int(grasp_px[1])
        cv2.drawMarker(img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        if chosen['shape'] == 'lying':  # jaw axis
            long_dir = np.array(chosen['long_dir_px'])
            d = np.array([-long_dir[1], long_dir[0]]) * 40  # the jaws close across the bottle
            cv2.line(img, (int(u - d[0]), int(v - d[1])), (int(u + d[0]), int(v + d[1])), (0, 255, 0), 2)
    if container is not None:
        if container.get('polygon'):
            cv2.polylines(img, [np.array(container['polygon'], np.int32)], True, (255, 160, 0), 2)
        if container.get('centre_px'):
            u, v = map(int, container['centre_px'])
            cv2.drawMarker(img, (u, v), (255, 160, 0), cv2.MARKER_TILTED_CROSS, 30, 3)
            cv2.putText(img, 'CONTAINER', (u + 14, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(img, status[:120], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# =====================================================================================================
# Robot side
# =====================================================================================================

HALF_OPEN_POS = g.HALF_OPEN_POS
HALF_OPEN_TOLERANCE = g.HALF_OPEN_TOLERANCE
open_gripper = g.open_gripper  # shared with the colour sorter


async def grab_view(machine, camera):
    """(colour frame, depth or None, intrinsic, origin, basis) of the camera as it is now."""
    origin, basis = await sorter.camera_basis(machine)
    images, _ = await camera.get_images(timeout=20)
    frame = cv2.imdecode(np.frombuffer(next(i for i in images if i.name == 'color').data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Cannot decode the colour frame')
    p = (await camera.get_properties(timeout=15)).intrinsic_parameters
    intrinsic = (p.focal_x_px, p.focal_y_px, p.center_x_px, p.center_y_px)
    depth = None
    raw = next((i for i in images if i.name == 'depth'), None)
    if raw is not None:
        depth = decode_depth(raw.data)
        if depth.shape != frame.shape[:2]:
            raise ValueError('Depth and colour sizes differ; cannot use depth')
    return frame, depth, intrinsic, origin, basis


def locate_container(args, frame, detector, config, table_z, intrinsic, origin, basis):
    """dict(xy, polygon, centre_px, how) for the big container. Raises with advice when it cannot be found."""
    rim_z = table_z + args.container_height
    if args.container_xy:
        xy = np.array(args.container_xy, float)
        how = '--container-xy'
        polygon = None
    elif args.container_px:
        xy = container_from_pixel(*args.container_px, rim_z, intrinsic, origin, basis)
        how = '--container-px'
        polygon = None
    elif detector is not None:
        raw = detector.detect(frame, False)
        best, seen = pick_big_container(raw, rim_z, intrinsic, origin, basis, args.container_min_side)
        print('Container detector outlines (short x long mm at the rim): '
              + (', '.join(f'{a:.0f}x{b:.0f}' for a, b in seen) or 'none'), flush=True)
        if best is None:
            raise ValueError(f'Big container not detected (needs a short side >= {args.container_min_side:.0f} mm). '
                             'Give --container-xy X Y (world mm) or --container-px U V (pixel in outputs/pill_test/pill_overlay.jpg)')
        xy, polygon, how = best['centre'], best['polygon'], f"detector (conf {best['confidence'] or 0:.2f}, {best['extent_mm'][0]:.0f}x{best['extent_mm'][1]:.0f} mm)"
    else:
        raise ValueError('No container source: give --container-xy / --container-px, or allow detection')
    u, v = world_to_pixel([xy[0], xy[1], rim_z], intrinsic, origin, basis)
    return dict(xy=[float(xy[0]), float(xy[1])], polygon=polygon, centre_px=[u, v], how=how, rim_z=rim_z)


def exclusion_polygons(container, detector_polygons):
    polys = [p for p in detector_polygons if p]
    if container.get('polygon'):
        polys.append(container['polygon'])
    elif container.get('centre_px'):  # no outline: blank a generous disc around the given container centre
        u, v = container['centre_px']
        r = 190
        polys.append([[u + r * math.cos(a), v + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
    return polys


async def scan(args, machine, camera, detector, config, table_z):
    """One look: container + pill, overlay saved. Returns dict(pill, container, heights, view...)."""
    frame, depth, intrinsic, origin, basis = await grab_view(machine, camera)
    forward = basis[:, 2]
    if forward[2] > -0.8:
        print('WARNING: the camera is not looking down at the table; detection will be poor. '
              'Move the arm to the scan pose first.', flush=True)
    heights = None
    if depth is not None:
        table_z, heights, offset = measure_table(depth, intrinsic, origin, basis, table_z)
        print(f'Table surface measured from depth: z = {table_z:.0f} mm (the Viam table model is {offset:+.0f} mm off)', flush=True)
    container = locate_container(args, frame, detector, config, table_z, intrinsic, origin, basis)
    print(f"Container: world xy ({container['xy'][0]:.0f}, {container['xy'][1]:.0f}) at rim z {container['rim_z']:.0f} mm, from {container['how']}", flush=True)
    detector_polys = []
    if detector is not None:
        try:
            detector_polys = [d['polygon'] for d in detector.detect(frame, False) if d.get('polygon')]
        except Exception:
            detector_polys = []
    chosen, cands = None, []
    if args.pill_xy:
        top = table_z + args.pill_height
        chosen = dict(center_px=list(world_to_pixel([*args.pill_xy, top], intrinsic, origin, basis)), area_px=0.0, contour=[],
                      height_mm=args.pill_height, rejected=[], top_z=top, world_xy=list(args.pill_xy), size_mm=[0, 0], aspect=1.0,
                      shape='standing', width_mm=0, long_yaw=None, long_dir_px=[1.0, 0.0])
        cands = [chosen]
        print(f'Pill: using --pill-xy {args.pill_xy} with height {args.pill_height:.0f} mm (no vision)', flush=True)
    else:
        if depth is None:
            raise ValueError('No depth image: cannot separate the pill from the white table; give --pill-xy')
        cands = find_pill(frame, heights, intrinsic, origin, basis, table_z, max_width_mm=args.max_pill_mm,
                          exclude=exclusion_polygons(container, detector_polys))
        good = eligible(cands)
        for c in cands:
            print('  white blob at px ({:.0f},{:.0f}) {}: {}'.format(*c['center_px'], f"{c['size_mm'][0]:.0f}x{c['size_mm'][1]:.0f} mm, {c['height_mm']:.0f} mm tall, {c['shape']}" if 'size_mm' in c else '',
                  'OK' if not c['rejected'] else '; '.join(c['rejected'])), flush=True)
        if good:
            chosen = min(good, key=lambda c: math.hypot(c['world_xy'][0] - container['xy'][0], c['world_xy'][1] - container['xy'][1]))
            if len(good) > 1:
                print(f'{len(good)} pills seen; taking the one nearest the container (run again for the rest).', flush=True)
    grasp = None
    if chosen is not None:
        gx, gy, note = grasp_point(chosen)
        chosen['grasp_xy'], chosen['grasp_note'] = [gx, gy], note
        grasp = world_to_pixel([gx, gy, chosen['top_z']], intrinsic, origin, basis)
        print(f"Pill: {chosen['shape']}, {chosen['size_mm'][0]:.0f}x{chosen['size_mm'][1]:.0f} mm, top {chosen['height_mm']:.0f} mm above the table; "
              f"grasp at ({gx:.0f}, {gy:.0f}) [{note}]", flush=True)
    status = (f"pill {'FOUND ' + chosen['shape'] if chosen else 'NOT found'} | container via {container['how'][:40]}")
    FOLDER.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FOLDER / 'pill_frame.jpg'), frame)
    overlay = draw_overlay(frame, cands, chosen, container, grasp, status)
    cv2.imwrite(str(FOLDER / 'pill_overlay.jpg'), overlay)
    (FOLDER / 'pill_scan.json').write_text(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), container=container,
        pill={k: v for k, v in (chosen or {}).items() if k != 'contour'}, candidates=len(cands)), indent=2))
    print('Saved', FOLDER / 'pill_overlay.jpg', flush=True)
    return dict(pill=chosen, container=container, intrinsic=intrinsic, origin=origin, basis=basis)


async def run(args, config):
    if args.stage == 'go':
        rim = max(src['rim_z_mm'] for src in config['sources'])
        sorter.check_ready({**config, 'motion_commissioned': True, 'sources': [{**s, 'rim_z_mm': rim} for s in config['sources']]})
    load_dotenv(sorter.ROOT / '.env')
    detector = None
    if not (args.container_xy or args.container_px):
        from container_detector import ContainerDetector  # load before connecting: loading starves the session heartbeat
        detector = ContainerDetector()
    machine = await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'], api_key_id=os.environ['VIAM_API_KEY_ID'])), 30)
    arm = gripper = None
    try:
        arm = Arm.from_robot(machine, 'arm'); gripper = Gripper.from_robot(machine, 'gripper')
        camera = Camera.from_robot(machine, 'cam'); motion = MotionClient.from_robot(machine, 'builtin')
        world, _ = await inspect_setup(machine, motion)
        table_pose = (await motion.get_pose('table', 'world', timeout=15)).pose
        table_geometry = next(o for o in world.obstacles if o.reference_frame == 'table').geometries[0]
        table_z = table_pose.z + table_geometry.center.z + table_geometry.box.dims_mm.z / 2
        config['_table_top_z_mm'] = table_z
        if args.stage == 'scan':
            found = await scan(args, machine, camera, detector, config, table_z)
            print('\nScan only; the arm did not move.' if found['pill'] else '\nNo single pill found; the arm did not move.')
            print('If the overlay looks right, run:  .venv/Scripts/python -u boxer.py pill --stage go ' + command_tail())
            return
        if (await gripper.is_holding_something(timeout=15)).is_holding_something:
            raise RuntimeError('Gripper is holding something; clear it first')
        mover = g.DirectFallbackMotion(motion, arm)
        state = motion_world_state(world)
        await arm.do_command({'set_speed': g.SPEED}, timeout=15)
        jaw_offset = await g.measure_jaw_offset(machine, motion)
        orient = config['pick_orientation']; transit = config['transit_z_mm']

        async def move(pose):
            if not await mover.move(component_name='gripper', destination=PoseInFrame(reference_frame='world', pose=pose),
                    world_state=state, extra={'motion_profile': 'linear'}, timeout=g.MOVE_TIMEOUT_S):
                raise RuntimeError('Motion not confirmed')

        await g.move_arm_to_scan(motion, world, config)
        await asyncio.sleep(1)
        found = await scan(args, machine, camera, detector, config, table_z)
        pill, container = found['pill'], found['container']
        if pill is None:
            raise ValueError('No eligible pill found; no motion issued (see the overlay)')
        flip = 0
        carrying = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            x, y = pill['grasp_xy']
            swing = retry_swing(pill, attempt) + flip
            pick_orient = grasp_orientation(orient, jaw_offset, pill, swing)
            grasp_z = args.grasp_z if args.grasp_z is not None else GRASP_Z_MM
            approach_z = min(transit, max(pill['top_z'] + CLAW_REACH_MM + APPROACH_CLEAR_MM, grasp_z + 80))
            pick = Pose(x=x, y=y, z=grasp_z, **pick_orient)
            approach = sorter.clone_pose(pick, z=approach_z)
            print(f"--- pill (try {attempt}): pick ({x:.0f},{y:.0f}) z {grasp_z:.0f}, wrist swing {swing:+.0f} deg, {pill['shape']}", flush=True)
            try:
                here = (await motion.get_pose('gripper', 'world', timeout=15)).pose
                await move(sorter.clone_pose(here, z=max(here.z, transit)))
                await move(sorter.clone_pose(approach, z=transit))
                await open_gripper(gripper, args.half_open)
                await move(approach)
                mover.allow_table_contact = True
                try:
                    await move(pick)
                finally:
                    mover.allow_table_contact = False
                await gripper.grab(timeout=20)
                await move(approach)
                frame, depth, *_ = await grab_view(machine, camera)
                fraction = white_in_jaw_view(frame, depth)
                holding = (await gripper.is_holding_something(timeout=15)).is_holding_something
                print(f'  jaw check: gripper reports {"HOLDING" if holding else "empty"}; white in the jaw view {fraction:.0%}', flush=True)
                if not holding:
                    raise RuntimeError('gripper reports nothing held after the grasp')
                if args.require_white and fraction < JAW_WHITE_FRACTION:
                    raise RuntimeError(f'camera does not see white in the jaws ({fraction:.0%})')
                carrying = True
                await move(sorter.clone_pose(approach, z=transit))
                drop_xy = container['xy']
                if args.container_recheck and not (args.container_xy or args.container_px):
                    try:
                        await g.move_arm_to_scan(motion, world, config)
                        fresh = await scan_container_only(args, machine, camera, detector, config, table_z)
                        moved = math.hypot(fresh['xy'][0] - drop_xy[0], fresh['xy'][1] - drop_xy[1])
                        print(f'  container recheck: moved {moved:.1f} mm', flush=True)
                        if moved > 40:
                            print('  container moved a lot; keeping the new position', flush=True)
                        drop_xy = fresh['xy']
                    except Exception as error:
                        print('  container recheck failed; using the scanned position:', str(error)[:100], flush=True)
                drop_z = container['rim_z'] + args.drop_above_rim
                drop = Pose(x=drop_xy[0], y=drop_xy[1], z=drop_z, **pick_orient)
                print(f'  dropping over ({drop_xy[0]:.0f},{drop_xy[1]:.0f}) at z {drop_z:.0f} (rim {container["rim_z"]:.0f})', flush=True)
                await g.place_block(move, gripper, drop, transit, False)
                print('Pill released into the container.', flush=True)
                break
            except Exception as error:
                reason = str(error)[:160]
                print(f'  pill try {attempt} failed: {reason}', flush=True)
                try:
                    await arm.do_command({'set_speed': g.SPEED}, timeout=15)
                except Exception:
                    pass
                if 'self-collision' in str(error):  # wrist hit its camera: use the mirrored wrist turn
                    flip = 180 if flip != 180 else -180
                if carrying:
                    print('The pill is in the jaws but was not placed; NOT releasing blindly. Use --stage go after clearing, or release by hand.', flush=True)
                    raise
                if attempt == MAX_ATTEMPTS:
                    raise
                await gripper.open(timeout=15)
                here = (await motion.get_pose('gripper', 'world', timeout=15)).pose
                await move(sorter.clone_pose(here, z=max(here.z, transit)))
                if args.pill_xy:
                    continue
                await g.move_arm_to_scan(motion, world, config)
                again = await scan(args, machine, camera, detector, config, table_z)
                if again['pill'] is None:
                    raise ValueError(f'{reason}; pill not found after the rescan')
                pill = again['pill']
        try:
            await g.move_arm_to_scan(motion, world, config)
        except Exception:
            pass
    finally:
        try:
            if arm is not None:
                try:
                    await arm.stop(timeout=10)
                finally:
                    if gripper is not None:
                        await gripper.stop(timeout=10)
        finally:
            await machine.close()


async def scan_container_only(args, machine, camera, detector, config, table_z):
    frame, depth, intrinsic, origin, basis = await grab_view(machine, camera)
    if depth is not None:
        table_z = measure_table(depth, intrinsic, origin, basis, table_z)[0]
    return locate_container(args, frame, detector, config, table_z, intrinsic, origin, basis)


def command_tail():
    """The options the user typed, without --stage, so the 'go' hint repeats the same container/pill settings."""
    out, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
        elif a == '--stage':
            skip = True
        elif not a.startswith('--stage='):
            out.append(a)
    return ' '.join(out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['scan', 'go'], default='scan', help='scan: look only, no motion (default). go: pick and drop')
    parser.add_argument('--container-xy', nargs=2, type=float, metavar=('X', 'Y'), help='big container centre, world mm (skips vision for it)')
    parser.add_argument('--container-px', nargs=2, type=float, metavar=('U', 'V'), help='big container centre as a pixel in the scan image')
    parser.add_argument('--container-height', type=float, default=DEFAULT_CONTAINER_HEIGHT_MM,
                        help=f'container rim height above the table, mm (default {DEFAULT_CONTAINER_HEIGHT_MM}; MEASURE IT)')
    parser.add_argument('--container-min-side', type=float, default=CONTAINER_MIN_SIDE_MM, help='detector: shortest side of the big container, mm')
    parser.add_argument('--pill-xy', nargs=2, type=float, metavar=('X', 'Y'), help='pill centre, world mm (skips white detection; assumes standing)')
    parser.add_argument('--pill-height', type=float, default=70.0, help='with --pill-xy: pill height above the table, mm')
    parser.add_argument('--max-pill-mm', type=float, default=60.0, help='widest pill the jaws can take, mm')
    parser.add_argument('--grasp-z', type=float, default=None, help=f'tool-point z for the grasp (default {GRASP_Z_MM:.1f}, the lowest the planner takes)')
    parser.add_argument('--drop-above-rim', type=float, default=DROP_ABOVE_RIM_MM, help='tool-point height above the container rim at release, mm')
    parser.add_argument('--no-container-recheck', dest='container_recheck', action='store_false', help='go: do not re-detect the container before the drop')
    parser.add_argument('--speed', type=float, default=None, metavar='N', help='arm speed (normal 50); 5 = super slow and quiet')
    parser.add_argument('--half-open', nargs='?', type=float, const=HALF_OPEN_POS, default=None, metavar='POS',
                        help=f'go: open the jaws only to POS (0..850, default {HALF_OPEN_POS:.0f} = half) before the grasp instead of fully')
    parser.add_argument('--require-white', action='store_true', help='go: also fail the grasp when the camera sees no white in the jaws')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    g.apply_speed(args.speed)
    asyncio.run(run(args, g.load_config()))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Stopped; no automatic release or return.')
    except Exception as error:
        raise SystemExit('Pill stage stopped: ' + str(error))
