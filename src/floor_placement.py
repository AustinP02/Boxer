"""Fit a fixed-orientation block footprint on currently observed clear floor.

No robot I/O. Invalid depth never becomes free floor. Pick and drop must use
the same tool orientation for the measured world footprint to remain valid.
"""
import cv2
import numpy as np


def footprint_offsets(item, origin, basis, intrinsics):
    if 'world_mm' not in item or not item.get('instance_outline'):
        return None
    fx, fy, cx, cy = intrinsics
    z = item['world_mm'][2]
    pixels = np.array(item['instance_outline'], dtype=float)
    rays = np.column_stack(((pixels[:, 0]-cx)/fx,
                            (pixels[:, 1]-cy)/fy, np.ones(len(pixels)))) @ basis.T
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        return None
    lengths = (z-origin[2])/rays[:, 2]
    if np.any(lengths <= 0):
        return None
    points = origin[:2]+rays[:, :2]*lengths[:, None]
    # Enclose the complete visible mask rather than joining same-color masks.
    corners = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
    return (corners-np.array(item['world_mm'][:2])).tolist()


def floor_context(container, depth, floor_z, origin, basis, intrinsics):
    """Rasterize a bin in world XY at 1 mm, validating each cell from depth."""
    if depth is None or floor_z is None:
        return None
    x, y, w, h = container['roi']
    polygon = np.array(container.get('polygon') or
                       [[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=float)
    fx, fy, cx, cy = intrinsics
    rays = np.column_stack(((polygon[:, 0]-cx)/fx,
                            (polygon[:, 1]-cy)/fy, np.ones(len(polygon)))) @ basis.T
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        return None
    lengths = (floor_z-origin[2])/rays[:, 2]
    if np.any(lengths <= 0):
        return None
    world = origin+rays*lengths[:, None]
    lower = np.floor(world[:, :2].min(axis=0))-2
    upper = np.ceil(world[:, :2].max(axis=0))+2
    size = (upper-lower+1).astype(int)
    if np.any(size <= 0) or np.any(size > 600):
        return None
    xx, yy = np.meshgrid(np.arange(size[0])+lower[0],
                         np.arange(size[1])+lower[1])
    points = np.stack((xx, yy, np.full_like(xx, floor_z)), axis=-1)
    camera = (points-origin) @ basis
    if np.any(camera[..., 2] <= 0):
        return None
    uu = np.rint(fx*camera[..., 0]/camera[..., 2]+cx).astype(int)
    vv = np.rint(fy*camera[..., 1]/camera[..., 2]+cy).astype(int)
    in_image = (uu >= 0) & (uu < depth.shape[1]) & (vv >= 0) & (vv < depth.shape[0])
    bin_mask = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(bin_mask, [polygon.astype(np.int32)], 255)
    bin_mask = cv2.erode(bin_mask, np.ones((5, 5), np.uint8))
    available = np.zeros(xx.shape, np.uint8)
    ys, xs = np.where(in_image)
    ds = depth[vv[ys, xs], uu[ys, xs]]
    # Same six-millimetre height tolerance as existing floor verification.
    error_z = np.abs((ds-camera[ys, xs, 2])*basis[2, 2])
    good = (ds > 100) & (ds < 2000) & np.isfinite(ds) & (error_z < 6)
    good &= bin_mask[vv[ys, xs], uu[ys, xs]] > 0
    available[ys[good], xs[good]] = 255
    return dict(available=available, lower=lower, floor_z=floor_z,
                origin=np.array(origin), basis=np.array(basis), intrinsics=intrinsics)


def fit_footprint(offsets, context, margin_mm=5):
    """Require every footprint cell plus 5 mm clearance to be measured free."""
    if offsets is None or context is None:
        return None
    offsets = np.array(offsets, dtype=float)
    if offsets.shape != (4, 2) or not np.all(np.isfinite(offsets)):
        return None
    margin = int(np.ceil(margin_mm))
    extent = int(np.ceil(np.max(np.abs(offsets))))+margin+2
    if not 1 < extent < 200:
        return None
    anchor = (extent, extent)
    kernel = np.zeros((2*extent+1, 2*extent+1), np.uint8)
    cv2.fillConvexPoly(kernel, np.rint(offsets+extent).astype(np.int32), 1)
    kernel = cv2.dilate(kernel, np.ones((2*margin+1, 2*margin+1), np.uint8))
    fits = cv2.erode(context['available'], kernel, anchor=anchor,
                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if not np.any(fits):
        return None
    # Prefer an interior valid position rather than the first edge pixel.
    distance = cv2.distanceTransform(fits, cv2.DIST_L2, 5)
    _, _, _, pixel = cv2.minMaxLoc(distance)
    xy = context['lower']+np.array(pixel)
    camera = context['basis'].T @ (np.array([*xy, context['floor_z']])-context['origin'])
    fx, fy, cx, cy = context['intrinsics']
    uv = [int(round(fx*camera[0]/camera[2]+cx)), int(round(fy*camera[1]/camera[2]+cy))]
    return dict(free_center=uv, free_world_xy_mm=xy.tolist(),
                placement_method='fixed-orientation footprint on measured floor',
                placement_margin_mm=margin_mm)


def choose_placement(color, bins, offsets, contexts):
    matching = [b for b in bins if b['colors'] == [color]]
    # An existing assignment keeps priority even if that bin currently has no room.
    candidates = matching or [b for b in bins if b['empty_verified'] and not b['colors']]
    for container in candidates:
        placement = fit_footprint(offsets, contexts.get(container['id']))
        if placement:
            return dict(container, **placement)
    return None
