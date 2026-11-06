from types import SimpleNamespace

import numpy as np
import pytest

from boxer.pcd import decode_pcd
from boxer.perception import (
    CloudImage,
    cluster_points,
    color_mask,
    extract_candidates,
    grasp_candidate,
    project_cloud,
)


def pcd_blob(mode="binary", floating_rgb=False):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4" if floating_rgb else "<u4")])
    data = np.zeros(3, dtype)
    data["x"] = [0.01, 0.02, 0]
    data["y"] = 0.03
    data["z"] = [0.5, 0.6, np.nan]
    packed = np.array([0xFF0000, 0x00FF00, 0], dtype="<u4")
    data["rgb"] = packed.view("<f4") if floating_rgb else packed
    header = (
        "VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F "
        + ("F" if floating_rgb else "U")
        + "\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA "
        + mode
        + "\n"
    ).encode()
    if mode == "binary":
        return header + data.tobytes()
    payload = "\n".join(" ".join(str(row[k]) for k in data.dtype.names) for row in data)
    return header + payload.encode()


@pytest.mark.parametrize("mode", ["ascii", "binary"])
@pytest.mark.parametrize("floating_rgb", [False, True])
def test_pcd_units_color_packing_invalid_depth(mode, floating_rgb):
    xyz, rgb = decode_pcd(pcd_blob(mode, floating_rgb))
    np.testing.assert_allclose(xyz, [[10, 30, 500], [20, 30, 600]], rtol=1e-6)
    assert rgb.tolist() == [[255, 0, 0], [0, 255, 0]]


def test_truncated_pcd_rejected():
    with pytest.raises(ValueError, match="Truncated"):
        decode_pcd(pcd_blob()[:-8])


def test_projection_uses_nearest_point_and_preserves_pixel_to_3d():
    intr = SimpleNamespace(
        width_px=100, height_px=100, focal_x_px=100, focal_y_px=100, center_x_px=50, center_y_px=50
    )
    xyz = np.array([[0, 0, 500], [0, 0, 600], [10, 0, 500], [10000, 0, 500]])
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)
    result = project_cloud(xyz, rgb, intr)
    assert result.rgb[50, 50].tolist() == [255, 0, 0]
    assert len(result.camera_mm) == 2
    for point, pixel in zip(result.camera_mm, result.uv):
        np.testing.assert_allclose(pixel, [50 + point[0] / point[2] * 100, 50])


def test_four_color_masks_and_red_hue_wrap(settings):
    rgb = np.array(
        [[255, 0, 0], [255, 0, 15], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 255, 255], [5, 0, 0]],
        np.uint8,
    )
    for color, indexes in [("red", [0, 1]), ("green", [2]), ("blue", [3]), ("yellow", [4])]:
        mask = color_mask(rgb, settings.data["colors"][color]["hex"], settings.data["perception"])
        assert np.flatnonzero(mask).tolist() == indexes


def test_separate_same_color_items_split(candidate):
    points = np.vstack([candidate.points_mm, candidate.points_mm + [90, 0, 0]])
    assert len(cluster_points(points, 2, 5, 30)) == 2


def test_irregular_triangle_has_grasp_without_shape_label(settings):
    x, y = np.meshgrid(np.arange(125, 176, 2), np.arange(125, 176, 2))
    keep = (x - 125) + (y - 125) <= 50
    points = np.column_stack([x[keep], y[keep], np.full(keep.sum(), 50)])
    c, reason = grasp_candidate("yellow", points, settings)
    assert reason is None and c.color == "yellow"
    assert c.grasp_mm[2] == 38


def test_ring_center_is_not_a_grasp(settings):
    theta = np.linspace(0, 2 * np.pi, 200)
    points = np.column_stack([150 + 20 * np.cos(theta), 150 + 20 * np.sin(theta), np.full(200, 50)])
    c, reason = grasp_candidate("red", points, settings)
    assert c is None and "surface" in reason


def test_wide_cluster_rejected(settings, candidate):
    points = candidate.points_mm.copy()
    points[:, :2] = (points[:, :2] - 150) * 4 + 150
    c, _reason = grasp_candidate("blue", points, settings)
    assert c is None


def test_source_roi_keeps_destination_items_out(settings, candidate):
    pts = np.vstack([candidate.points_mm, candidate.points_mm + [300, 0, 0]])
    rgb = np.tile([0, 0, 255], (len(pts), 1)).astype(np.uint8)
    uv = np.column_stack([np.arange(len(pts)), np.zeros(len(pts))]).astype(int)
    cloud = CloudImage(np.zeros((1, len(pts), 3), np.uint8), pts, uv, rgb)
    boxes = {c: [] for c in settings.data["colors"]}
    boxes["blue"] = [SimpleNamespace(x_min=0, y_min=0, x_max=len(pts), y_max=1)]
    candidates, _, objects = extract_candidates(cloud, pts, boxes, settings)
    assert len(candidates) == 1 and len(objects) == 1
    assert candidates[0].center_mm[0] == pytest.approx(150)


def test_both_sources_segment_independently_and_track_source_id(settings, candidate):
    pts = np.vstack([candidate.points_mm, candidate.points_mm + [-300, 0, 0]])
    rgb = np.tile([0, 0, 255], (len(pts), 1)).astype(np.uint8)
    uv = np.column_stack([np.arange(len(pts)), np.zeros(len(pts))]).astype(int)
    cloud = CloudImage(np.zeros((1, len(pts), 3), np.uint8), pts, uv, rgb)
    detections = {c: [] for c in settings.data["colors"]}
    detections["blue"] = [SimpleNamespace(x_min=0, y_min=0, x_max=len(pts), y_max=1)]
    targets, _, objects = extract_candidates(cloud, pts, detections, settings)
    assert {t.source_id for t in targets} == {"source-1", "source-2"}
    assert len(objects) == 2
    targets, _, _ = extract_candidates(cloud, pts, detections, settings, "source-2")
    assert len(targets) == 1 and targets[0].source_id == "source-2"
