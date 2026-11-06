from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxer.geometry import footprints_overlap, inside_container
from boxer.layout import (
    TablePlane,
    fit_table_plane,
    layout_proposal,
    pixels_on_rim,
    settings_from_proposal,
    world_to_pixels,
)
from boxer.settings import SetupError


def scene():
    intr = SimpleNamespace(
        width_px=800, height_px=800, focal_x_px=1000, focal_y_px=1000, center_x_px=400, center_y_px=400
    )
    rotation, origin = np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 600.0])
    rgb = np.full((800, 800, 3), 210, np.uint8)
    dimensions = {
        "source": {"width": 108, "length": 138, "height": 60, "height_reference": "tabletop"},
        "destination": {"width": 52, "length": 138, "height": 35, "height_reference": "tabletop"},
    }
    for kind, centers in (
        ("source", [(-60, 80), (60, 80)]),
        ("destination", [(-90, -90), (-30, -90), (30, -90), (90, -90)]),
    ):
        spec = dimensions[kind]
        for x, y in centers:
            w, h, z = spec["width"], spec["length"], spec["height"]
            quad = np.array(
                [
                    [x - w / 2, y - h / 2, z],
                    [x + w / 2, y - h / 2, z],
                    [x + w / 2, y + h / 2, z],
                    [x - w / 2, y + h / 2, z],
                ]
            )
            px = np.rint(world_to_pixels(quad, intr, rotation, origin)).astype(np.int32)
            cv2.polylines(rgb, [px], True, (90, 90, 90), 3)
            if kind == "source":
                center = np.rint(world_to_pixels(np.array([[x, y, z]]), intr, rotation, origin)[0]).astype(
                    int
                )
                cv2.rectangle(rgb, tuple(center - 35), tuple(center + 35), (255, 0, 0), -1)
    rng = np.random.default_rng(3)
    points = np.column_stack((rng.uniform(-300, 300, (5000, 2)), rng.normal(0, 0.25, 5000)))
    return rgb, intr, rotation, origin, points, dimensions


def test_known_height_projects_rim_without_using_plastic_depth():
    _, intr, rotation, origin, _, _ = scene()
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), 0, 0.2, 0.8)
    point = pixels_on_rim(np.array([[400, 400]]), intr, rotation, origin, plane, 60)
    np.testing.assert_allclose(point, [[0, 0, 60]])
    np.testing.assert_allclose(world_to_pixels(point, intr, rotation, origin), [[400, 400]])


def test_table_fit_rejects_block_tops_and_depth_outliers():
    _, _, _, _, points, _ = scene()
    rng = np.random.default_rng(4)
    blocks = np.column_stack((rng.uniform(-150, 150, (1000, 2)), np.full(1000, 60)))
    plane = fit_table_plane(np.vstack([points, blocks]))
    assert plane.z_at(0, 0) == pytest.approx(0, abs=0.2)
    assert plane.support > 0.7 and plane.rms_mm < 0.5


def test_automatic_layout_recovers_two_sources_and_four_bins():
    rgb, intr, rotation, origin, points, dimensions = scene()
    proposal = layout_proposal(
        rgb, intr, rotation, origin, points, dimensions, {"x": 0, "y": 0, "z": 500}, scene_consistent=True
    )
    assert proposal["complete"], proposal["counts"]
    assert proposal["counts"] == {"source": 2, "destination": 4}
    assert proposal["motion_enabled"] is False
    for c in proposal["containers"]:
        expected = 60 if c["kind"] == "source" else 35
        assert np.mean(np.array(c["quad_world_mm"])[:, 2]) == pytest.approx(expected, abs=0.2)


def test_missing_bins_is_reported_as_incomplete():
    rgb, intr, rotation, origin, points, dimensions = scene()
    rgb[410:] = 210
    proposal = layout_proposal(rgb, intr, rotation, origin, points, dimensions, {})
    assert not proposal["complete"]


def test_unknown_height_reference_is_not_used_for_coordinates():
    rgb, intr, rotation, origin, points, dimensions = scene()
    dimensions["source"]["height_reference"] = "inside"
    with pytest.raises(SetupError, match="tabletop"):
        layout_proposal(rgb, intr, rotation, origin, points, dimensions, {})


def test_missing_depth_fails_instead_of_inventing_a_table():
    with pytest.raises(SetupError, match="Insufficient"):
        fit_table_plane(np.zeros((0, 3)))


def test_rotated_container_roi_and_overlap_are_not_axis_aligned():
    box = {"min_mm": [0, 0, 0], "max_mm": [100, 40, 60], "wall_mm": 2, "yaw_deg": 90}
    assert inside_container(np.array([[50, 60, 30]]), box)[0]
    assert not inside_container(np.array([[90, 20, 30]]), box)[0]
    other = {"min_mm": [85, 0, 0], "max_mm": [100, 40, 60], "wall_mm": 2}
    assert not footprints_overlap(box, other)


def proposal_with_measured_walls(settings):
    rgb, intr, rotation, origin, points, dimensions = scene()
    for spec in dimensions.values():
        spec.update(width_length_reference="outside", wall_thickness=2, bottom_thickness=2)
    settings.data["layout_detection"] = dimensions
    proposal = layout_proposal(
        rgb,
        intr,
        rotation,
        origin,
        points,
        dimensions,
        {"x": 0, "y": 0, "z": 500, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},
        scene_consistent=True,
    )
    return proposal


def test_detected_layout_generates_all_operating_coordinates_without_enabling_motion(settings):
    proposal = proposal_with_measured_walls(settings)
    generated = settings_from_proposal(settings, proposal)
    assert not generated.data["calibration_reviewed"]
    assert not generated.data["camera_frame_verified"]
    for source in generated.data["sources"].values():
        assert source["container"] is not None
        assert source["observe_pose"]["z"] == 500
    for output in generated.data["bins"].values():
        assert output["container"] is not None and output["drop_pose"] is not None
    generated.validate_workcell(require_review=False)


def test_saving_layout_requires_confirmed_measurements(settings):
    proposal = proposal_with_measured_walls(settings)
    settings.data["layout_detection"]["destination"]["bottom_thickness"] = None
    with pytest.raises(SetupError, match="bottom_thickness"):
        settings_from_proposal(settings, proposal)


def test_incomplete_or_unstable_layout_never_generates_coordinates(settings):
    proposal = proposal_with_measured_walls(settings)
    proposal["scene_consistent"] = False
    with pytest.raises(SetupError, match="complete"):
        settings_from_proposal(settings, proposal)


def test_rim_duplicate_overlap_handles_shared_corners_and_containment():
    from boxer.layout import _overlap

    outer = np.array([[0, 0], [50, 0], [50, 140], [0, 140]], dtype=float)
    inner = np.array([[0, 3], [50, 3], [50, 140], [0, 140]], dtype=float)
    assert _overlap(outer, inner) == pytest.approx(1)
    assert _overlap(outer, outer + 1000) == 0


def test_layout_selection_skips_high_score_overlapping_alternative():
    from boxer.layout import select_layout

    def rectangle(kind, x, y, w, h, score):
        quad = np.array(
            [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]]
        )
        return {
            "kind": kind,
            "center_xy_mm": [x, y],
            "quad_world_mm": np.column_stack((quad, np.zeros(4))).tolist(),
            "score": score,
            "colored_fraction": 0.5 if kind == "source" else 0,
        }

    sources = [rectangle("source", x, 80, 108, 138, 0.9) for x in (-60, 60)]
    outputs = [rectangle("destination", x, -90, 52, 138, 0.9) for x in (-90, -30, 30, 90)]
    false = rectangle("destination", -20, -90, 52, 138, 0.99)
    dimensions = {"source": {"width": 108, "length": 138}, "destination": {"width": 52, "length": 138}}
    selected, reason = select_layout([*sources, *outputs, false], dimensions)
    assert reason is None and len(selected) == 6
    assert all(item is not false for item in selected)
    assert {tuple(c["center_xy_mm"]) for c in selected if c["kind"] == "destination"} == {
        (-90, -90),
        (-30, -90),
        (30, -90),
        (90, -90),
    }
