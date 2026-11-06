"""Construct Viam world geometry and sequential TCP waypoints, in millimeters."""

import numpy as np
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
    WorldState,
)

from .geometry import local_points, rotation_xy


def pose_dict(pose):
    return {k: getattr(pose, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


def make_box(label, center, dims):
    return Geometry(
        label=label,
        center=Pose(x=center[0], y=center[1], z=center[2], o_z=1),
        box=RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])),
    )


def container_geometry(label, box):
    lo, hi = np.array(box["min_mm"], float), np.array(box["max_mm"], float)
    t = box["wall_mm"]
    center, size = (lo + hi) / 2, hi - lo
    geoms = [
        make_box(
            f"{label}-floor", [center[0], center[1], lo[2] - t / 2], [size[0] + 2 * t, size[1] + 2 * t, t]
        )
    ]
    for axis in (0, 1):
        for sign, side in ((-1, "min"), (1, "max")):
            c, s = center.copy(), size.copy()
            c[axis] = (lo if sign == -1 else hi)[axis] + sign * t / 2
            s[axis] = t
            s[1 - axis] += 2 * t
            geoms.append(make_box(f"{label}-wall-{axis}-{side}", c, s))
    if box.get("yaw_deg", 0):
        for geom in geoms:
            p = np.array([geom.center.x, geom.center.y])
            rotated = (p - center[:2]) @ rotation_xy(box).T + center[:2]
            geom.center.x, geom.center.y = rotated
            geom.center.theta = box["yaw_deg"]
    return geoms


def world_state(settings, objects=(), target=None, carrying=False):
    d = settings.data
    obstacles = []
    for source_id, cfg in d["sources"].items():
        obstacles += container_geometry(f"boxer-{source_id}", cfg["container"])
    for bin_id, cfg in d["bins"].items():
        obstacles += container_geometry(f"boxer-{bin_id}", cfg["container"])
    for i, (color, points) in enumerate(objects):
        # Exclude only this exact target cluster; keep all other detected objects.
        if target is not None and points is target.points_mm:
            continue
        lo, hi = np.min(points, axis=0), np.max(points, axis=0)
        obstacles.append(make_box(f"boxer-object-{i}-{color}", (lo + hi) / 2, np.maximum(hi - lo, 4)))
    state = WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=obstacles)])
    if carrying:
        # Conservative square payload envelope accounts for unknown underside/held yaw.
        side = payload_side_mm(target, settings)
        floor = d["sources"][target.source_id]["container"]["min_mm"][2]
        below = min(d["grasp"]["payload_below_tcp_mm"], target.grasp_mm[2] - floor)
        above = d["grasp"]["insertion_mm"]
        state.transforms.append(
            Transform(
                reference_frame="boxer-payload",
                pose_in_observer_frame=PoseInFrame(reference_frame=d["gripper"], pose=Pose(o_z=1)),
                physical_object=make_box(
                    "boxer-payload", [0, 0, (below - above) / 2], [side, side, below + above]
                ),
            )
        )
    return state


def waypoints(candidate, settings, bin_id):
    d = settings.data
    # At the OV south pole, Ry(pi)Rz(theta) makes local x yaw = 180 - theta.
    # yaw_offset_deg is taught for the actual gripper's finger-closing axis.
    theta = (180 - candidate.yaw_deg + d["grasp"]["yaw_offset_deg"] + 180) % 360 - 180
    grasp = dict(zip(("x", "y", "z"), map(float, candidate.grasp_mm)))
    grasp.update(o_x=0.0, o_y=0.0, o_z=-1.0, theta=theta)
    above = {**grasp, "z": d["travel_z_mm"]}
    drop = d["bins"][bin_id]["drop_pose"].copy()
    # Carry the object straight down with the taught grasp yaw; a square payload bounds yaw changes.
    bin_above = {**drop, "z": d["travel_z_mm"]}
    return {
        "observe": d["sources"][candidate.source_id]["observe_pose"],
        "above_pick": above,
        "pick": grasp,
        "lift": above.copy(),
        "above_bin": bin_above,
        "drop": drop,
        "retreat": bin_above.copy(),
    }


def bin_fits(candidate, settings, bin_id):
    d = settings.data
    c = d["bins"][bin_id]
    # No in-bin packing: reject an item whose conservative envelope won't fit the opening.
    side = payload_side_mm(candidate, settings)
    local = local_points([c["drop_pose"]["x"], c["drop_pose"]["y"], c["drop_pose"]["z"]], c["container"])
    return all(
        c["container"]["min_mm"][i] + side / 2 < local[i] < c["container"]["max_mm"][i] - side / 2
        for i in (0, 1)
    )


def payload_side_mm(candidate, settings):
    # A TCP can be off-center on an irregular object. Bound the full observed cloud
    # around that TCP, including all yaw orientations, rather than just bbox width.
    radius = np.max(np.linalg.norm(candidate.points_mm[:, :2] - candidate.grasp_mm[:2], axis=1))
    return float(2 * (radius + settings.data["grasp"]["finger_margin_mm"]))
