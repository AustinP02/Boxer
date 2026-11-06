"""Local workcell measurements. All Cartesian values are millimeters in world."""

import colorsys
import json
import math
from pathlib import Path

from .storage import atomic_json

COLORS = ("blue", "green", "yellow", "red")
SOURCE_IDS = ("source-1", "source-2")
BIN_IDS = ("bin-1", "bin-2", "bin-3", "bin-4")
HEX = dict(zip(COLORS, ("#0000FF", "#00FF00", "#FFFF00", "#FF0000")))
POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


class SetupError(ValueError):
    pass


def vector(value, name, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SetupError(f"{name} must contain {length} finite numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise SetupError(f"{name} must contain {length} finite numbers")
    return value


def check_pose(value, name):
    if not isinstance(value, dict) or set(value) != set(POSE_KEYS):
        raise SetupError(f"Teach {name}; expected {', '.join(POSE_KEYS)}")
    vector(list(value.values()), name, 7)
    norm = sum(value[k] ** 2 for k in ("o_x", "o_y", "o_z")) ** 0.5
    if abs(norm - 1) > 0.01:
        raise SetupError(f"{name} orientation vector must have unit length")


def check_container(container, name):
    if not isinstance(container, dict):
        raise SetupError(f"Measure {name}: inner min_mm, max_mm, wall_mm")
    lo = vector(container.get("min_mm"), f"{name}.min_mm")
    hi = vector(container.get("max_mm"), f"{name}.max_mm")
    if any(a >= b for a, b in zip(lo, hi)):
        raise SetupError(f"{name} inner min_mm must be below max_mm on every axis")
    wall = container.get("wall_mm")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise SetupError(f"{name}.wall_mm must be positive")


class Settings:
    def __init__(self, data, path=None):
        self.data = data
        self.path = Path(path).resolve() if path is not None else None
        if data.get("schema_version") != 2:
            raise SetupError("Settings schema 2 is required; migrate the one-source settings first")
        for key in ("address", "arm", "gripper", "camera", "motion"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise SetupError(f"Missing {key}")
        if set(data.get("colors", {})) != set(COLORS):
            raise SetupError("Exactly blue, green, yellow and red must be configured")
        for color, cfg in data["colors"].items():
            if not cfg.get("detector"):
                raise SetupError(f"Missing detector for {color}")
            try:
                h = cfg["hex"]
                if len(h) != 7 or h[0] != "#":
                    raise ValueError
                int(h[1:], 16)
            except (KeyError, TypeError, ValueError):
                raise SetupError(f"Invalid {color} hex") from None
        p = data.get("perception", {})
        for key in ("hue_tolerance", "min_saturation", "min_value"):
            v = p.get(key)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1:
                raise SetupError(f"perception.{key} must be between zero and one")
        hues = []
        for color, cfg in data["colors"].items():
            rgb = [int(cfg["hex"][i : i + 2], 16) / 255 for i in (1, 3, 5)]
            hue, sat, val = colorsys.rgb_to_hsv(*rgb)
            if sat < p["min_saturation"] or val < p["min_value"]:
                raise SetupError(f"{color} target hex is too gray or dark for its configured cutoffs")
            hues.append((color, hue * 360))
        for i, (color, hue) in enumerate(hues):
            for other, h in hues[i + 1 :]:
                if abs((hue - h + 180) % 360 - 180) <= p["hue_tolerance"] * 360:
                    raise SetupError(f"Color hue windows overlap: {color}, {other}")
        for group, keys in {
            "perception": ("voxel_mm", "cluster_radius_mm", "min_points", "min_object_mm", "max_object_mm"),
            "grasp": (
                "max_width_mm",
                "finger_margin_mm",
                "insertion_mm",
                "floor_clearance_mm",
                "wall_clearance_mm",
                "center_support_mm",
                "neighbor_clearance_mm",
                "payload_below_tcp_mm",
            ),
            "motion_limits": ("timeout_s", "settle_s", "clearance_mm", "linear_tolerance_mm"),
        }.items():
            for key in keys:
                v = data.get(group, {}).get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise SetupError(f"{group}.{key} must be positive and finite")
        if p["min_points"] < 5 or int(p["min_points"]) != p["min_points"]:
            raise SetupError("min_points must be an integer >= 5")
        if p["min_object_mm"] >= p["max_object_mm"]:
            raise SetupError("min_object_mm must be below max_object_mm")
        if set(data.get("sources", {})) != set(SOURCE_IDS):
            raise SetupError("Configure source-1 and source-2")
        if set(data.get("bins", {})) != set(BIN_IDS):
            raise SetupError("Configure bin-1 through bin-4 independently of color")
        if not isinstance(data.get("state_file"), str) or not data["state_file"]:
            raise SetupError("state_file is required")
        self.validate_optional_poses()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), path=path)

    def save(self, path):
        atomic_json(path, self.data)
        self.path = Path(path).resolve()

    @property
    def state_path(self):
        base = self.path.parent if self.path else Path.cwd() / "config"
        return (base / self.data["state_file"]).resolve()

    def validate_optional_poses(self):
        for source_id, source in self.data["sources"].items():
            if source.get("container") is not None:
                check_container(source["container"], source_id)
            if source.get("observe_pose") is not None:
                check_pose(source["observe_pose"], f"{source_id} observe_pose")
        for bin_id in BIN_IDS:
            c = self.data["bins"][bin_id]
            if c.get("drop_pose") is not None:
                check_pose(c["drop_pose"], f"{bin_id} drop_pose")
            if c.get("container") is not None:
                check_container(c["container"], f"{bin_id} container")

    def validate_workcell(self):
        d = self.data
        if d.get("calibration_reviewed") is not True or d.get("camera_frame_verified") is not True:
            raise SetupError(
                "Verify camera/world alignment and workcell measurements, then set both review flags true"
            )
        vector([d.get("travel_z_mm")], "travel_z_mm", 1)
        travel = d["travel_z_mm"]
        vector([d.get("grasp", {}).get("yaw_offset_deg")], "yaw_offset_deg", 1)
        containers = []
        for source_id, source in d["sources"].items():
            check_container(source.get("container"), source_id)
            check_pose(source.get("observe_pose"), f"{source_id} observe_pose")
            if source["observe_pose"]["z"] < travel:
                raise SetupError(f"{source_id} observe_pose TCP must be at or above travel_z_mm")
            containers.append((source_id, source["container"]))
        for bin_id in BIN_IDS:
            cfg = d["bins"][bin_id]
            check_container(cfg.get("container"), f"{bin_id} container")
            check_pose(cfg.get("drop_pose"), f"{bin_id} drop_pose")
            box, pose = cfg["container"], cfg["drop_pose"]
            margin = d["grasp"]["wall_clearance_mm"]
            if not all(
                box["min_mm"][i] + margin < pose[axis] < box["max_mm"][i] - margin
                for i, axis in enumerate(("x", "y"))
            ):
                raise SetupError(f"{bin_id} drop pose must be well inside its bin opening")
            if not (
                box["max_mm"][2] + d["motion_limits"]["clearance_mm"] + d["grasp"]["payload_below_tcp_mm"]
                <= pose["z"]
                <= travel
            ):
                raise SetupError(f"{bin_id} drop pose must clear the rim and be below travel_z_mm")
            if pose["o_z"] > -0.98:
                raise SetupError(f"{bin_id} drop pose must point the gripper down")
            containers.append((bin_id, box))
        for name, box in containers:
            if travel < (
                box["max_mm"][2] + d["motion_limits"]["clearance_mm"] + d["grasp"]["payload_below_tcp_mm"]
            ):
                raise SetupError(f"travel_z_mm must clear {name} rim")
        for i, (name, box) in enumerate(containers):
            for other, b in containers[i + 1 :]:
                if all(
                    box["min_mm"][j] - box["wall_mm"] < b["max_mm"][j] + b["wall_mm"]
                    and b["min_mm"][j] - b["wall_mm"] < box["max_mm"][j] + box["wall_mm"]
                    for j in (0, 1)
                ):
                    raise SetupError(f"Container footprints overlap: {name}, {other}")


def detector_blocks(settings):
    p = settings.data["perception"]
    return [
        {
            "name": c["detector"],
            "api": "rdk:service:vision",
            "model": "rdk:builtin:color_detector",
            "attributes": {
                "detect_color": c["hex"],
                "label": color,
                "camera_name": settings.data["camera"],
                "hue_tolerance_pct": p["hue_tolerance"],
                "segment_size_px": 60,
                "saturation_cutoff_pct": p["min_saturation"],
                "value_cutoff_pct": p["min_value"],
            },
        }
        for color, c in settings.data["colors"].items()
    ]
