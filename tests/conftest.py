import json
from pathlib import Path

import numpy as np
import pytest

from boxer.bin_state import BinState
from boxer.hardware import Observation
from boxer.perception import Candidate
from boxer.settings import BIN_IDS, COLORS, Settings


@pytest.fixture
def settings():
    d = json.loads((Path(__file__).parents[1] / "config/sorter.example.json").read_text())
    d.update(calibration_reviewed=True, camera_frame_verified=True, travel_z_mm=300)
    d["sources"] = {
        "source-1": {
            "container": {"min_mm": [50, 50, 0], "max_mm": [250, 250, 100], "wall_mm": 5},
            "observe_pose": {"x": 150, "y": 150, "z": 350, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},
        },
        "source-2": {
            "container": {"min_mm": [-250, 50, 0], "max_mm": [-50, 250, 100], "wall_mm": 5},
            "observe_pose": {"x": -150, "y": 150, "z": 350, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},
        },
    }
    corners = ([300, 0, 0], [-450, 0, 0], [-200, -350, 0], [100, -350, 0])
    for bin_id, lo in zip(BIN_IDS, corners):
        d["bins"][bin_id]["container"] = {
            "min_mm": list(lo),
            "max_mm": [lo[0] + 150, lo[1] + 150, 100],
            "wall_mm": 5,
        }
        d["bins"][bin_id]["drop_pose"] = {
            "x": lo[0] + 75,
            "y": lo[1] + 75,
            "z": 250,
            "o_x": 0,
            "o_y": 0,
            "o_z": -1,
            "theta": 0,
        }
    d["motion_limits"]["settle_s"] = 0.001
    return Settings(d)


@pytest.fixture
def candidate():
    x, y = np.meshgrid(np.arange(130, 171, 2), np.arange(140, 161, 2))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 50.0)))
    return Candidate(
        "blue", points, np.array([150, 150, 50]), np.array([40, 20, 10]), np.array([150, 150, 38]), 90, 20
    )


@pytest.fixture
def observation(candidate):
    import time

    return Observation(
        [candidate],
        [],
        [("blue", candidate.points_mm)],
        {c: [] for c in COLORS},
        np.zeros((100, 100, 3), np.uint8),
        time.monotonic(),
        np.zeros(6),
    )


@pytest.fixture
def state(settings, tmp_path):
    result = BinState.empty(settings, tmp_path / "bins.json")
    result.save()
    return result
