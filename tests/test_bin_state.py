import copy
import json
from unittest.mock import patch

import pytest

from boxer.bin_state import BinState
from boxer.settings import SetupError


def release(state, color, bin_id="bin-1", source_id="source-1"):
    state.begin(color, bin_id, source_id)
    state.stage("grasped")
    state.stage("release_pending")
    state.confirm_release()


def test_first_release_assigns_color_and_survives_reload(settings, state):
    state.begin("yellow", "bin-1", "source-2")
    assert state.data["bins"]["bin-1"]["color"] is None
    state.stage("release_pending")
    state.confirm_release()
    restored = BinState.load(settings, state.path)
    assert restored.available_bins("yellow") == ["bin-1"]
    assert restored.data["bins"]["bin-1"]["released_count"] == 1
    assert "bin-1" not in restored.available_bins("red")


def test_repeated_color_cannot_claim_another_empty_bin(state):
    release(state, "green")
    with pytest.raises(SetupError, match="Cannot place"):
        state.begin("green", "bin-2", "source-1")
    release(state, "green")
    assert state.data["bins"]["bin-1"]["released_count"] == 2


def test_bin_never_accepts_a_different_color(state):
    release(state, "blue")
    with pytest.raises(SetupError, match="Cannot place"):
        state.begin("red", "bin-1", "source-2")


@pytest.mark.parametrize("stage", ["reserved", "grasped", "release_pending"])
def test_restart_blocks_unresolved_placement(settings, state, stage):
    state.begin("red", "bin-2", "source-2")
    if stage != "reserved":
        state.stage(stage)
    restored = BinState.load(settings, state.path)
    with pytest.raises(SetupError, match="Unresolved"):
        restored.available_bins("yellow")


def test_resolve_not_released_keeps_new_bin_unassigned(state):
    state.begin("yellow", "bin-3", "source-1")
    state.resolve(released=False)
    assert state.data["bins"]["bin-3"] == {"color": None, "released_count": 0}


def test_resolve_released_recovers_uncertain_placement(state):
    state.begin("yellow", "bin-3", "source-1")
    state.stage("release_pending")
    state.resolve(released=True)
    assert state.available_bins("yellow") == ["bin-3"]


def test_missing_state_is_not_silently_treated_as_empty(settings, tmp_path):
    with pytest.raises(SetupError, match="uninitialized"):
        BinState.load(settings, tmp_path / "missing.json")


def test_moved_output_bin_invalidates_state(settings, state):
    settings.data["bins"]["bin-1"]["drop_pose"]["x"] += 10
    with pytest.raises(SetupError, match="another output-bin layout"):
        BinState.load(settings, state.path)


def test_source_recalibration_preserves_existing_bin_ownership(settings, state):
    release(state, "yellow")
    settings.data["sources"]["source-1"]["observe_pose"]["z"] += 20
    assert BinState.load(settings, state.path).available_bins("yellow") == ["bin-1"]


def test_corrupt_state_cannot_be_used(settings, state):
    state.path.write_text("{truncated")
    with pytest.raises(SetupError, match="Cannot read"):
        BinState.load(settings, state.path)


def test_failed_atomic_write_preserves_pending_release_in_memory_and_on_disk(settings, state):
    state.begin("red", "bin-1", "source-1")
    state.stage("release_pending")
    before = copy.deepcopy(state.data)
    with patch("boxer.bin_state.atomic_json", side_effect=OSError("disk failure")), pytest.raises(OSError):
        state.confirm_release()
    assert state.data == before
    assert json.loads(state.path.read_text())["pending"]["stage"] == "release_pending"
    assert BinState.load(settings, state.path).data["bins"]["bin-1"]["color"] is None
