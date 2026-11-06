import asyncio
from copy import copy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from boxer.bin_state import BinState
from boxer.controller import Sorter
from boxer.settings import SetupError


def hardware(observation):
    class Hardware:
        pass

    hw = Hardware()
    for name in ("assert_ready", "assert_idle", "move", "open", "stop", "set_motion_limits"):
        setattr(hw, name, AsyncMock())
    hw.holding = AsyncMock(side_effect=[False, True, True, False])
    hw.grab = AsyncMock(return_value=True)

    async def observe(source_id=None):
        if source_id == "source-2":
            empty = copy(observation)
            empty.candidates, empty.objects, empty.source_id = [], [], source_id
            return empty
        return observation

    hw.observe = AsyncMock(side_effect=observe)
    hw.joints = AsyncMock(return_value=observation.joints)
    return hw


async def test_dry_run_never_sends_hardware_commands(settings, observation, state):
    hw = hardware(observation)
    before = state.path.read_bytes()
    result = await Sorter(hw, settings, state, lambda _: None).dry_run()
    assert result["mode"] == "dry-run"
    for name in ("move", "open", "grab", "stop", "set_motion_limits"):
        getattr(hw, name).assert_not_called()
    assert state.path.read_bytes() == before


@pytest.mark.parametrize("color", ["blue", "green", "yellow", "red"])
async def test_successful_pick_to_color_bin(settings, observation, state, color):
    observation.candidates[0].color = color
    hw = hardware(observation)
    assert await Sorter(hw, settings, state, lambda _: None).run() == 1
    assert hw.open.await_count == 2
    assert hw.move.await_count == 9
    # Source observation, approach, descend, lift, above bin, release height, retreat.
    drop = hw.move.await_args_list[7].args[0]
    assert drop == settings.data["bins"]["bin-1"]["drop_pose"]
    assert hw.move.await_args_list[4].kwargs["linear"]
    hw.stop.assert_not_called()
    assert BinState.load(settings, state.path).available_bins(color) == ["bin-1"]


async def test_failed_grab_stops_without_release_or_bin_move(settings, observation, state):
    hw = hardware(observation)
    hw.grab.return_value = False
    with pytest.raises(RuntimeError, match="failed grasp"):
        await Sorter(hw, settings, state, lambda _: None).run()
    assert hw.move.await_count == 5
    assert hw.open.await_count == 1
    hw.stop.assert_awaited_once()
    assert state.data["pending"]["stage"] == "reserved"
    assert state.data["bins"]["bin-1"]["color"] is None


async def test_lost_item_after_lift_stops(settings, observation, state):
    hw = hardware(observation)
    hw.holding.side_effect = [False, False]
    with pytest.raises(RuntimeError, match="after lift"):
        await Sorter(hw, settings, state, lambda _: None).run()
    assert hw.move.await_count == 6
    hw.stop.assert_awaited_once()


async def test_missing_calibration_does_not_touch_robot(settings, observation, state):
    settings.data["calibration_reviewed"] = False
    hw = hardware(observation)
    with pytest.raises(SetupError):
        await Sorter(hw, settings, state).run()
    hw.assert_idle.assert_not_called()
    hw.stop.assert_not_called()


async def test_other_script_running_does_not_send_stop(settings, observation, state):
    hw = hardware(observation)
    hw.assert_idle.side_effect = RuntimeError("Other controller active")
    with pytest.raises(RuntimeError, match="Other controller"):
        await Sorter(hw, settings, state).run()
    hw.set_motion_limits.assert_not_called()
    hw.stop.assert_not_called()


async def test_cancelled_motion_stops_without_homing_or_opening(settings, observation, state):
    hw = hardware(observation)
    hw.move.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await Sorter(hw, settings, state, lambda _: None).run()
    hw.stop.assert_awaited_once()
    hw.open.assert_not_called()


async def test_stale_observation_never_starts_pick(settings, observation, state):
    observation.captured_at -= 30
    hw = hardware(observation)
    with pytest.raises(RuntimeError, match="stale"):
        await Sorter(hw, settings, state, lambda _: None).run()
    assert hw.move.await_count == 3
    hw.open.assert_not_called()


async def test_no_target_is_not_claimed_as_sorted(settings, observation, state):
    observation.candidates = []
    hw = hardware(observation)
    assert await Sorter(hw, settings, state, lambda _: None).run() == 0
    hw.grab.assert_not_called()


def test_random_selection_pool_includes_both_sources_and_all_heights(settings, observation, state):
    first = observation.candidates[0]
    second = replace(first, color="yellow", source_id="source-2", grasp_mm=first.grasp_mm + [0, 0, -10])

    class ChooseLast:
        def choice(self, values):
            assert {c.source_id for c, _ in values} == {"source-1", "source-2"}
            return values[-1]

    sorter = Sorter(hardware(observation), settings, state, rng=ChooseLast())
    target, destination = sorter.choose([first, second])
    assert target is second and destination == "bin-1"


async def test_source_two_is_scanned_and_selected_without_source_one_targets(settings, observation, state):
    import numpy as np

    c = observation.candidates[0]
    second = replace(
        c,
        source_id="source-2",
        points_mm=c.points_mm + [-300, 0, 0],
        center_mm=c.center_mm + [-300, 0, 0],
        grasp_mm=c.grasp_mm + [-300, 0, 0],
    )
    obs2 = replace(
        observation, candidates=[second], objects=[("blue", second.points_mm)], source_id="source-2"
    )
    empty = replace(observation, candidates=[], objects=[], source_id="source-1")
    hw = hardware(observation)

    async def observe(source_id=None):
        return obs2 if source_id == "source-2" else empty

    hw.observe.side_effect = observe
    assert await Sorter(hw, settings, state, lambda _: None).run() == 1
    np.testing.assert_allclose(hw.move.await_args_list[4].args[0]["x"], second.grasp_mm[0])
    assert [a.args[0] for a in hw.observe.await_args_list] == ["source-1", "source-2", "source-2"]


async def test_retreat_failure_preserves_confirmed_assignment(settings, observation, state):
    hw = hardware(observation)
    hw.move.side_effect = [None] * 8 + [RuntimeError("retreat failed")]
    with pytest.raises(RuntimeError, match="retreat failed"):
        await Sorter(hw, settings, state, lambda _: None).run()
    assert state.available_bins("blue") == ["bin-1"]
    assert state.data["pending"] is None


async def test_uncertain_release_blocks_restart_before_hardware_calls(settings, observation, state):
    hw = hardware(observation)
    hw.open.side_effect = [None, RuntimeError("connection lost at release")]
    with pytest.raises(RuntimeError, match="connection lost"):
        await Sorter(hw, settings, state, lambda _: None).run()
    restored = BinState.load(settings, state.path)
    assert restored.data["pending"]["stage"] == "release_pending"
    another = hardware(observation)
    with pytest.raises(SetupError, match="Unresolved"):
        await Sorter(another, settings, restored).run()
    another.assert_ready.assert_not_called()


async def test_refreshed_missing_target_never_starts_grasp(settings, observation, state):
    hw = hardware(observation)
    hw.observe.side_effect = [
        observation,
        replace(observation, candidates=[], objects=[]),
        replace(observation, candidates=[], objects=[]),
    ]
    with pytest.raises(RuntimeError, match="no longer graspable"):
        await Sorter(hw, settings, state, lambda _: None).run()
    hw.grab.assert_not_called()
    assert state.data["pending"] is None
