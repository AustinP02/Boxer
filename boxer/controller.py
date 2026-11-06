"""Random eligible picks across two sources, with journaled color-exclusive output bins."""

import asyncio
import json
import random
import time

import numpy as np

from .planning import bin_fits, waypoints, world_state
from .settings import SOURCE_IDS, SetupError


class Sorter:
    def __init__(self, hardware, settings, state, log=print, rng=None):
        self.hw, self.settings, self.state, self.log = hardware, settings, state, log
        self.rng = rng if rng is not None else random.Random()

    def choose(self, candidates):
        """Uniform choice among eligible objects, with no color/source/height ordering."""
        eligible = []
        for candidate in candidates:
            destinations = self.state.available_bins(candidate.color)
            fitting = [b for b in destinations if bin_fits(candidate, self.settings, b)]
            if fitting:
                # First fitting unassigned bin is claimed by this color's first successful placement.
                eligible.append((candidate, fitting[0]))
            else:
                self.log(f"Skipped {candidate.color} from {candidate.source_id}: no compatible fitting bin")
        return self.rng.choice(eligible) if eligible else None

    async def dry_run(self):
        self.settings.validate_workcell()
        self.state.require_resolved()
        # A dry run never moves the camera. Only sources visible from its current pose contribute.
        observation = await self.hw.observe()
        selected = self.choose(observation.candidates)
        result = {"mode": "dry-run", "observation": observation.summary(), "bin_state": self.state.summary()}
        result["bin_state_mode"] = (
            "saved" if self.state.path is not None else "simulation only; empty bins assumed"
        )
        if selected:
            target, bin_id = selected
            result.update(
                source_id=target.source_id,
                destination_bin=bin_id,
                planned_waypoints=waypoints(target, self.settings, bin_id),
            )
        result["note"] = (
            "Current stationary view only; no motion, IK validation, reservation or bin-state write occurred."
        )
        self.log(json.dumps(result, indent=2))
        return result

    async def scan_sources(self):
        observations = {}
        for source_id in SOURCE_IDS:
            self.log(f"Observing {source_id}")
            await self.hw.move(
                self.settings.data["sources"][source_id]["observe_pose"], world_state(self.settings)
            )
            await asyncio.sleep(self.settings.data["motion_limits"]["settle_s"])
            observations[source_id] = await self.hw.observe(source_id)
        return observations

    async def refresh_target(self, target):
        # Scan order must not bias selection or leave a source-1 target using stale wrist coordinates.
        source = self.settings.data["sources"][target.source_id]
        await self.hw.move(source["observe_pose"], world_state(self.settings))
        await asyncio.sleep(self.settings.data["motion_limits"]["settle_s"])
        fresh = await self.hw.observe(target.source_id)
        matching = [
            c for c in fresh.candidates if c.color == target.color and c.source_id == target.source_id
        ]
        if not matching:
            raise RuntimeError("Selected object is no longer graspable; no pick attempted")
        nearest = min(matching, key=lambda c: np.linalg.norm(c.center_mm - target.center_mm))
        if np.linalg.norm(nearest.center_mm - target.center_mm) > 15:
            raise RuntimeError("Selected object changed position; no pick attempted")
        return nearest, fresh

    async def run(self, max_items=1):
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        self.settings.validate_workcell()
        self.state.require_resolved()
        if self.state.path is None or not self.state.path.exists():
            raise SetupError("Initialize the persistent bin state before live sorting")
        await self.hw.assert_ready()
        await self.hw.assert_idle()
        if await self.hw.holding():
            raise RuntimeError("Gripper already holds something; resolve it before starting")
        completed = 0
        active = False
        try:
            active = True
            await self.hw.set_motion_limits()
            for _ in range(max_items):
                observations = await self.scan_sources()
                pool = [c for observation in observations.values() for c in observation.candidates]
                selected = self.choose(pool)
                if selected is None:
                    self.log(
                        "No eligible grasp/destination across either source; stopped for inspection (not proof of emptiness)"
                    )
                    self.log(json.dumps({s: o.summary() for s, o in observations.items()}, indent=2))
                    break
                target, bin_id = selected
                target, fresh = await self.refresh_target(target)
                observations[target.source_id] = fresh
                if not bin_fits(target, self.settings, bin_id):
                    raise RuntimeError("Refreshed object no longer fits selected bin")
                if time.monotonic() - fresh.captured_at > 15:
                    raise RuntimeError("Perception is stale; no pick attempted")
                await self.hw.assert_idle()
                joints = await self.hw.joints()
                if joints.shape != fresh.joints.shape or np.max(np.abs(joints - fresh.joints)) > 0.15:
                    raise RuntimeError("Arm changed after observation; no pick attempted")
                self.log(f"Selected {json.dumps(target.summary())} -> {bin_id}")
                poses = waypoints(target, self.settings, bin_id)
                objects = [obj for observation in observations.values() for obj in observation.objects]
                empty_state = world_state(self.settings, objects, target)
                loaded_state = world_state(self.settings, objects, target, carrying=True)
                # Persist intent before starting a pick. Any subsequent uncertainty survives restart.
                self.state.begin(target.color, bin_id, target.source_id)
                await self.hw.move(poses["above_pick"], empty_state)
                await self.hw.open()
                await self.hw.move(poses["pick"], empty_state, linear=True)
                if not await self.hw.grab():
                    raise RuntimeError(
                        "Gripper reported a failed grasp; inspect source and resolve the pending placement"
                    )
                await self.hw.move(poses["lift"], loaded_state, linear=True)
                if not await self.hw.holding():
                    raise RuntimeError(
                        "No object held after lift; inspect workcell and resolve the pending placement"
                    )
                self.state.stage("grasped")
                await self.hw.move(poses["above_bin"], loaded_state)
                await self.hw.move(poses["drop"], loaded_state, linear=True)
                if not await self.hw.holding():
                    raise RuntimeError(
                        "Object lost during transport; inspect workcell and resolve the pending placement"
                    )
                self.state.stage("release_pending")
                await self.hw.open()
                if await self.hw.holding():
                    raise RuntimeError("Gripper still reports an object after release")
                # Commit before retreat: a retreat failure must not erase a successful placement.
                self.state.confirm_release()
                completed += 1
                self.log(f"Released item {completed}: {target.color}, {target.source_id} -> {bin_id}")
                await self.hw.move(poses["retreat"], empty_state, linear=True)
            return completed
        except BaseException:
            if active:
                try:
                    await asyncio.shield(self.hw.stop())
                except (Exception, asyncio.CancelledError) as stop_error:  # noqa: BLE001 - preserve original failure
                    self.log(f"Stop could not be confirmed ({type(stop_error).__name__}); use physical stop")
            raise
