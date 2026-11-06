import argparse
import asyncio
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
from grpclib.exceptions import GRPCError, StreamTerminatedError

from .bin_state import BinState
from .controller import Sorter
from .hardware import connection
from .settings import BIN_IDS, SOURCE_IDS, Settings, SetupError, detector_blocks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/local.json"


@contextmanager
def controller_lock(state_path):
    """Prevents two local Boxer runs. Other computers/scripts require coordination."""
    path = state_path.with_name(state_path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SetupError(
            f"A Boxer controller lock exists at {path}. Stop its process; remove the lock only if stale"
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def build_parser():
    p = argparse.ArgumentParser(description="Boxer: sort blue, green, yellow and red objects with Viam")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create local settings without connecting")
    commands.add_parser("validate", help="Validate completed workcell measurements without connecting")
    commands.add_parser("blocks", help="Print required Viam detector blocks without connecting")
    commands.add_parser("inspect", help="Read resources, camera properties and machine health")
    preview = commands.add_parser("preview", help="Read-only detections, 3D candidates and PNG captures")
    preview.add_argument("--frames", type=int, default=1)
    preview.add_argument(
        "--image-only", action="store_true", help="Monitor RGB detections during other test scripts"
    )
    preview.add_argument("--output", type=Path, default=ROOT / "captures")
    teach = commands.add_parser("teach", help="Record CURRENT gripper world pose; does not move the arm")
    teach.add_argument("target", choices=(*SOURCE_IDS, *BIN_IDS))
    run = commands.add_parser("run", help="Dry-run by default; --execute sends hardware commands")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--max-items", type=int, default=1)
    bins = commands.add_parser("bins", help="Inspect or initialize persistent bin-color assignments")
    actions = bins.add_subparsers(dest="bin_action", required=True)
    actions.add_parser("status", help="Show saved bin assignments and any interrupted placement")
    for action in ("init", "reset"):
        initialize = actions.add_parser(
            action,
            help="Initialize empty bins"
            if action == "init"
            else "Clear assignments after physically emptying all bins",
        )
        initialize.add_argument(
            "--empty-confirmed",
            action="store_true",
            help="Operator has checked that all four output bins are empty",
        )
    resolve = actions.add_parser(
        "resolve", help="Resolve an interrupted placement after physically inspecting it"
    )
    resolve.add_argument("--outcome", required=True, choices=("released", "not-released"))
    commands.add_parser("stop", help="Explicitly stop arm and gripper; does not open or return home")
    return p


def save_preview(observation, output):
    output.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
    for color, boxes in observation.detections.items():
        for box in boxes:
            cv2.rectangle(image, (box.x_min, box.y_min), (box.x_max, box.y_max), (255, 255, 255), 2)
            cv2.putText(
                image,
                color,
                (box.x_min, max(18, box.y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
    if not cv2.imwrite(str(output / "latest.png"), image):
        raise RuntimeError("Could not write capture image")
    (output / "latest.json").write_text(json.dumps(observation.summary(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(observation.summary(), indent=2))
    print(f"Preview: {output / 'latest.png'}")


async def connected_command(args, settings, state=None):
    async with connection(settings) as hw:
        if args.command == "inspect":
            print(json.dumps(await hw.inspect(), indent=2))
        elif args.command == "teach":
            await hw.assert_idle()
            pose = await hw.current_pose()
            if args.target in SOURCE_IDS:
                settings.data["sources"][args.target]["observe_pose"] = pose
            else:
                settings.data["bins"][args.target]["drop_pose"] = pose
            settings.data["calibration_reviewed"] = False
            settings.save(args.config)
            print(f"Recorded {args.target} TCP pose in world: {json.dumps(pose)}")
        elif args.command == "preview":
            if args.frames < 1:
                raise SetupError("--frames must be positive")
            for i in range(args.frames):
                observation = await hw.color_preview() if args.image_only else await hw.observe()
                save_preview(observation, args.output)
                if i + 1 < args.frames:
                    await asyncio.sleep(0.5)
        elif args.command == "run":
            if args.execute:
                await Sorter(hw, settings, state).run(args.max_items)
            else:
                await Sorter(hw, settings, state).dry_run()
        elif args.command == "stop":
            await hw.stop()
            print("Arm and gripper stop acknowledged")


def bin_command(args, settings):
    path = settings.state_path
    if args.bin_action == "status":
        if not path.exists():
            print("Bin state is uninitialized; all four bins must be checked empty before bins init")
            return
        print(json.dumps(BinState.load(settings, path).summary(), indent=2))
        return
    with controller_lock(path):
        if args.bin_action in ("init", "reset"):
            if not args.empty_confirmed:
                raise SetupError("Physically empty/check all four bins, then supply --empty-confirmed")
            settings.validate_workcell()
            if args.bin_action == "init" and path.exists():
                raise SetupError("Bin state already exists; use bins status or reset after emptying all bins")
            if args.bin_action == "reset" and path.exists():
                shutil.copy2(path, path.with_name(path.name + ".previous.json"))
            state = BinState.empty(settings, path)
            state.save()
            print("Recorded all four output bins as empty and unassigned")
        else:
            state = BinState.load(settings, path)
            state.resolve(released=args.outcome == "released")
            print("Interrupted placement resolved; verify the gripper is empty before restarting")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            if args.config.exists():
                raise SetupError(f"Settings already exist: {args.config}")
            settings = Settings.load(ROOT / "config/sorter.example.json")
            settings.save(args.config)
            print(f"Created {args.config}; teach two source observation poses and four bin drop poses")
            return
        if not args.config.exists():
            raise SetupError("Run python -m boxer init first")
        settings = Settings.load(args.config)
        if args.command == "validate":
            settings.validate_workcell()
            print("Workcell settings valid. Physical calibration and motion still need bench verification.")
        elif args.command == "blocks":
            print(json.dumps({"services": detector_blocks(settings)}, indent=2))
        elif args.command == "bins":
            bin_command(args, settings)
        else:
            if args.command == "run":
                settings.validate_workcell()
                if args.max_items < 1:
                    raise SetupError("--max-items must be positive")
            if args.command == "run" and args.execute:
                with controller_lock(settings.state_path):
                    state = BinState.load(settings, settings.state_path)
                    state.require_resolved()
                    asyncio.run(connected_command(args, settings, state))
            elif args.command == "run":
                state = (
                    BinState.load(settings, settings.state_path)
                    if settings.state_path.exists()
                    else BinState.empty(settings)
                )
                asyncio.run(connected_command(args, settings, state))
            else:
                asyncio.run(connected_command(args, settings))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (SetupError, RuntimeError, ValueError, OSError, GRPCError, StreamTerminatedError) as error:
        print(f"Boxer: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
