# Boxer: two-source color sorting with Viam

Boxer sorts **blue, green, yellow and red** objects from **two mixed input containers**
into **four initially unassigned output bins**. Shape affects the grasp, not the destination.
Target: `hackathons / Arm Farm / armfarm04`, xArm6 `arm`, UFactory `gripper`, RealSense D435 `cam`.

The work is on branch `feature/two-source-dynamic-bins`.

## Operating flow

1. Validate calibrated settings, required Viam resources, saved bin ownership, and that no
   interrupted placement remains unresolved. Confirm the arm is idle and the gripper empty.
2. Visit the taught observation pose for **source-1**, stop, and capture color/depth. Repeat
   for **source-2**. Perception processes each container separately in world coordinates.
3. Exclude unusable grasps: missing depth, excessive jaw width, unsupported centers/holes,
   insufficient wall/floor/finger clearance, or an object that will not fit its allowed bin.
4. Randomly choose **one object from the combined eligible pool**. There is no color order,
   height priority, or requirement to empty one source before the other.
5. For that object's color, use its already assigned bin. If the color is new, select the
   first fitting unassigned bin in numerical order. A color never spills into another bin
   merely because its existing bin is unsuitable.
6. Return to the selected source's observation pose and capture again. Match the selected
   object by source, color and 3D location; stop if it disappeared/moved or is no longer usable.
7. Persist a **pending placement** before beginning the pick. Approach, open, descend
   vertically, grab, lift vertically, and check gripper holding feedback.
8. Carry to the chosen bin, check holding again, persist `release_pending`, open, and check
   that the gripper reports empty. Commit that bin's color and increment its release count
   **before retreating**. The first confirmed release determines the bin's color.
9. Retreat and repeat with fresh scans of both sources. A batch is bounded by `--max-items`.
   No eligible target stops the batch for inspection; it does not prove the inputs are empty.

Example: yellow first claims bin-1; red next claims bin-2; another yellow returns to bin-1.
Blue might then claim bin-3 and green bin-4. The saved assignments persist across restarts.

## Viam integration

Four services are already installed: `boxer-blue`, `boxer-green`, `boxer-yellow`, `boxer-red`,
using `rdk:builtin:color_detector`. `builtin` motion is supplied by Viam. Shape-finder and
TensorFlow services are not dependencies of this sorter.

The Camera API supplies a registered colored PCD. Its XYZ values are converted from meters
into millimeters, and the actual camera intrinsics project it into an image. All four Viam
vision services receive that same capture through `GetDetections` (JPEG for SDK 0.85.2's wire
contract). The original point colors then filter each detection's depth points, and nearby
points are grouped into objects independently within each source container.

Viam frame transforms account for the moving wrist camera. Planned motions target the
**gripper TCP**, with both source containers, all four output containers, observed non-target
objects, and a carried-object envelope supplied through `WorldState`. Pick/lift/drop/retreat
use linear constraints. The driver is capped at 10 degrees/s and 20 degrees/s squared during
live runs. Errors and interruptions request arm/gripper stop without opening or homing.

## Install and preview

Python 3.11+; developed with Python 3.12. Run commands from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m boxer init
.\.venv\Scripts\python -m boxer inspect
.\.venv\Scripts\python -m boxer preview --image-only
```

This checkout already has `.venv` and `config/local.json`; skip creation commands if present.
The existing `.env` is loaded automatically. Required names: `VIAM_ADDRESS`, `VIAM_API_KEY_ID`,
`VIAM_API_KEY`. The target address must match the settings. `.env.example` contains placeholders;
no secrets were copied into new source files. The inherited Git commit already tracks `.env`;
`.gitignore` does not remove it from that existing history.

Preview writes `captures/latest.png` and `latest.json`. `--frames 100` repeatedly refreshes them.
`--image-only` permits color monitoring during another controller's movements. Normal `preview`
requires a stationary arm and calculates source-specific 3D candidates for whichever calibrated
sources are visible from the current view. Neither preview command moves hardware.

## Calibrate the six-container layout

Settings use **schema 2**. The earlier local one-source settings were backed up as
`config/local.v1.json`; the working `config/local.json` is migrated but remains uncalibrated.
No coordinates were estimated from the phone photo.

Use `sources.source-1` and `sources.source-2` for the two large containers (top-left and
bottom-left in the reference photo). Use `bins.bin-1` through `bins.bin-4` for the four narrow
outputs, top-to-bottom in that photo. Output calibration is independent of color.

Stop other test scripts before teaching/sorting. Use slow supervised movements with someone
at the robot. The last inspected arm configuration had collision sensitivity 0 (disabled);
review that setting and the physical stop before commissioning.

All coordinates are **world-frame millimeters**. Verify the configured gripper TCP against
physical fingertips before measuring from it, and verify wrist-camera/world alignment with
a known object's measured position. The RealSense module's registered PCD already uses the
color camera frame; do not apply another color/depth baseline offset.

For **each of the six containers**, measure the axis-aligned interior:

```json
{
  "container": {
    "min_mm": [100, 50, 0],
    "max_mm": [350, 300, 100],
    "wall_mm": 5
  }
}
```

These numbers are explanatory, not calibrated coordinates. `min_mm` gives the inner lower
X/Y corner and inner floor Z; `max_mm` gives the opposite inner corner and rim Z. The planner
receives a hollow floor-and-four-wall model. Rotated/curved containers need an appropriate
geometry model before use. The narrow output openings must actually fit the objects/fingers.

Teach one stationary camera observation pose per source, seeing that source's full interior:

```powershell
# Position the arm above source-1 first, then record its CURRENT tool pose:
.\.venv\Scripts\python -m boxer teach source-1
# Reposition above source-2, then record:
.\.venv\Scripts\python -m boxer teach source-2
```

Teach downward gripper drop poses centered over the four outputs; move to the correct bin
**between commands**. These commands only record the current pose and never move the robot:

```powershell
.\.venv\Scripts\python -m boxer teach bin-1
.\.venv\Scripts\python -m boxer teach bin-2
.\.venv\Scripts\python -m boxer teach bin-3
.\.venv\Scripts\python -m boxer teach bin-4
```

Set `grasp.payload_below_tcp_mm` to cover the maximum possible held-item extent below the TCP.
Set `travel_z_mm` to clear every rim plus that payload extent and `motion_limits.clearance_mm`
(default 50 mm). Each drop TCP must meet this clearance and be at or below travel height;
both observation TCP poses must be at or above travel height. Reachability still requires
Viam planning and physical commissioning.

Tune the four detector hues/cutoffs under actual lighting and keep them identical to local
`colors.*.hex` and `perception` settings. `boxer blocks` prints matching service blocks, which
can also be applied through MCP. Review insertion depth, jaw width, finger margin and the
gripper's closing-axis `grasp.yaw_offset_deg`. Color windows must not overlap.

After verifying the real measurements, camera frame and grasps, set `camera_frame_verified`
and `calibration_reviewed` true in local settings. Teaching again clears the latter flag.

```powershell
.\.venv\Scripts\python -m boxer validate
.\.venv\Scripts\python -m boxer preview
.\.venv\Scripts\python -m boxer run
```

`run` defaults to a **read-only dry run from the current stationary camera view**. It does not
visit both observation poses or validate IK. If no bin-state file exists, it explicitly uses
simulation-only empty-bin assumptions. It does not reserve bins, assign colors or write state.

## Initialize bins and execute

After calibration and physically checking that all four output bins are empty:

```powershell
.\.venv\Scripts\python -m boxer bins init --empty-confirmed
.\.venv\Scripts\python -m boxer bins status
# This command MOVES THE ROBOT; start with one separated object:
.\.venv\Scripts\python -m boxer run --execute --max-items 1
# After the first physically validated cycles, use a bounded batch:
.\.venv\Scripts\python -m boxer run --execute --max-items 20
```

Ownership and placement counts live in `state/bin-state.json`, saved atomically. Missing,
corrupt, mismatched-layout or unresolved state prevents live sorting. Output-bin recalibration
invalidates the state layout identity; source observation changes do not erase ownership.
Only one controller may use this physical robot. A lock next to the state file prevents two
local processes sharing that file; it does not coordinate independent files on different PCs.

After physically emptying **all four** output bins for a fresh batch:

```powershell
.\.venv\Scripts\python -m boxer bins reset --empty-confirmed
```

Reset backs up the previous state. Do not reset merely to dismiss a placement error while
bins contain items, and do not move items between assigned bins without reconciling state.

## Interrupted placements

A failed grasp, dropped item, timeout, Ctrl+C or uncertain release retains the placement
journal. Check `bins status`, inspect the source, destination, gripper and table, then resolve
that **specific pending placement**:

```powershell
# Use only if the pending item really was released into its intended bin:
.\.venv\Scripts\python -m boxer bins resolve --outcome released
# Otherwise, after accounting for/removing the item and ensuring the intended bin was not contaminated:
.\.venv\Scripts\python -m boxer bins resolve --outcome not-released
```

A confirmed release commits the bin color even if retreat fails. An unresolved release cannot
silently free the bin on restart. Both resolve commands change local bookkeeping, not hardware.
The gripper must be empty before restarting live sorting. To explicitly stop hardware:

```powershell
.\.venv\Scripts\python -m boxer stop
```

## Physical limits and verification

This is a top-down geometric pinch baseline. Touching same-color pieces can merge; occluded,
transparent, reflective, thin, hollow or entangled objects may require a different grasp or
instance-segmentation strategy. Random selection includes only eligible objects. Gripper
feedback does not prove that exactly one piece was lifted. Hidden obstacles cannot be inferred
from one view. Start commissioning with separated opaque pieces before the piled arrangement.

Outputs use fixed drop poses above their openings; bin fullness is not measured and there is
no in-bin packing. Placement counts mean confirmed gripper release over the selected bin,
not camera-verified landing. No autonomous physical pick/place has been commissioned yet.
Calibration remains deliberately unset; only read-only live integration was exercised.

`hello_wave.py` and `gripper_loop.py` are retired test entry points; their originals remain in
commit `d50e1e6`. `live_feed.py` runs read-only previews.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check boxer tests
```

Tests cover RGBD decoding/projection, hue masks, source isolation, grasps, frame math, payload
geometry, both-source selection, randomized eligible pools, all color routes, persistent
ownership, restart recovery, atomic-write failure, no-motion dry runs and failure stops.

References: [Viam Camera API](https://docs.viam.com/reference/apis/components/camera/),
[color detectors](https://docs.viam.com/reference/services/vision/color_detector/),
[RealSense module](https://github.com/viam-modules/viam-camera-realsense),
[frame system](https://docs.viam.com/motion-planning/reference/frame-system-api/),
[picking objects](https://docs.viam.com/motion-planning/move-an-arm/pick-an-object/).
