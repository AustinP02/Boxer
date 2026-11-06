# Boxer / Viam xArm6 handoff

Prepared 2026-11-06, America/New_York. This handoff incorporates the newer Claude-assisted files, not just the earlier conversation. It is a snapshot: another runner was active while the files were reviewed.

## Start here

- **Actual project:** `T:\Code Project\Boxer`.
- This Codex chat's workspace is `T:\NJIT\Viam x IEEE Robotics Hackathon`. Many scripts there are old staging/install/diagnostic helpers. **Do not copy those over the newer project files.**
- User's latest task was to inspect Claude's changes and export this Markdown. Robot work was paused in this chat. No robot commands were issued during the export/review.
- A separate process was observed running `python.exe -u boxer.py green --stage auto` (PIDs 27180 / 11644, likely launcher/interpreter pair). It was not started, stopped, or modified by the exporting chat. Recheck processes/current robot state before starting another controller. A running process alone does not prove a successful pickup or completed sort.
- The latest runner is **`src\green_to_container.py`**, launched by **`boxer.py green`**. Do not assume the older `sort` or `pickup` runner describes its current behavior.
- **Current offline test status: 46 tests, 45 pass, 1 error.** The error is caused by `green_to_container` changing `color_sorter.source_height_error` at import time. The 12 sorter tests pass in a fresh process when run alone. Details below.
- No complete physical pickup-and-place success was directly verified by this chat after Claude's changes. The newer cache/target files show scans and plans; they do not independently prove successful placements.

## User's goal and latest scene

Original goal: detect and track movable containers and individual colored shapes with a wrist-mounted depth camera, then sort by color. A two-block demonstration was the immediate goal, initially one green and one yellow block. The newer runner supports red, yellow, blue, and green.

Original setup: two large source trays, four narrow destination containers. **Claude's current code says the source trays have been removed, and blocks now sit directly on the table.** It substitutes two fixed pixel zones for source trays. Check the latest image and physical scene; old tray measurements remain in the config.

Original destination rule requested by user: prefer a bin already containing the same color; otherwise use an unassigned empty bin. Colors should follow contents when containers move. The newer `auto` demo assigns distinct color-free bins from one snapshot and optionally reuses cached coordinates. **That is not continuous container tracking and is not identical to the original routing requirement.**

User is under time pressure and prefers concrete commands and action over repeated calibration questions. They explicitly requested a supervised pickup and accepted relaxed checks, saying they had the emergency stop. Do not silently revert their later Claude-assisted choices. Still describe actual software behavior accurately, particularly commands that move without `--execute`.

## Hardware and coordinates

- Ufactory **xArm6**, component `arm`.
- Wrist-mounted Intel RealSense **D435**, component `cam`.
- Component `gripper`; Viam motion service `builtin`.
- Obstacle components `table`, `wall-front`, `wall-side`, `ceiling` use the Gripper API through `vmodutils/obstacle`.
- Arm mounting frame was verified as identity relative to `world`.
- **User measured full flange attachment face to fingertip: 160 mm.** The rubber gripping surface is 35 mm long. Do not confuse the pad length with tool length.
- Live gripper frame was verified as parent `arm`, translation `(0, 0, 160)` mm, orientation vector `(0, 0, 1)`, theta `180°`. It was previously 105 mm.
- Camera frame last verified: parent `arm`, translation `(-83, 14, 18)` mm, orientation vector `(0.03039099469, 0.00353799938, 0.9995318254)`, theta `-97.731173°`.
- Camera settings previously verified saved: color and depth, 1280×720, `align_color_depth: true`, `little_endian_depth: false`. Driver `realsense/realsense`, version 0.22.6 at that check.
- Intrinsics at that resolution: `fx=910.1533203125`, `fy=909.8717041015625`, `cx=645.0556640625`, `cy=375.7763671875`. Reported distortion coefficients were zero.
- Depth payload: `image/vnd.viam.dep`, magic `DEPTHMAP`, big-endian 8-byte width and height, big-endian uint16 distances in millimetres. Invalid pixels are not distances.
- Last inspected table geometry: box 3000×3000×200 mm, center world Z −123 mm, modeled top **−23 mm**. Wall-front at X 740 mm; wall-side at Y −500 mm; ceiling center Z 1050 mm. Reinspect saved Viam geometry before relying on these old values.

**Arm-end pose and gripper-tip pose are different points.** A value from Viam's arm pose UI cannot be substituted directly for a fingertip height.

## Configuration snapshot

`config\sort_config.json` is the actual config. `.env` holds credentials; do not paste secrets into handoffs or logs.

| Setting | Current recorded value |
|---|---|
| Arm scan XYZ (mm) | 255.78, 260.58, 568.82 |
| Scan orientation vector / theta | −0.04, −0.08, −1 / −117.94° |
| Grasp offset | **−15 mm**, chosen by user; downward relative to detected top |
| Source 1 rim Z | 67.37008156544425 mm |
| Source 2 rim Z | 63.58653342192505 mm |
| Destination release Z | 56.639194083392304 mm |
| Destination floor Z | 1.6391940833923044 mm |
| Transit tip Z | 409.47091166064706 mm |
| Pick orientation | `o_x=-0.04284880561496688`, `o_y=-0.08023701766185716`, `o_z=-0.9958544074582898`, `theta=62.05764094580942` |
| `motion_commissioned` | false in the saved config |
| `depth_aligned_verified` | true |
| `shared_bin_heights_verified` | true |
| Source inside dimensions | 138×108×60 mm, historical trays |
| Destination inside dimensions | 138×52×35 mm |
| Other checks | max grasp width 60 mm; max object span 120 mm; jaw margin 15 pixels; minimum area 300 pixels; approach 80 mm; depth spread 20 mm |

These heights were mathematically rebased after verifying the 105→160 mm tool-frame change. For each recorded pose, `new_world_z = old_world_z + 55 * recorded_tool_direction_z`; orientations did not change. The config's `tool_height_rebase` section preserves the old readings and direction vectors. Backup: `config\sort_config.before_tool_160mm.json`. **Do not apply another 55 mm correction.**

The two source rims still differ by about 3.78 mm. Original `sort check_ready` rejects a difference above 3 mm. The newer runner bypasses that comparison using a temporary config copy with both rims set to the larger value. Source trays are now reported removed; that old source-height model needs to be treated as historical, not a newly measured table surface.

## Project map

```text
boxer.py                         runpy launcher, puts src on sys.path
src/
  green_to_container.py          Claude-assisted current demo runner
  color_sorter.py                scan, teach, plan, watch, full guarded sort
  pickup_demo.py                 earlier approach / finish pickup-only stages
  floor_placement.py             strict fixed-orientation footprint placement
  sample_scan.py                 inspect_setup + motion_world_state + small scan
  object_instances.py            SAM2 individual mask hypotheses
  instance_analysis.py           color/depth measurements per individual mask
  container_detector.py          YOLOE profile / layout handling
  rim_fusion.py                  RGB black-rim + depth-edge support proposals
  container_tracking.py          track IDs; stale tracks never become pick targets
  sort_vision.py                 depth decoding, bin contents and routing helpers
  tracking_render.py             overlays
  check_alignment.py             color/depth alignment viewer
  project_paths.py               project-root-relative paths
config/sort_config.json
models/                         local YOLOE, SAM2, container profile
outputs/sort_runs/               latest RGB, overlay, depth, JSON (overwritten)
outputs/green_test/              current demo cache and remembered targets
outputs/pickup_demo/             older approach image/state
tests/
docs/
requirements/base.txt, ml.txt
examples/                       moving gripper/dance/wave scripts
archive/, cache/
```

Use `T:\Code Project\Boxer\.venv\Scripts\python.exe`. Existing environment is Python 3.15. Installed Ultralytics was 8.4.203, with torch 2.15. Models include `sam2.1_t.pt`, `yoloe-26s-seg.pt`, `container_profile.pt`. SAM2 runs slowly on CPU. No training dataset was created; SAM2 masks remain hypotheses, not validated custom block recognition.

## What Claude changed in the current demo

Read current source before editing. Header/docstrings lag actual behavior in places.

### `green_to_container.py`

- CLI stages: `scan`, `go`, `drop`, `all`, `auto`; `--colors` and `--use-cache`.
- **`SPEED = 50.0`**, raised from earlier lower speeds according to its source comment. Do not assume the old 2 or 5 degree/second demo speed. Setting persists on hardware.
- Import-time overrides modify the shared `color_sorter` module:
  - `choose_placement` → strict footprint fit with a color-free-bin clear-disc fallback (minimum radius 18 mm).
  - `validate_layout` → discards detected source trays and inserts fixed table pixel zones `(380,50,370,290)` and `(770,50,370,290)` for the saved scan view.
  - `tray_walls` → no source-tray geometry.
  - `snapshot` → relaxes empty-bin checks after the ordinary snapshot.
  - `source_height_error` → lambda returning `None`, since there are no longer source trays.
- Color-free bins may be treated as empty even when depth verification fails.
- `green_target()` uses mutable global `COLOR`, so it can process other colors despite its name. It ignores crowded-neighbor rejections. `usable()` in auto mode also ignores uneven-depth, boundary, and destination-rejection prefixes.
- Commissioning and matching-source-rim gates are bypassed in temporary config copies; saved `motion_commissioned` remains false.
- **`scan` is not motion-free:** `run()` sets speed and moves to the scan pose for scan/go/drop. CLI help/header saying "scan: no motion" is misleading. There is no `--execute` CLI option for this runner; stage selection controls execution.
- `go`: rescans, compares against the remembered target (15 mm shift limit), substitutes saved position/depth, then calls `sorter.pick_one(..., ignore_approach_depth=True)`.
- `drop`: checks holding, lifts, returns to scan, selects a bin, opens to release, and retreats.
- `all`: runs scan then go for each requested color; on go failure, attempts the drop recovery stage.
- **`auto`**: one connection, one initial scan or cache load, distinct bins per requested color, repeated pick/lift/drop. Cached coordinates can be reused for up to **900 seconds** with `--use-cache`. It does not continuously reacquire containers during this loop.
- In auto, default `--colors ['green']` is expanded to `red yellow blue green`. Even explicitly passing only `--colors green` currently triggers that same expansion.
- Auto's bin assignment first uses measured clear discs, then falls back to a bin's polygon center when usable floor depth is missing. It selects from color-free bins; this is not general matching-content routing.
- `DirectFallbackMotion` strips transforms, including the held-object envelope, from world-state requests. It catches planner constraint errors on near-vertical moves (XY difference <2 mm, orientation difference <1, Z difference ≤250 mm), then retries goal Z upward in 3 mm increments, up to 24 mm. **The implementation still calls the motion service; despite the class docstring, it does not currently call a direct arm-position move.** The actual accepted grasp height may therefore differ from configured −15 mm.
- Auto catches per-color failures and may continue if its holding check is false. If holding is true it breaks the loop, but still subsequently attempts a return to scan. Read this control flow before assuming "failure means stays still".
- No persisted final auto-result artifact was found during review; result summaries are printed to the launching terminal. Ask for that terminal result to establish actual physical success.

### `color_sorter.py`

- `pick_one` now accepts `ignore_approach_depth=False`.
- In the new go mode, a failed color/depth agreement check prints a warning and continues when this is true. Some other visibility checks still raise.
- The original strict `sort` runner still has commissioning and other checks in a fresh process. Importing `green_to_container` first changes several of those functions for that process.

## Earlier work that remains relevant

- Duplicate `table` collision-name errors were fixed by `sample_scan.motion_world_state()`: configured table/walls/ceiling are retained locally for measurements but omitted from supplemental move requests, because Viam already includes their configured geometry. Runtime geometry/transforms are normally preserved by that helper. The newer wrapper may further strip transforms.
- `teach` records the current configured gripper tip through motion service and no longer requires an available camera first. It does not move the robot.
- Separate per-piece SAM2 masks replaced same-color connected-component clustering in the active perception path. No mask means no color-blob fallback.
- Black-rim detection can work independently of YOLOE. Depth holes support edges only with RGB evidence; holes never provide valid distances.
- `floor_placement` checks a projected footprint with 5 mm margin against valid measured floor cells and keeps tool orientation fixed during transfer. This strict path is still present; the newer demo has fallbacks that are less restrictive.
- Earlier `pickup_demo` commanded a real slow approach above green, then stopped before descent/grasp because approach color/depth agreement failed. `outputs\pickup_demo\approach.jpg` shows its red prediction cross on the visible green block. At that stop, actual gripper tip was about `(389.992,275.067,127.404)` mm; no grasp was confirmed in that run.
- `pickup` also has `--in-place`, but a close view clipped the source tray and initially yielded no source detections. A fixed scan view was used afterward.
- No Git cleanup/reset or branch operations were performed. Earlier Git commands reported an unusual non-worktree state despite `.git` being present; inspect without resetting anything.

## Commands and side effects

Run from `T:\Code Project\Boxer`. These are references, **not instructions to resume motion during this paused handoff**.

Observation-only in a fresh process:

```powershell
.\.venv\Scripts\python.exe .\boxer.py sort plan
.\.venv\Scripts\python.exe .\boxer.py sort watch --preview
.\.venv\Scripts\python.exe .\boxer.py sort watch --preview --instances
```

Commands that can physically move/open/close/release:

```powershell
.\.venv\Scripts\python.exe .\boxer.py sort scan --move
.\.venv\Scripts\python.exe .\boxer.py pickup --color green --stage approach --execute
.\.venv\Scripts\python.exe .\boxer.py pickup --color green --stage finish --execute
.\.venv\Scripts\python.exe .\boxer.py green --stage scan
.\.venv\Scripts\python.exe .\boxer.py green --stage go --colors green
.\.venv\Scripts\python.exe .\boxer.py green --stage all --colors green yellow
.\.venv\Scripts\python.exe .\boxer.py green --stage auto --colors red yellow blue green
.\.venv\Scripts\python.exe .\boxer.py green --stage auto --use-cache --colors red yellow blue green
```

`sort run --execute --max-objects 2` is a different, older guarded runner. The saved config is still uncommissioned, so it is not the current fastest demo path.

## Artifacts and test evidence

- `outputs\green_test\cache.json`: latest reviewed snapshot at **14:49:07 EST / 19:49:07 UTC**. Four colors assigned distinct bins. Green used the no-depth bin-center fallback. This file is mutable and is not a completion log.
- `outputs\green_test\target_green.json`: reviewed timestamp **14:47:37 EST**, XYZ about `(443.527,107.794,28.731)` mm; remembered bin-3.
- `target_red.json` and `target_yellow.json` are earlier scans; `target.json` is an older artifact. They can disagree with the auto cache because scenes moved and assignments were recomputed.
- `outputs\sort_runs\latest.*` is overwritten by live runners. Its JSON is written before the green wrapper applies some relaxed overrides, so it can report unverified bins/rejected items that the wrapper later accepts.
- No `outputs\pickup_demo\target.json` existed at the handoff check; the approach image remains. Do not run the older pickup `finish` stage blindly.
- Test command executed during review:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: **46 tests, 1 error** (`test_height_disagreement_is_rejected_before_grasp`). Error: `TypeError: argument of type 'NoneType' is not a container or iterable` because the test expects a guard string after the green module imported a lambda replacing that guard. Discovery imports all test modules before running cases, so this is cross-module contamination, not evidence that the standalone guard implementation disappeared.

Isolation check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_color_sorter.py -v
```

Result: **12/12 pass**. The green tests currently test target selection only; they do not validate real movement, auto caching, relaxed placement, or recovery behavior.

## Recommended next-chat priorities

1. Read the live files and check whether another controller is still running; coordinate with the user before resuming paused robot work. Obtain the newer auto-run terminal summary/current physical result. Do not infer success from cache files.
2. Preserve Claude's recent implementation. Scope import-time overrides to the demo execution context rather than permanently changing the shared sorter module; fix the observed test contamination with meaningful isolation tests.
3. Correct stale help/docstrings so every stage that moves is identified accurately. Record commanded/accepted heights and per-color final results in a persistent run artifact.
4. Reconcile the measured 160 mm fingertip frame with the native collision model and upward-grasp retries; a successful approach does not prove a correct grasp height.
5. Choose the intended next milestone explicitly: repeat a fixed-scene demo, or restore the original contents-based routing and fresh tracking for moving containers. Current cached auto mode is a fixed-scene demonstration path.

Credentials were intentionally excluded. This file records inspected code and observations, not proof of a fully commissioned robot cell.

## Export verification

Exported at 2026-11-06 14:52:45 America/New_York (EST). Source-file SHA-256 fingerprints at export:

~~~json
[
  {
    "Path": "T:\\Code Project\\Boxer\\src\\green_to_container.py",
    "Hash": "76FBFA28F2A05196A6585DA49E2A2EE5C684930D2D45C5D08AF611478D1B8EFA"
  },
  {
    "Path": "T:\\Code Project\\Boxer\\src\\color_sorter.py",
    "Hash": "984FE3F2ED51520FB26768B7D6ADD8A3F5C01F397C16B0C4E8C14A960D96E162"
  },
  {
    "Path": "T:\\Code Project\\Boxer\\config\\sort_config.json",
    "Hash": "4CB88B460317805EF4B3BB1BB3B51086BD64CC268E8ED7C2A535EF4D2B14DF31"
  },
  {
    "Path": "T:\\Code Project\\Boxer\\boxer.py",
    "Hash": "96A0E219664461E55CBDC46B0DE5FA4037D1DAF13B9A11A53BA9149D47220DE4"
  }
]
~~~
