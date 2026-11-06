# Onboarding prompt: continue the Boxer robot sorter

You are taking over an existing project for Jude. Continue from this state; do not restart
the implementation or ask the user to repeat confirmed measurements.

## User goal and authorization

- Sort arbitrary-shaped blue, green, yellow and red pieces from **two mixed source tubs**
  into **four output bins**. Pick randomly among accessible/graspable objects across both
  sources; there is no required color order or requirement to empty one source first.
- Output bins start unassigned. The first confirmed placement assigns a bin its color.
  Later pieces of that color must use that same bin. Preserve ownership across restarts.
- **Color recognition belongs in Python libraries**, specifically the local OpenCV/NumPy
  pipeline. Do not move it back to Viam configuration/vision-service blocks.
- Detect the containers and calculate their positions rather than asking for manually
  entered XYZ coordinates. Known heights constrain rim planes above the detected table.
- Keep all code changes on `feature/two-source-dynamic-bins`, and push that branch to GitHub.
  Do not merge or push to main. GitHub updates are authorized.
- **Do not deploy code, modify robot configuration, or send arm/gripper motion commands.**
  The user explicitly limited changes to GitHub for now. Read-only robot camera/status/frame
  inspection and local files are authorized. Obtain a new explicit instruction before
  robot-side writes or physical actuation; do not infer authorization from "layout ready."
- Physical labels on containers are not required. Source/bin names are software identifiers.

## Repository and environment

- Workspace: `G:\Other computers\MINIPC\Boxer`.
- Remote: `https://github.com/AustinP02/Boxer.git`.
- Branch: `feature/two-source-dynamic-bins` (tracks the same origin branch).
- Windows PowerShell, Python 3.12, local environment `.venv`.
- Use an explicit working directory and `login: false` with shell tools; a login shell
  previously jumped to a different directory.
- Installed Viam SDK: `0.85.2`. Run via `.\.venv\Scripts\python`.
- README explains commands and the physical limitations. Read it and relevant source files.
- `.env` has the existing connection credentials, loaded automatically. Do not print keys,
  include them in screenshots/logs, or copy them into new code or documentation. The user's
  friend originally committed `.env`; this project did not rewrite that history.
- `config/local.json`, `state/`, `captures/`, and `.venv/` are ignored. They are present in
  this checkout but will not arrive when cloning GitHub onto another machine.

## Robot connection

- Organization: `hackathons`, ID `6d6c7293-bc67-43e4-9c8b-31c76aac27d3`.
- Location: `Arm Farm`, ID `310sld03v2`.
- Machine: `armfarm04`, ID `9a85b34a-7c7e-4f14-b71e-e493206021a9`.
- Main part: `armfarm4-main`, ID `b4ab0af6-d7bb-4500-977f-dbd55cafd670`.
- Address: `armfarm4-main.310sld03v2.viam.cloud`.
- Resources: UFactory xArm6 `arm`, UFactory `gripper`, RealSense D435 `cam`, motion `builtin`.
- Viam MCP can list/read machine config, status, API resources and registry documentation.
  Discover tools as needed. Respect the robot-write prohibition above.
- Existing `shape-detector` uses OpenCV, not ML. Earlier `mlmodel-1` lacked a model and was
  unhealthy; these services are irrelevant to the Python pipeline. A later `vision-1` was
  also added externally. Do not repair/remove unrelated services merely because they exist.
- Four `boxer-*` color services were once installed, but were already absent on the later
  configuration read. The current code does not need any Viam vision service.
- An earlier arm config had collision sensitivity 0. Treat this as unverified current state,
  and check before future physical commissioning; do not change it without authorization.

## Confirmed measurements — do not ask again

All dimensions are **outside measurements**. All heights are **tabletop-to-top-rim**.

| Type | Count | Width | Length | Height | Wall | Bottom |
| --- | --- | --- | --- | --- | --- | --- |
| Source | 2 | 108 mm | 138 mm | 60 mm | 2 mm | 2 mm |
| Destination | 4 | 52 mm | 138 mm | 35 mm | 2 mm | 2 mm |

Nominal inner openings are therefore 104 x 134 mm and 48 x 134 mm. Interior floor is
table height + 2 mm. Measurements are recorded in local settings and the example on GitHub.

All containers must rest directly on the same table for the current height-based detector.
The user previously put one on a wooden board; they were told to remove the height offset.
The intended arrangement is a parallel pair of large tubs beside a parallel row of four
narrow bins. Clear rims need visible edges; the user has traced the output rims in dark ink.
Small gaps between outer rims help resolve the boundaries.

## Latest verified status

The user just adjusted the layout and asked whether it is good, then requested this handoff.

- Latest read-only capture is under `captures/layout/`.
- `overlay.png`: verified visually after the latest detector fix. It shows both source tubs
  at the top of the live camera image and bins 1–4 left-to-right below them.
- `proposal.json`: `complete: true`, source count 2, destination count 4, stationary-scene
  check true, `motion_enabled: false`.
- `settings-proposed.json`: generated by `settings_from_proposal()` and passed all structural
  workcell/geometry checks. This is a draft, not deployed robot configuration.
- The active `config/local.json` still has unfilled operating container/pose fields and
  false physical-review flags. **The operating coordinates have not been saved/activated.**
- No autonomous physical pick/place has been commissioned. No motion was sent in these checks.
- Validation after the detector fix: all 70 tests pass, Ruff checks pass, and `git diff --check`
  is clean.
- Captures are time-specific. If the user moves anything or this is a different checkout,
  take a fresh read-only capture rather than trusting these coordinates.

## Why the latest code changed

The first detector pass discarded valid alternative rim rectangles too early. It then either
rejected the four-bin group or chose overlapping estimates, even though the physical layout
looked good. The latest fix:

1. Retains more candidate rim alternatives during nonmaximum suppression.
2. Searches compatible combinations rather than only consecutive candidates in a sorted list.
3. Requires the selected rim rectangles to be nonoverlapping within each group and across
   source/output groups. It does not shrink obstacles or relax the workcell overlap check.
4. Adds a regression where a higher-scoring overlapping rectangle must be skipped in favor
   of a valid four-bin group.

The same saved camera capture was reprocessed after this fix; all six containers were found
and the resulting operating geometry validated. Inspect `git log` for the final commit
containing this fix and this handoff.

## Code map

- `boxer/perception.py`: Python HSV bands (`COLOR_HUE_RANGES`), connected color regions,
  colored 3D clustering, source ROI filtering, conservative top-down pinch estimates.
- `boxer/layout.py`: tabletop-plane fit, known-height ray intersection, rim lines, rectangle
  alternatives, two-plus-four layout matching, overlay, and proposed settings generation.
- `boxer/geometry.py`: yaw-aware container coordinates, ROI transforms and overlap checks.
- `boxer/hardware.py`: Viam SDK adapter; camera and frame reads, planned TCP moves,
  gripper feedback, stop handling. Its write methods must not be called under current scope.
- `boxer/controller.py`: scans both sources, chooses randomly, refreshes selected source,
  journals intent, and executes sequential pick/place only under explicit live execution.
- `boxer/bin_state.py`: atomic saved color ownership, placement journal and recovery gate.
- `boxer/settings.py`: schema 2, measurements and workcell validation.
- `boxer/cli.py`: entry points; default run is read-only. `--execute` sends hardware commands.
- `config/sorter.example.json`: versioned defaults and confirmed dimensions.
- `tests/`: perception, layout, geometry, SDK adapter, controller, state persistence and CLI.

## Useful commands

```powershell
# Read-only camera tests; no deployment or motion:
.\.venv\Scripts\python -m boxer preview --image-only
.\.venv\Scripts\python -m boxer detect-layout

# Offline/local validation and tests:
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check boxer tests
.\.venv\Scripts\python -m boxer validate

# Saves local proposed operating geometry; does NOT deploy/move, and keeps review flags false:
.\.venv\Scripts\python -m boxer save-layout
```

`save-layout` backs up local settings and refuses incomplete proposals, unknown measurements,
overlapping models, a view below required travel height, or conflicting saved bin ownership.
The source observation poses can both use the actual capture pose because that view sees all
six containers. Output centers and travel height are calculated from the detected geometry.

Do not run `run --execute`, `stop`, old wave/gripper scripts, runtime speed commands, module
reloads, or robot configuration writes under the current authorization. `stop` itself sends
hardware commands; use it only when specifically warranted/authorized.

## Next steps and limits

1. Confirm the latest local draft/overlay remains current and tell the user this adjusted
   layout passed the detection and geometry checks.
2. Keep GitHub as the only deployment destination. Keep captures and credentials out of commits.
3. Physical verification of camera-to-world alignment and the configured gripper tool point
   remains necessary before setting `camera_frame_verified` / `calibration_reviewed` true.
   Image detection alone does not verify those physical transforms.
4. Future physical commissioning needs explicit authorization and an empty gripper/output
   bin check. Initialize bin state only after checking all four outputs empty. Pending/uncertain
   placements block restart until their real outcome is inspected and resolved.
5. **Unresolved grasp/placement limitation:** `bin_fits()` uses a conservative yaw-invariant
   square payload envelope. The 48 mm inner output width will reject many long pieces that
   could physically fit when oriented lengthwise. An oriented footprint/drop-yaw implementation
   and physical tests are needed; do not simply remove the fit check or claim all pieces fit.
6. Touching same-color pieces may merge; concave/occluded/entangled/reflective shapes can be
   skipped. This remains a top-down pinch baseline, not general dense-pile bin picking.
   Gripper feedback cannot prove exactly one item was lifted or camera-verify landing.
7. Output-bin fullness is not measured. Container identities are not automatically remapped
   while already-owned bins move. Preserve saved assignments and fail safely on uncertainty.

Keep explanations concrete and brief. Jude wants action and dislikes repeated permission
questions; only ask for information/authorization that is genuinely missing.
