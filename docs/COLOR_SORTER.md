# Moving-container color sorting prototype

## What is ready

Local ML container reacquisition + color/depth perception + routing + calibration
tools. `color_sorter.py scan --move` uses the requested arm scan pose:
X=257.92, Y=263.35, Z=505.5 mm; orientation=(-0.02,-0.06,-1), theta=-105.07 deg.
The machine's arm mounting frame was read and is identity in world; the program
checks this before applying these coordinates. No live robot motion or physical
sorting was executed during development.

PyTorch and Ultralytics were installed in Boxer's existing virtual environment.
The local YOLOE-26s segmentation weights and visual-prompt profile are included.
The profile uses the user's screenshot to illustrate the two loaded source trays
and the small clear destination-container type. No camera imagery was uploaded
to a training or inference service. Downloads were pretrained public weights.

Measured inside dimensions supplied by the user: source 138 x 108 x 60 mm;
destination 138 x 52 x 35 mm. These are stored in `config/sort_config.json`. A loose
physical-size check helps reject merged ML proposals. Inside dimensions are not
an exact collision model of the outer rim. Destination floor height can be derived
from a taught release point 20 mm above the rim and the measured 35 mm inside
height; `teach destination-floor` can override it with a measured value.

## Start here: LIVE VISION, NO MOVEMENT

Preview can observe from the current wrist-camera pose. For physical planning,
the specified scan pose is still required. From PowerShell:

```powershell
cd 'T:\Code Project\Boxer'
.\.venv\Scripts\python.exe .\boxer.py sort watch --preview
```

This continuously reacquires the visible containers with ML and displays colored
pieces inside the newly detected source regions. No fixed source image regions
are reused as movement targets. Q closes the viewing window. Move containers in
the view to test reacquisition. Preview tolerates fewer than four visible
destinations and ALWAYS displays the layout as unconfirmed: it never picks.

If the arm is elsewhere and you want it to physically move to the requested scan
pose, stop other arm-control scripts, clear the swept workspace, then run:

```powershell
.\.venv\Scripts\python.exe .\boxer.py sort scan --move
```

This command DOES move the arm, through Viam's motion service and configured
obstacles. The no-flag `scan` only observes in place and requires the correct
scan pose. It saves `outputs/sort_runs/latest.jpg`, `latest_overlay.jpg`, and `latest.json`.
The configured joint speed becomes 5 deg/s and remains set afterward.

## Routing rules requested by the user

- Source trays: two. Their contents can contain multiple colors.
- Destinations: four for full sorting; all four must be completely visible.
- If a destination contains yellow objects, it is assigned yellow. Same for red,
  blue, and green. Identity/assignment comes from contents, not fixed left/right.
- Prefer an existing same-color destination over any empty destination.
- Otherwise use the first verified-empty destination in current left-to-right
  order. After a successful placement it becomes assigned by its new contents.
- Multiple colors in one destination => exclude it and flag mixed contents.
- No colored pixels alone is NOT proof of emptiness. The depth surface must match
  the calibrated destination floor with sufficient valid coverage.
- Destination placement requires a clear floor patch large enough for a
  conservative circular footprint of the held piece. A matching-color bin with
  no verified room is skipped; this prototype does not stack pieces blindly.
- A hidden, clipped, merged, or missing container blocks full-layout operation.
- The system re-detects containers before each pick and rescans while holding
  before choosing a destination. Stored bin pixel positions are not drop targets.

Current live evaluation found both sources. The low/cropped destination area was
returned as ONE partial region, not two reliable separate containers. Full-view
examples and the measured container dimensions are needed before trusting the
destination detections. The program does not claim to have detected all four.

## Improve the container examples

Put both sources and all four destinations fully in view. Run `scan --move`,
then `setup` (stop watch before doing setup):

```powershell
.\.venv\Scripts\python.exe .\boxer.py sort setup
.\.venv\Scripts\python.exe .\boxer.py sort watch
```

Select the two full source containers and four full destination containers when
prompted. Enter accepts each rectangle. YOLOE builds visual class examples; it
does not train a new neural network from six boxes. Watch reacquires them each
iteration instead of assuming their example coordinates remain valid. Count and
overlap checks reject uncertain results. Rotations, reflections, occlusion, and
new appearances can still confuse the model. If it cannot separate the clear
containers, collect full-view labeled examples and fine-tune an instance model.

## Depth alignment and ruler check

The camera provides 1280x720 color + depth. Depth decoding was checked against the
actual device: typical valid ranges were about 424–510 mm at the captured pose.
This alone does NOT verify registration or mounting calibration.

For the RealSense module, verify `align_color_depth: true` in the CAMERA attributes
and visually check the depth boundaries match the color objects. Do not replace
the rest of the camera configuration. After this has really been verified:

```powershell
.\.venv\Scripts\python.exe .\boxer.py sort calibrate --aligned-depth
.\.venv\Scripts\python.exe .\boxer.py sort scan
.\.venv\Scripts\python.exe .\boxer.py sort ruler --mm 100
```

In the ruler window, click two actual tick marks 100 mm apart (or supply their
actual spacing). The tool compares depth-derived 3D separation with that value.
This is a scale check, not a replacement for hand-eye/pose calibration. A ruler
does not create one global pixels-to-millimeters conversion for objects at
different heights.

The existing Viam `shape-detector` service was queried and reported no configured
camera. Set its `camera_name` attribute to `cam` if using that service. The new
local ML container detector does not depend on that service.

## Physical sorting is NOT commissioned yet

`config/sort_config.json` deliberately has `motion_commissioned: false`. The physical
runner refuses to connect/move in `run` mode until calibration and commissioning
are complete. Do not flip this flag just to bypass an error. The viewing and
scan-position commands above are usable separately.

Still required:

1. Both source rims measured at the same world Z (or persistent identities for
   differently sized sources), and source/destination inside dimensions.
2. All four destinations' floor/rim heights checked. Current implementation
   assumes equal-height destination bins sliding on the same horizontal table.
3. A calibrated gripper TCP, downward picking orientation, grasp offset from
   object top, and collision-checked transit height.
4. Full-layout detection/localization tested across actual tray movements.
5. A supervised isolated-object pick and placement trial, then motion review.

Read-only teaching commands record the gripper's CURRENT world pose, using your
existing controls to place it first:

```powershell
.\.venv\Scripts\python.exe .\boxer.py sort teach source-1-rim
.\.venv\Scripts\python.exe .\boxer.py sort teach source-2-rim
.\.venv\Scripts\python.exe .\boxer.py sort teach destination-floor
.\.venv\Scripts\python.exe .\boxer.py sort teach destination
.\.venv\Scripts\python.exe .\boxer.py sort teach orientation
.\.venv\Scripts\python.exe .\boxer.py sort teach transit
```

For `destination`, the TCP should be 20 mm above the rim at a release location.
For `destination-floor`, it represents the floor plane; use a known/calibrated
TCP and existing controls. Teach commands themselves never move the arm.
Use `calibrate --grasp-offset-mm VALUE` only with the measured grasp offset; the
program does not invent it. `calibrate --shared-bin-heights` records that the
common destination-height assumption was checked physically.

## Limits

- The sample scene is a cluttered pile. Color regions can merge same-color
  touching pieces; contour/depth checks reject oversized/crowded proposals but
  are not a trained instance segmenter for every block. A failed grab stops.
- Shape labels are silhouettes, not validated pill/container identities.
- Wrist-camera views cannot monitor the entire table throughout every grasp.
  Reacquisition occurs at scan views and the target is checked at approach.
  Bin motion during a blind final descent remains outside this prototype's
  supported operation. Global continuous monitoring needs a suitable fixed view.
- Unknown/mixed/full/occluded bins are not silently assigned. No eligible target
  does not mean the source trays are empty.
- Held-object geometry is conservative and may make a plan fail. It does not
  justify ignoring collision failures, container walls, or unmodeled contents.
- No automatic E-stop reset, blind retry, return-home-on-error, or release on
  failure is issued. Physical E-stop remains the immediate stopping method.

## Validation and sources

Ten offline tests cover depth parsing, projection units, content routing,
unknown/mixed bins, empty-floor verification, region movement, duplicate proposals,
and the uncommissioned no-connection guard. Local ML inference reacquired both
source trays on a translated example after duplicate-group filtering. This is a
small smoke test, not an accuracy/safety certification or physical sort test.

- https://docs.ultralytics.com/models/yoloe/
- https://github.com/viam-modules/viam-camera-realsense
- https://docs.viam.com/tutorials/pick-and-place/perception-guided-picking/
- https://docs.viam.com/motion-planning/move-an-arm/move-to-pose/


## Visible black rims and missing-depth clues

A narrow matte-black outline around a container rim supplies RGB edges. The
sorter now searches for closed quadrilaterals near each ML container proposal,
refines a matching proposal when the visible rim supports it, and still applies
the measured-size and full-layout checks. `[rim supported]` means this extra
image evidence was found; it is not a depth-quality or safety certification.

When alignment has been verified, RGB edges adjacent to missing-depth bands can
also support a rim. A black depth pixel alone never creates a container detection
or a distance reading. Missing-depth pixels are not filled or used to grasp.
Visible-black-rim refinement can operate before depth alignment is verified.

Restart `python boxer.py sort watch --preview` after marking the rim. The
alignment viewer also has F to inspect the clues: yellow = dark RGB pixels,
green = RGB edge near missing depth. That overlay is diagnostic and cannot
verify alignment or distinguish a cable/hand from a container by itself.


Black-rim fallback update: a closed visible marker outline no longer needs an
overlapping ML detection. It proposes geometric candidates which must pass a
unique measured-size role match. Such candidates have no fabricated ML confidence.
Ambiguous role matches are rejected. Gray boxes in the preview show rejected ML
proposals, and latest.json includes detection_debug counts. Camera mounting and
rim-plane assumptions still affect the metric check; this does not enable picks.


## Tracking and revised scan pose

The scan pose is now X255.78/Y260.58/Z568.82 mm, orientation (-0.04,-0.08,-1),
theta -117.94 degrees, supplied by the user. Motion commissioning remains off.
Restart viewers after a code update. Preview assigns persistent track IDs and
smooths only display outlines. Briefly missed detections appear as orange dashed
STALE outlines for at most 3 seconds / 2 missed updates. They are excluded from
current detections, color assessment, counts, and routing. A camera-pose change
resets image tracks. Raw observed coordinates, not smoothed display coordinates,
remain the measurement data. `watch --preview --debug` shows rejected ML boxes.

Two 52 mm destination bins together resemble one 108 mm source tray. Independent
child rims take precedence over an enclosing proposal regardless of its class.
Clipped proposals no longer bypass the size filter or count as detected bins.


## Individual pieces rather than hue clusters

`watch --preview` is now a fast container-only preview. Add `--instances` to run
local SAM2 mask hypotheses on the colored foreground inside current source trays.
Colors are assigned within each separate mask, never used to join masks into one
object. Multiple seed masks on one surface are deduplicated; no masks means no
object proposals, not a fallback to one giant color blob. `plan`/`run` use this
instance path too. This costs CPU time; the first snapshot can take several
seconds. With SAM2, stale display outlines expire after at most 6 seconds or two
missed observations; they remain excluded from planning and routing.

SAM2 is a promptable segmenter, not a validated block detector for this pile.
Masks may describe visible faces, miss hidden pieces, or merge touching objects.
First demo: one isolated block, one verified empty destination, one supervised
transfer. Full cluttered-tray sorting has not been demonstrated.
