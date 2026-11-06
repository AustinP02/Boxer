# Top-to-side inspection prototype

## Small moving calibration sample (xArm6)

`sample_scan.py` is the PHYSICAL sample runner. It moves relative to the current
world-frame arm tool pose: default +30 mm Y, -15 mm Z, +/-5 mm X, seven views.
Wrist orientation stays fixed; this is a small position/axis calibration sweep,
not the full wrist-tilting top-to-horizontal inspection. It saves images,
overlays, actual tool poses, and detections in a timestamped scan_captures folder.

First place the arm in a clear overhead starting pose through your existing
controls. Verify world +Z is up and the chosen sign of Y leads into clear space.
Measure clearance below the lowest attached tool/camera part; ensure the whole
small sweep is clear of objects, cables, and people. Validate configured obstacle
frames against the real table/walls. The planner only sees modeled geometry.

Inspect configuration and print absolute goals without motion:

```powershell
.\.venv\Scripts\python.exe .\boxer.py scan
```

Run the physical sample, if measured starting clearance is 100 mm:

```powershell
.\.venv\Scripts\python.exe .\boxer.py scan --execute --clearance-mm 100
```

Replace 100 with your measured clearance. The minimum accepted value is Z drop
plus 50 mm. Use `--travel-y -30` to reverse Y if that is the clear direction.
Sample size is capped at 50 mm Y travel, 25 mm Z drop, and 10 mm X excursion.
The script verifies the camera frame is under the arm and requires table,
wall-front, wall-side, and ceiling frames and geometry. It passes this geometry
to the builtin motion service and requests linear moves. A missing frame,
planner refusal, camera failure, or excessive tool position error stops the run.

The xArm module's documented DoCommand sets joint speed to 5 degrees/second;
this setting remains after the scan. No E-stop reset is attempted. Ctrl+C
requests arm.stop; keep the physical E-stop available for immediate stopping.
The arm stays at its last position; it does not automatically return on error.
Keep other scripts that command this arm stopped during the scan.

Only offline and fake-hardware checks have been run for this runner. It has not
been executed on the real robot. This code collects calibration observations;
it does not solve camera intrinsics, hand-eye calibration, or 3D reconstruction.

Motion source: https://docs.viam.com/motion-planning/move-an-arm/move-to-pose/
Speed source: https://github.com/viam-modules/viam-ufactory-xarm

`live_feed.py` now draws color and silhouette-based shape hypotheses. Colors:
red, yellow, blue, green. Shape references come from the supplied example video
(cuboid and cylinders) and photo (green notched shape). Inference compares each
silhouette against ALL reference classes, regardless of its detected color.
Ambiguous, unfamiliar, and clipped silhouettes remain uncertain. Match distance
is a heuristic, not a calibrated probability. Labels have a question mark.

Run from Boxer using `.\.venv\Scripts\python.exe .\boxer.py feed`.
R selects a table region; M shows masks; C clears the region; Q quits. S saves an
image plus detection JSON in `outputs/scan_captures`. Move the camera using your existing
robot controls and save separate overhead, oblique, left, right, and near-side
views. These views are not automatically fused or reconstructed into 3D.

## Requested motion

The camera is attached to the arm. Desired inspection: start farther overhead,
keep X approximately fixed with small left/right excursions, sweep along Y and
lower Z toward a side view while pointing the camera toward the target.

`inspection_arc.py` previews that geometry in millimeters. For example, this uses
an ARBITRARY target and dimensions for mathematical inspection, not robot goals:

```powershell
.\.venv\Scripts\python.exe .\boxer.py arc --target 0 0 0 --far 450 --near 250 --side 30
```

Camera positions follow a 75-degree vertical arc with decreasing radius. X has a
small sinusoidal offset, Y advances from the back (-Y) side toward the target,
and Z decreases. The endpoint is 15 degrees above horizontal. World-axis signs
must be checked against the real robot's frame. The curve is a preview only;
no motion SDK calls exist in this script.

Execution still requires the arm model, calibrated camera-to-tool transform,
target/table pose in the arm world frame, working distance/focus range, and
obstacle geometry for table, objects, fixtures, camera, and cables. Camera
positions need look-at orientations and conversion through the camera mount to
tool poses; the motion service must plan and verify the resulting arm path.
Neither pixel coordinates nor a bounding box provides metric depth.

## Limitations relevant to pill-box use

Color segmentation proposes separated colored regions; touching objects can
merge and poor lighting changes masks. A silhouette cannot reliably distinguish
every cylinder from a cuboid from every viewpoint. The white-capped labeled
medicine bottle is NOT modeled as a whole object by this prototype. Color and
shape alone do not identify a medication. For a pill-box task, define the actual
target classes and collect labeled views of those specific containers/items.
True 3D understanding needs calibrated depth or multi-view correspondence and
reconstruction, which are not implemented here.

## Offline review

`analyze_media.py input.mp4 annotated.mp4` writes an overlay video and a JSONL
report without connecting to Viam. It also accepts image input/output paths.
`build_shape_templates.py reference.mp4 reference.png` rebuilds the references
using the known labeled example instances. Results on those source examples
are a smoke check, not independent accuracy validation.

Sources:
- https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- https://docs.viam.com/operate/reference/components/camera/calibrate/
- https://docs.viam.com/motion-planning/how-planning-works/
