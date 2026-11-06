# Boxer

Start the live ML preview (no robot movement):

```powershell
cd 'T:\Code Project\Boxer'
.\.venv\Scripts\python.exe .\boxer.py sort watch --preview
```

Press Q to close. Physical sorting is not commissioned yet.

## Main commands

| Command after `python boxer.py` | Purpose |
| --- | --- |
| `sort watch --preview` | Live ML container/color preview; no movement |
| `sort scan` | Observe at the configured scan pose; no movement |
| `sort scan --move` | Physically move to the configured scan pose and observe |
| `sort setup` | Select container examples for the ML model |
| `sort plan` | Inspect proposed color routing; no pick/place |
| `sort teach NAME` | Record a calibration value from the current gripper pose |
| `sort ruler --mm 100` | Check a known ruler distance in the latest aligned-depth scan |
| `alignment` | Read-only RGB/depth edge overlay to check registration |
| `feed` | Basic camera feed with color/shape estimates |
| `scan` | Inspect the small calibration sweep; `--execute` enables motion |
| `arc --target 0 0 0 --far 450 --near 250` | Print a mathematical arc preview |

## Folders

| Folder | Contents |
| --- | --- |
| `src/` | Application code and shared path settings |
| `config/` | Sorting pose, measured dimensions, calibration values |
| `models/` | Local ML weights, container profile, shape references |
| `outputs/` | Captured images, detections, scans |
| `docs/` | Detailed sorting and scan instructions |
| `examples/` | Earlier gripper, wave, and dance scripts (physical motion demos) |
| `requirements/` | Base and ML dependency lists |
| `archive/` | Preserved earlier script versions |
| `cache/` | Generated library settings and legacy bytecode |
| `tests/` | Offline checks; never operate the robot |

Credentials stay in root `.env`; the existing `.venv` is unchanged. Restart any
running viewer through the new launcher after this reorganization.

## Before a physical sorting demo

1. Make both source trays and all four destination bins fully visible.
2. Verify color/depth registration and the ruler scale check.
3. Teach source rims, destination release/floor, downward orientation, transit
   height, and the measured grasp offset with the calibrated gripper TCP.
4. Verify detection/localization and commission one supervised isolated-object
   transfer before enabling a full sort. Keep other arm-control scripts stopped.

Known inside dimensions: sources 138 x 108 x 60 mm; destinations 138 x 52 x 35 mm.
Bins are assigned by their contents: matching color first, otherwise verified
empty space. Unknown/mixed containers are excluded.

See [sorting guide](docs/COLOR_SORTER.md) and [scan guide](docs/SCAN_README.md).

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests use fake hardware and synthetic images; they do not connect to or
move the robot.
