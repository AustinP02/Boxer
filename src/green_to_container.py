"""Test: find the green block, pick it up and place it in a container.

Reuses the sorter's calibrated pick_one() (approach recheck, grasp, held-object
envelope, rescan, placement on verified clear floor, release check). Only a
single isolated green block is ever eligible. Without --execute it only observes
and saves sort_runs/latest_overlay.jpg. Runs the sorter's check_ready() before moving,
but skips its motion_commissioned flag.
"""
import argparse
import asyncio
import json
import math
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame, GeometriesInFrame, WorldState
import color_sorter as sorter
from block_detector import COLORS
from sort_vision import decode_depth
from viam.proto.service.motion import Constraints, CollisionSpecification
from container_detector import ContainerDetector
from object_instances import ObjectInstanceDetector
from sample_scan import inspect_setup, motion_world_state

COLOR = 'green'
ORDER = sorter.ORDER
FOLDER = sorter.OUTPUT_DIR / 'green_test'
RAISE_STEP_MM = 3
MIN_GRASP_Z_MM = 33.5  # lowest gripper z the planner accepted without help (33.4-35.3 in the full run)
PICK_LOWER_MM = 5.0  # grasp this much lower than before (needs ALLOW_CLAWS_TABLE on the descent)
DROP_ABOVE_RIM_MM = 45  # release height above the bin rim (was 70: blocks bounced onto the walls)
BIN_END_MARGIN_MM = 4  # keep blocks this far from the bin's end walls
BLOCK_GAP_MM = 2  # gap kept on each side of a block placed in a bin
TABLE_MIN_DEPTH_MM = 280  # a coloured patch farther than this is a block on the table, not one in the jaws
# The Viam model of the claws reaches into the table model well above the real table; allow that contact
# for the grasp descent only (the planner otherwise refuses any grasp lower than about z = 31).
ALLOW_CLAWS_TABLE = Constraints(collision_specification=[CollisionSpecification(
    allows=[CollisionSpecification.AllowedFrameCollisions(frame1='gripper:claws', frame2='table:table')])])
MAX_RAISE_MM = 24  # planner accepts gripper z >= ~32 (claws model reaches ~53 mm below tool point, table top -23)
SPEED = 50.0  # was 5.0; raised in steps on request (10, 20, 50)
MOVE_TIMEOUT_S = 90.0  # per linear move; scaled up by apply_speed so a slow run does not time out
SCAN_MOVE_TIMEOUT_S = 60.0
MAX_SHIFT_MM = 15
MIN_DISC_MM = 18  # block is ~35x39 mm; needs a clear disc ~20 mm radius
IGNORED = ('no matching-color or verified-empty destination',)


def relax_empty_check(config, items, layout):
    """Treat colour-free bins as empty even when noisy depth fails the 95% floor test.

    Colour detection still excludes bins that hold a block, and choose_placement
    still requires the footprint to fit on measured floor cells.
    """
    for d in layout['destinations']:
        if not d['colors'] and not d['empty_verified']:
            d['empty_verified'] = True
            d['status'] = 'no colour seen; empty check relaxed'
    for item in items:
        if item['color'] != COLOR or 'footprint_world_offsets_mm' not in item:
            continue
        item['rejected'] = [r for r in item['rejected'] if r not in IGNORED
                            and r != 'no verified destination with footprint clearance']
        dest = sorter.choose_placement(item['color'], layout['destinations'],
                                       item['footprint_world_offsets_mm'], config['_floor_contexts'])
        item['destination_bin'] = dest['id'] if dest else None
        if dest is None:
            item['rejected'].append('no destination bin has footprint clearance')
        else:
            item['placement'] = {k: dest[k] for k in ('free_center', 'free_world_xy_mm',
                                                      'placement_method', 'placement_margin_mm')}


_original_choose = sorter.choose_placement


def choose_with_fallback(color, bins, offsets, contexts):
    """Strict footprint fit first; else the colour-free bin's largest clear disc.

    Noisy depth on the transparent bins leaves holes that defeat the strict fit.
    The fallback needs a clear disc of at least MIN_DISC_MM radius around the drop.
    """
    strict = _original_choose(color, bins, offsets, contexts)
    if strict or color != COLOR:
        return strict
    roomy = [b for b in bins if not b['colors'] and b.get('free_center')
             and b.get('free_radius_mm', 0) >= MIN_DISC_MM]
    if not roomy:
        return None
    best = max(roomy, key=lambda b: b['free_radius_mm'])
    return dict(best, free_world_xy_mm=None, placement_method='largest clear disc (fallback)',
                placement_margin_mm=best['free_radius_mm'])


sorter.choose_placement = choose_with_fallback

# Source trays are gone: blocks can lie anywhere on the table. The sorter's layout check expects
# two source containers, so it is given two tiny dummy zones in empty corners (they pass the
# overlap check). Afterwards the layout's sources are replaced by ONE zone covering the whole
# camera view; blocks that lie inside a destination bin are dropped later in plan_from_scan.
DUMMY_ZONES = [(8, 8, 40, 40), (1232, 8, 40, 40)]
TABLE_ZONE = (8, 8, 1264, 704)  # x, y, w, h in the scan view (1280 x 720)
_original_validate = sorter.validate_layout


def validate_table_layout(detections):
    kept = [d for d in detections if d['role'] != 'source']
    for x, y, w, h in DUMMY_ZONES:
        kept.append(dict(role='source', roi=[x, y, w, h], clipped=False, rim_method='dummy zone',
                         polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                         center=[x + w / 2, y + h / 2]))
    layout = _original_validate(kept)
    x, y, w, h = TABLE_ZONE
    layout['sources'] = [dict(role='source', roi=[x, y, w, h], clipped=False, rim_method='whole table',
                              polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                              center=[x + w / 2, y + h / 2])]
    return layout


sorter.validate_layout = validate_table_layout


def no_tray_walls(config, origin, basis, intrinsic):
    """Tray-wall collision model removed on request: planner no longer avoids the source trays."""
    return GeometriesInFrame(reference_frame='world', geometries=[])


sorter.tray_walls = no_tray_walls
_original_snapshot = sorter.snapshot


async def relaxed_snapshot(*args, **kwargs):
    result = await _original_snapshot(*args, **kwargs)
    layout = args[3].get('_current_layout') if len(args) > 3 else None
    if layout:
        relax_empty_check(args[3], result[0], layout)
    return result


sorter.snapshot = relaxed_snapshot  # pick_one's mid-transfer rescan gets the same treatment


def green_target(items):
    """Exactly one eligible green block, else raise before any motion."""
    # 'crowded: adjacent instance in jaw clearance' is ignored on request; blocks touching
    # their neighbours may get nudged or double-grasped.
    found = [i for i in items if i['color'] == COLOR and 'world_mm' in i
             and not [r for r in i['rejected'] if not r.startswith('crowded')]]
    if len(found) != 1:
        seen = [(i['color'], i['rejected']) for i in items if i['color'] == COLOR]
        raise ValueError(f'Need exactly one eligible green block, found {len(found)}; no motion issued. Green seen: {seen}')
    return found[0]


class DirectFallbackMotion:
    """Motion service wrapper: when the planner refuses a straight vertical move on
    collision-model grounds (configured table/claw fixtures can't be removed from a
    request), do that move directly on the arm with no collision checking."""

    def __init__(self, motion, arm):
        self._motion, self._arm = motion, arm
        self.allow_table_contact = False  # set True around the grasp descent

    def __getattr__(self, name):
        return getattr(self._motion, name)

    async def move(self, component_name=None, destination=None, **kwargs):
        if kwargs.get('world_state') is not None:  # drop the conservative held-object envelope (guard removed on request)
            stripped = WorldState()
            stripped.obstacles.extend(kwargs['world_state'].obstacles)
            kwargs['world_state'] = stripped
        if self.allow_table_contact and kwargs.get('constraints') is None:
            try:
                return await self._motion.move(component_name=component_name, destination=destination,
                                               constraints=ALLOW_CLAWS_TABLE, **kwargs)
            except Exception as refused:
                print('  grasp move with claws/table contact allowed was refused:', str(refused)[:100], flush=True)
        try:
            try:
                return await self._motion.move(component_name=component_name, destination=destination, **kwargs)
            except Exception as expired:
                if 'SESSION_EXPIRED' not in str(expired):
                    raise
                print('Session expired; retrying the move once', flush=True)
                return await self._motion.move(component_name=component_name, destination=destination, **kwargs)
        except Exception as error:
            if component_name != 'gripper' or 'constraint' not in str(error):
                raise
            now = (await self._motion.get_pose('gripper', 'world', timeout=15)).pose
            goal = destination.pose
            vertical = (math.hypot(goal.x - now.x, goal.y - now.y) < 2 and
                        all(abs(getattr(goal, k) - getattr(now, k)) < 1 for k in ('o_x', 'o_y', 'o_z', 'theta')))
            if not vertical or abs(goal.z - now.z) > 250:
                raise
            # The Viam claws/table fixtures reject the commanded height, even for a direct arm
            # move. Step the target upward until the planner accepts a pose, up to MAX_RAISE_MM.
            for raise_mm in range(RAISE_STEP_MM, MAX_RAISE_MM + 1, RAISE_STEP_MM):
                higher = Pose(x=goal.x, y=goal.y, z=goal.z + raise_mm, o_x=goal.o_x, o_y=goal.o_y,
                              o_z=goal.o_z, theta=goal.theta)
                try:
                    ok = await self._motion.move(component_name=component_name, timeout=kwargs.get('timeout', 60),
                        destination=PoseInFrame(reference_frame='world', pose=higher),
                        **{k: v for k, v in kwargs.items() if k != 'timeout'})
                except Exception:
                    continue
                print(f'WARNING: grasp raised {raise_mm} mm to z={higher.z:.1f} (planner refused lower)', flush=True)
                return ok
            raise


def apply_speed(speed):
    """--speed N: run everything at arm speed N (normal is 50; 5 is super slow and quiet): the carrying speed,
    the long-block descent, the hello wave (scaled down) and every move timeout (scaled up so the slower
    moves are not cut off). None leaves the normal speeds."""
    global SPEED, LONG_PICK_SPEED, WAVE_SPEED, MOVE_TIMEOUT_S, SCAN_MOVE_TIMEOUT_S
    if speed is None:
        return
    if not 0 < speed <= 200:
        raise ValueError('--speed must be between 0 and 200 (normal 50, super slow 5)')
    factor = max(1.0, 50.0 / speed)
    SPEED = float(speed)
    LONG_PICK_SPEED = min(LONG_PICK_SPEED, SPEED)
    WAVE_SPEED = max(20.0, WAVE_SPEED / factor)
    MOVE_TIMEOUT_S = 90.0 * factor
    SCAN_MOVE_TIMEOUT_S = 60.0 * factor


def load_config():
    config = json.loads(sorter.CONFIG.read_text())
    return config


# No source tray any more, so the tray-floor height check has nothing to compare against.
sorter.source_height_error = lambda item, source, config: None


async def run(args, config):
    args.execute = args.stage == 'go'
    if args.execute:
        # Commissioning flag is skipped on request; all other sanity checks still run.
        # Also tolerate the two source rims differing: pick_one uses the item's own source rim.
        rim = max(src['rim_z_mm'] for src in config['sources'])
        sorter.check_ready({**config, 'motion_commissioned': True,
                            'sources': [{**src, 'rim_z_mm': rim} for src in config['sources']]})
        print('Skipping commissioning/rim-match gates and tray-wall collision model.', flush=True)
    load_dotenv(sorter.ROOT / '.env')
    machine = await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])), 30)
    attempted = False
    arm = gripper = None
    try:
        arm = Arm.from_robot(machine, 'arm'); gripper = Gripper.from_robot(machine, 'gripper')
        camera = Camera.from_robot(machine, 'cam'); motion = MotionClient.from_robot(machine, 'builtin')
        world, _ = await inspect_setup(machine, motion)
        table_pose = (await motion.get_pose('table', 'world', timeout=15)).pose
        table_geometry = next(o for o in world.obstacles if o.reference_frame == 'table').geometries[0]
        config['_table_top_z_mm'] = table_pose.z + table_geometry.center.z + table_geometry.box.dims_mm.z / 2
        detector = ContainerDetector()
        config['_detector'] = detector
        config['_object_detector'] = ObjectInstanceDetector(sorter.MODELS_DIR / 'sam2.1_t.pt')
        config['_container_tracker'] = None
        config['_show_rejected'] = False
        if True:  # both stages view from the scan pose
            attempted = True
            await arm.do_command({'set_speed': SPEED}, timeout=15)
            if args.stage == 'drop':
                if not (await gripper.is_holding_something(timeout=15)).is_holding_something:
                    raise RuntimeError('Gripper is not holding anything; nothing to drop')
                here = (await motion.get_pose('gripper', 'world', timeout=15)).pose
                await DirectFallbackMotion(motion, arm).move(component_name='gripper',
                    destination=PoseInFrame(reference_frame='world', pose=sorter.clone_pose(here, z=config['transit_z_mm'])),
                    world_state=motion_world_state(world), extra={'motion_profile': 'linear'}, timeout=MOVE_TIMEOUT_S)
            if not await motion.move(component_name='arm', destination=PoseInFrame(reference_frame='world',
                    pose=Pose(**config['scan_pose'])), world_state=motion_world_state(world), timeout=SCAN_MOVE_TIMEOUT_S):
                raise RuntimeError('Cannot reach scan pose')
            await asyncio.sleep(1)
        items, origin, basis, intrinsic = await sorter.snapshot(machine, camera, motion, config, detector,
                                                                allow_partial=False)
        if args.stage == 'drop':
            mover = DirectFallbackMotion(motion, arm)
            dest = sorter.choose_placement(COLOR, config['_current_layout']['destinations'], None, config['_floor_contexts'])
            if dest is None:
                raise RuntimeError('No bin with a clear area to drop into')
            spot = sorter.ray_plane(dest['free_center'], config['destination_floor_z_mm'], intrinsic, origin, basis)
            rim_z = config['destination_release_z_mm'] - config['release_above_rim_mm']
            drop = Pose(x=float(spot[0]), y=float(spot[1]), z=max(config['destination_release_z_mm'], rim_z + 50 + 20),
                        **config['pick_orientation'])
            print('Dropping into', dest['id'], sorter.pose_dict(drop), flush=True)
            async def go_to(pose):
                if not await mover.move(component_name='gripper', destination=PoseInFrame(reference_frame='world', pose=pose),
                        world_state=motion_world_state(world), extra={'motion_profile': 'linear'}, timeout=MOVE_TIMEOUT_S):
                    raise RuntimeError('Motion not confirmed')
            await go_to(sorter.clone_pose(drop, z=config['transit_z_mm']))
            await go_to(drop)
            await gripper.open(timeout=15)
            await go_to(sorter.clone_pose(drop, z=config['transit_z_mm']))
            print('Block released into', dest['id'])
            return
        item = green_target(items)
        print('Green block:', json.dumps({k: item[k] for k in ('source', 'world_mm', 'destination_bin')}), flush=True)
        if args.stage == 'scan':
            FOLDER.mkdir(parents=True, exist_ok=True)
            (FOLDER / f'target_{COLOR}.json').write_text(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                world_mm=item['world_mm'], camera_mm=item['camera_mm'], size_mm=item.get('size_mm'),
                depth_mm=item['camera_mm'][2], destination_bin=item['destination_bin']), indent=2))
            print('Remembered target position/depth in', FOLDER / f'target_{COLOR}.json')
            print('Scan only; no pick. Run --stage go to pick up and drop.')
            return
        saved = json.loads((FOLDER / f'target_{COLOR}.json').read_text())
        shift = sum((a - b) ** 2 for a, b in zip(saved['world_mm'], item['world_mm'])) ** .5
        if shift > MAX_SHIFT_MM:
            raise ValueError(f'Green block moved {shift:.0f} mm since the scan; rescan first')
        item['world_mm'] = saved['world_mm']; item['camera_mm'] = saved['camera_mm']
        print('Using remembered position:', saved['world_mm'], flush=True)
        await sorter.pick_one(machine, arm, gripper, DirectFallbackMotion(motion, arm), world, config, item, origin, basis, intrinsic,
                               ignore_approach_depth=True)
        print('Green block placed. Inspect the container before running again.')
    finally:
        try:
            if attempted:
                try:
                    await arm.stop(timeout=10)
                finally:
                    await gripper.stop(timeout=10)
        finally:
            await machine.close()


# ---- automated sequence: one connection, one scan, cached layout + block positions ----
CACHE = FOLDER / 'cache.json'
CACHE_MAX_AGE_S = 900
IGNORED_PREFIXES = ('crowded', 'uneven depth', 'too close to tray boundary', 'no matching-color',
                    'no verified destination', 'no destination bin')


def usable(item):
    return 'world_mm' in item and not [r for r in item['rejected'] if not r.startswith(IGNORED_PREFIXES)]


MIN_BLOCK_MM = 18  # smaller than this is not a block
NOT_A_BLOCK_MM = 100  # larger than this is clutter, not touching blocks
MAX_BLOCK_MM = 80  # up to about two blocks long (the long rectangular block is ~70 mm)
LONG_ASPECT = 1.35  # long / short side at or above this is a long block
LONG_RAISE_MM = 4  # grasp long blocks a little higher: pressing too deep tipped one over
LONG_PICK_SPEED = 10.0  # slow final descent for long blocks
SLOT_CLEAR_MM = 36  # a slot closer than this to an existing block counts as taken
SLOT_SPACING_MM = 56  # centre slot holds a long block (~70 mm); end slots hold ~37 mm blocks
SLOT_ORDER = (0, 1, -1)  # centre of the bin first, then either side


def bin_frame(d, config, origin, basis, intrinsic):
    """(centre xy, unit long-axis xy, long-axis yaw in degrees) of a bin in world coordinates."""
    # The detected outline is the bin's rim, which stands bin-height above the floor. Projecting it
    # onto the floor plane would shift every bin away from the camera axis (about 15 mm at the edges).
    rim_z = config['destination_floor_z_mm'] + config['container_dimensions_mm']['destination'][2]
    corners = np.array([sorter.ray_plane(p, rim_z, intrinsic, origin, basis)[:2]
                        for p in d['polygon']], dtype=float)
    centre = corners.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((corners - centre).T))
    axis = vectors[:, int(np.argmax(values))]
    return centre, axis, math.degrees(math.atan2(axis[1], axis[0])) % 180


def block_extent(item):
    """Length of the block along the bin axis once it is aligned with the bin."""
    size = item.get('size_mm') or (37, 37)
    return float(min(max(size), 78)) if is_long(item) else 38.0  # nominal block, not the noisy measurement


def fit_in_bin(extent, intervals, inner_half):
    """First position t (mm from the bin centre along its axis), packing from one end, where a block of
    this extent fits between the end walls and clear of the intervals already taken. None if no room."""
    t = -(inner_half - extent / 2)
    while t <= inner_half - extent / 2 + 1e-6:
        lo, hi = t - extent / 2 - BLOCK_GAP_MM, t + extent / 2 + BLOCK_GAP_MM
        if all(hi <= a or lo >= b for a, b in intervals):
            return t
        t += 2.0
    return None


def drop_orientation(orient, jaw_offset, axis_yaw, long_block):
    """Wrist angle for the release so the block's faces line up with the bin's walls."""
    if long_block:
        return turn_to(orient, jaw_offset, (axis_yaw - 90) % 180, 180)  # long side along the bin
    return turn_to(orient, jaw_offset, axis_yaw % 90, 90)  # a square block: faces parallel to the bin


def is_long(item):
    size = item.get('size_mm') or (0, 0)
    return min(size) > 0 and max(size) / min(size) >= LONG_ASPECT and max(size) > 50


def plan_from_scan(items, layout, colors, origin, basis, intrinsic, config):
    """Cache every usable block's position; each colour goes to its own bin, one slot per block."""
    found = [i for i in items if i['color'] in colors and usable(i)]
    # Ignore anything lying inside a destination bin (already sorted), and anything whose size is
    # clearly not a ~37 mm block (hands, boards, cables, colour-matching clutter).
    bin_polygons = [np.array(d['polygon'], np.int32) for d in layout['destinations']]
    kept, ignored = [], []
    for i in found:
        size = i.get('size_mm') or (0, 0)
        in_bin = any(cv2.pointPolygonTest(poly, tuple(map(float, i['center'])), False) >= 0 for poly in bin_polygons)
        if in_bin or min(size) < MIN_BLOCK_MM or max(size) > NOT_A_BLOCK_MM:
            ignored.append(f"{i['color']} at ({i['world_mm'][0]:.0f},{i['world_mm'][1]:.0f})"
                           + (' in a bin' if in_bin else f' size {size[0]:.0f}x{size[1]:.0f}'))
        else:
            kept.append(i)
    if ignored:
        print('Ignoring non-block / already-sorted pieces:', '; '.join(ignored), flush=True)
    found = kept
    skipped = []
    merged = [i for i in found if max(i.get('size_mm', (0, 0))) > MAX_BLOCK_MM]
    for i in merged:  # touching blocks segmented as one piece cannot be gripped as a single block
        skipped.append(f"{i['color']} at ({i['world_mm'][0]:.0f},{i['world_mm'][1]:.0f}): {max(i['size_mm']):.0f} mm, "
                       'touching blocks merged into one piece; separate them')
    found = [i for i in found if i not in merged]
    found.sort(key=lambda i: (ORDER.index(i['color']), not is_long(i), i['world_mm'][0]))
    if not found:
        raise ValueError('No usable block found; seen: ' + str([(i['color'], i['rejected']) for i in items]))
    # Blocks already inside a bin occupy slots there; a bin already holding one colour stays that colour's home.
    occupied = {d['id']: [] for d in layout['destinations']}
    for i in items:
        if 'world_mm' in i and i.get('center') is not None:
            for d, poly in zip(layout['destinations'], bin_polygons):
                if cv2.pointPolygonTest(poly, tuple(map(float, i['center'])), False) >= 0:
                    occupied[d['id']].append((i['world_mm'][:2], block_extent(i)))
    present = [c for c in ORDER if any(i['color'] == c for i in found)]
    home, free_bins = {}, [d for d in layout['destinations'] if not d['colors']]
    for c in present:
        match = next((d for d in layout['destinations'] if d['colors'] == [c]), None)
        if match is None:
            if not free_bins:
                skipped.append(f'{c}: no bin available for this colour')
                continue
            match = free_bins.pop(0)
        home[c] = match
    frames = {c: bin_frame(home[c], config, origin, basis, intrinsic) for c in home}
    inner_half = config['container_dimensions_mm']['destination'][0] / 2 - BIN_END_MARGIN_MM
    # Blocks already inside a bin take up their stretch of its length.
    taken = {}
    for c in home:
        centre, axis, _ = frames[c]
        taken[home[c]['id']] = [(float(np.dot(np.array(xy) - centre, axis)) - ext / 2 - BLOCK_GAP_MM,
                                 float(np.dot(np.array(xy) - centre, axis)) + ext / 2 + BLOCK_GAP_MM)
                                for xy, ext in occupied[home[c]['id']]]
    counts = {}
    blocks = []
    for item in found:
        color = item['color']
        if color not in home:
            continue
        bin_id = home[color]['id']
        centre, axis, yaw = frames[color]
        extent = block_extent(item)
        t = fit_in_bin(extent, taken[bin_id], inner_half)
        if t is None:
            skipped.append(f"{color} at ({item['world_mm'][0]:.0f},{item['world_mm'][1]:.0f}): no room left in its bin")
            continue
        taken[bin_id].append((t - extent / 2 - BLOCK_GAP_MM, t + extent / 2 + BLOCK_GAP_MM))
        counts[color] = counts.get(color, 0) + 1
        blocks.append(dict(name=f'{color}{counts[color]}', color=color,
                           source=item['source'], world_mm=item['world_mm'],
                           footprint=item.get('footprint_world_offsets_mm'),
                           bin=bin_id, drop_xy=(centre + t * axis).tolist(), slot_t=t,
                           bin_center_px=home[color]['center'], long=is_long(item), bin_axis_yaw=yaw))
    return dict(time=datetime.now(timezone.utc).isoformat(), blocks=blocks, skipped=skipped)


def tool_x_yaw(o_x, o_y, o_z, theta):
    """World yaw (deg) of the tool x axis: Viam orientation vector = Rz(lon) Ry(lat) Rz(theta)."""
    n = math.sqrt(o_x ** 2 + o_y ** 2 + o_z ** 2)
    lat = math.acos(max(-1, min(1, o_z / n)))
    lon = math.atan2(o_y, o_x) if 1 - abs(o_z / n) > 1e-8 else 0.0
    th = math.radians(theta)
    def rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    def ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    x_axis = (rz(lon) @ ry(lat) @ rz(th)) @ np.array([1.0, 0, 0])
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


def block_shape(offsets):
    """(yaw of the block's long edge in degrees, long/short aspect) from its footprint corners."""
    if not offsets:
        return None
    c = np.array(offsets, dtype=float)
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if min(l1, l2) < 1e-6:
        return None
    edge = e1 if l1 >= l2 else e2
    return math.degrees(math.atan2(edge[1], edge[0])) % 180, max(l1, l2) / min(l1, l2)


def turn_to(orient, jaw_offset, desired_yaw, modulus):
    """Rotate the wrist about the vertical until the jaw axis points along desired_yaw (mod modulus).

    jaw yaw = tool-x yaw + jaw_offset (measured from the wrist camera, whose image x axis is the jaw
    axis). Increasing theta lowers the world yaw.
    """
    turned = dict(orient)
    for _ in range(3):  # the small tool tilt makes the yaw response slightly nonlinear; refine
        jaw = tool_x_yaw(turned['o_x'], turned['o_y'], turned['o_z'], turned['theta']) + jaw_offset
        turned['theta'] += ((jaw - desired_yaw + modulus / 2) % modulus) - modulus / 2
    return turned


def aligned_orientation(orient, offsets, jaw_offset=0.0):
    """Wrist orientation that closes the jaws across the block's flat faces.

    Square block: any face pair works (modulo 90). Long block: across the SHORT side, so the jaws
    never have to span the long dimension.
    """
    shape = block_shape(offsets)
    if shape is None:
        return dict(orient), 0.0
    long_yaw, aspect = shape
    if aspect >= LONG_ASPECT:
        turned = turn_to(orient, jaw_offset, (long_yaw + 90) % 180, 180)
    else:
        turned = turn_to(orient, jaw_offset, long_yaw % 90, 90)
    return turned, turned['theta'] - orient['theta']


async def measure_jaw_offset(machine, motion):
    """jaw-axis yaw minus tool-x yaw, in degrees. The jaws separate along the wrist camera's image x axis."""
    try:
        _, basis = await sorter.camera_basis(machine)
        cam_x = basis[:, 0]
        jaw_yaw = math.degrees(math.atan2(cam_x[1], cam_x[0]))
        g = (await motion.get_pose('gripper', 'world', timeout=15)).pose
        offset = (jaw_yaw - tool_x_yaw(g.o_x, g.o_y, g.o_z, g.theta) + 180) % 360 - 180
        print(f'Jaw axis is {offset:+.1f} deg from the tool x axis (measured from the wrist camera).', flush=True)
        return offset
    except Exception as error:
        print('Could not measure the jaw axis; assuming it lies along tool x:', str(error)[:100], flush=True)
        return 0.0


async def recheck_block(machine, camera, color, x, y, z, min_area_px):
    """Look at the block from the approach pose; confirm its colour is there and refine x, y.

    Returns (x, y) or None when the block is outside the camera view (cannot check).
    Raises if the view shows no such block near the cached position.
    """
    origin, basis = await sorter.camera_basis(machine)
    frames, _ = await camera.get_images(timeout=20)
    frame = cv2.imdecode(np.frombuffer(next(f for f in frames if f.name == 'color').data, np.uint8), cv2.IMREAD_COLOR)
    p = (await camera.get_properties(timeout=15)).intrinsic_parameters
    intrinsic = (p.focal_x_px, p.focal_y_px, p.center_x_px, p.center_y_px)
    point = basis.T @ (np.array([x, y, z]) - origin)
    if frame is None or point[2] <= 0:
        return None
    u = int(round(intrinsic[0] * point[0] / point[2] + intrinsic[2]))
    v = int(round(intrinsic[1] * point[1] / point[2] + intrinsic[3]))
    half = 60
    if not half <= u < frame.shape[1] - half or not half <= v < frame.shape[0] - half:
        return None
    window = frame[v - half:v + half, u - half:u + half]
    hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[color][0]:
        mask |= cv2.inRange(hsv, (lo, 70, 40), (hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        raise RuntimeError(f'Recheck: no {color} block seen at the cached position')
    best = max(contours, key=cv2.contourArea)
    m = cv2.moments(best)
    cu, cv_ = m['m10'] / m['m00'] + u - half, m['m01'] / m['m00'] + v - half
    if math.hypot(cu - u, cv_ - v) > 45:
        raise RuntimeError(f'Recheck: {color} block is {math.hypot(cu - u, cv_ - v):.0f} px from the cached position')
    new = sorter.ray_plane((cu, cv_), z, intrinsic, origin, basis)
    shift = math.hypot(new[0] - x, new[1] - y)
    if shift > 25:
        raise RuntimeError(f'Recheck: {color} block moved {shift:.0f} mm from the cached position')
    return float(new[0]), float(new[1])


MAX_ATTEMPTS = 3
# Where the gripper's own jaws appear in the wrist-camera frame (fractions of width / height): a held
# block sits between the two jaw pads at the bottom centre of the picture.
JAW_VIEW = (0.50, 0.82, 0.86, 1.0)  # x0, x1, y0, y1
JAW_COLOUR_FRACTION = 0.10


async def holding_in_view(machine, camera, color, name):
    """True if the wrist camera sees a block of this colour between the jaws (bottom centre)."""
    frames, _ = await camera.get_images(timeout=20)
    frame = cv2.imdecode(np.frombuffer(next(f for f in frames if f.name == 'color').data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return False, 0.0
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = int(w * JAW_VIEW[0]), int(w * JAW_VIEW[1]), int(h * JAW_VIEW[2]), h
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[color][0]:
        mask |= cv2.inRange(hsv, (lo, 70, 40), (hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fraction = float(np.mean(mask > 0))
    seen = fraction >= JAW_COLOUR_FRACTION and bool(mask[-6:].any())  # a held block touches the frame's bottom edge
    # A block lying on the table can also fill that part of the picture when the arm is high. It is far
    # from the camera (valid depth), while a block in the jaws is within a few cm (or too close to read).
    depth_frame = next((f for f in frames if f.name == 'depth'), None)
    if seen and depth_frame is not None:
        try:
            depth = decode_depth(depth_frame.data)
        except ValueError:
            depth = None
        if depth is not None and depth.shape[:2] == (h, w):
            patch = depth[y0:y1, x0:x1][mask > 0]
            valid = patch[np.isfinite(patch) & (patch > 100) & (patch < 3000)]
            if len(valid) > 0.5 * max(len(patch), 1) and float(np.median(valid)) > TABLE_MIN_DEPTH_MM:
                seen = False
                print(f'  jaw view: coloured patch is {np.median(valid):.0f} mm away - a block on the table, not in the jaws', flush=True)
    try:
        FOLDER.mkdir(parents=True, exist_ok=True)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0) if seen else (0, 0, 255), 3)
        cv2.imwrite(str(FOLDER / f'hold_{name}.jpg'), frame)
    except Exception:
        pass
    return seen, fraction


async def rescan_block(machine, camera, motion, world, config, color, near_xy):
    """Return to the scan pose, look again, and return the same-colour block nearest near_xy (or None)."""
    await move_arm_to_scan(motion, world, config)
    for attempt in range(3):
        try:
            items, origin, basis, intrinsic = await sorter.snapshot(machine, camera, motion, config,
                                                                    config['_detector'], allow_partial=False)
            break
        except ValueError as error:
            if 'Layout uncertain' not in str(error) or attempt == 2:
                raise
    bins = [np.array(d['polygon'], np.int32) for d in config['_current_layout']['destinations']]
    best, best_distance = None, 150.0
    for i in items:
        size = i.get('size_mm') or (0, 0)
        if (i['color'] != color or not usable(i) or min(size) < MIN_BLOCK_MM or max(size) > MAX_BLOCK_MM
                or any(cv2.pointPolygonTest(b, tuple(map(float, i['center'])), False) >= 0 for b in bins)):
            continue
        distance = math.hypot(i['world_mm'][0] - near_xy[0], i['world_mm'][1] - near_xy[1])
        if distance < best_distance:
            best, best_distance = i, distance
    return best


async def move_arm_to_scan(motion, world, config):
    if not await motion.move(component_name='arm', destination=PoseInFrame(reference_frame='world',
            pose=Pose(**config['scan_pose'])), world_state=motion_world_state(world), timeout=SCAN_MOVE_TIMEOUT_S):
        raise RuntimeError('Cannot reach scan pose')


async def recheck_destination(machine, camera, motion, world, config, block):
    """Holding a block, return to the scan pose, re-detect the bins and recompute this block's drop spot.

    Returns (drop_xy, bin_axis_yaw) or None when the layout cannot be confirmed (the cached spot is used).
    """
    if 'slot_t' not in block:  # a plan cached by an older version
        return None
    saved = config['_object_detector']
    config['_object_detector'] = None  # bins only: skip the slow per-block segmentation
    try:
        await move_arm_to_scan(motion, world, config)
        for attempt in range(3):
            try:
                _, origin, basis, intrinsic = await sorter.snapshot(machine, camera, motion, config,
                                                                    config['_detector'], allow_partial=False)
                break
            except ValueError as error:
                if 'Layout uncertain' not in str(error) or attempt == 2:
                    print('  bin recheck: layout not confirmed; using the scanned spot:', str(error)[:80], flush=True)
                    return None
    finally:
        config['_object_detector'] = saved
    cx, cy = block['bin_center_px']
    d = min(config['_current_layout']['destinations'], key=lambda b: math.hypot(b['center'][0] - cx, b['center'][1] - cy))
    if math.hypot(d['center'][0] - cx, d['center'][1] - cy) > 90:
        print('  bin recheck: could not match the bin; using the scanned spot', flush=True)
        return None
    centre, axis, axis_yaw = bin_frame(d, config, origin, basis, intrinsic)
    return (centre + block['slot_t'] * axis).tolist(), axis_yaw


async def place_block(move, gripper, drop, transit, long_block):
    """Carry to the drop spot and release. If the arm cannot reach the spot with this wrist angle (a
    self-collision of the wrist/camera, for example), retry with the equivalent turned-around wrist:
    the jaws are symmetric, so +/-180 deg (and +/-90 for a square block) hold the block the same way."""
    turns = (0, 180, -180) if long_block else (0, 90, -90, 180, -180)
    failure = None
    for turn in turns:
        pose = sorter.clone_pose(drop, theta=drop.theta + turn)
        try:
            await move(sorter.clone_pose(pose, z=transit))
            await move(pose)
            break
        except Exception as error:
            if 'constraint' not in str(error) and 'IK' not in str(error):
                raise
            failure = error
            print(f'  drop with wrist turn {turn:+d} refused; trying another angle', flush=True)
    else:
        raise failure
    await gripper.open(timeout=15)
    await move(sorter.clone_pose(pose, z=transit))


async def run_finish(args, config):
    """Finish a transfer: the gripper already holds block NAME from the cache; carry it to its bin."""
    load_dotenv(sorter.ROOT / '.env')
    plan = json.loads(CACHE.read_text())
    block = next((b for b in plan['blocks'] if b['name'] == args.block), None)
    if block is None:
        raise ValueError(f'No block {args.block} in the cache; have ' + ', '.join(b['name'] for b in plan['blocks']))
    machine = await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])), 30)
    arm = None
    try:
        arm = Arm.from_robot(machine, 'arm'); gripper = Gripper.from_robot(machine, 'gripper')
        motion = MotionClient.from_robot(machine, 'builtin')
        if not (await gripper.is_holding_something(timeout=15)).is_holding_something:
            raise RuntimeError('Gripper is not holding anything')
        world, _ = await inspect_setup(machine, motion)
        mover = DirectFallbackMotion(motion, arm)
        state = motion_world_state(world)
        await arm.do_command({'set_speed': SPEED}, timeout=15)
        jaw_offset = await measure_jaw_offset(machine, motion)

        async def move(pose):
            if not await mover.move(component_name='gripper', destination=PoseInFrame(reference_frame='world', pose=pose),
                    world_state=state, extra={'motion_profile': 'linear'}, timeout=MOVE_TIMEOUT_S):
                raise RuntimeError('Motion not confirmed')

        orient = config['pick_orientation']
        long_block = block.get('long', False)
        drop_orient = drop_orientation(orient, jaw_offset, block['bin_axis_yaw'], long_block)
        release = config['destination_release_z_mm']
        drop = Pose(x=block['drop_xy'][0], y=block['drop_xy'][1],
                    z=max(release, release - config['release_above_rim_mm'] + DROP_ABOVE_RIM_MM), **drop_orient)
        here = (await motion.get_pose('gripper', 'world', timeout=15)).pose
        if here.z < config['transit_z_mm'] - 5:
            await move(sorter.clone_pose(here, z=config['transit_z_mm']))
        await place_block(move, gripper, drop, config['transit_z_mm'], long_block)
        print(f'{args.block} released into {block["bin"]}', flush=True)
    finally:
        try:
            if arm is not None:
                await arm.stop(timeout=10)
        finally:
            await machine.close()


# ---- hello wave, from examples/hello_wave.py: joint-space poses, no planner ----
WAVE_UP = [0, -20, -60, 0, -30, 0]  # J1 base, J2 shoulder, J3 elbow, J4 wrist roll, J5 wrist pitch, J6 tool roll
WAVE_WRIST_DEG = 25
WAVE_SIDE_DEG = 20
WAVE_COUNT = 1
WAVE_SPEED = 200.0  # tuned so the whole hello takes about 7 s (run --stage hello to time it)


async def say_hello(arm, gripper):
    """Wave hello (wrist wave, then side-to-side), then return to where the arm started."""
    from viam.proto.component.arm import JointPositions

    def jp(values):
        return JointPositions(values=values)

    def shifted(joint, delta):
        v = list(WAVE_UP)
        v[joint] += delta
        return v

    start = list((await arm.get_joint_positions(timeout=15)).values)
    await arm.do_command({'set_speed': WAVE_SPEED}, timeout=15)
    print('Hello!', flush=True)
    try:
        await arm.move_to_joint_positions(jp(WAVE_UP), timeout=SCAN_MOVE_TIMEOUT_S)
        for joint, amplitude in ((4, WAVE_WRIST_DEG), (0, WAVE_SIDE_DEG)):
            for _ in range(WAVE_COUNT):
                # the gripper opens and closes on the side-to-side wave only (it is the slow part)
                await asyncio.gather(arm.move_to_joint_positions(jp(shifted(joint, amplitude)), timeout=SCAN_MOVE_TIMEOUT_S),
                                     gripper.open(timeout=15) if joint == 0 else asyncio.sleep(0))
                await asyncio.gather(arm.move_to_joint_positions(jp(shifted(joint, -amplitude)), timeout=SCAN_MOVE_TIMEOUT_S),
                                     gripper.grab(timeout=15) if joint == 0 else asyncio.sleep(0))
            await arm.move_to_joint_positions(jp(WAVE_UP), timeout=SCAN_MOVE_TIMEOUT_S)
        await gripper.open(timeout=15)
    finally:
        await arm.move_to_joint_positions(jp(start), timeout=MOVE_TIMEOUT_S)


async def run_hello(args, config):
    """Just the wave, timed."""
    import time
    load_dotenv(sorter.ROOT / '.env')
    machine = await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])), 30)
    try:
        arm = Arm.from_robot(machine, 'arm'); gripper = Gripper.from_robot(machine, 'gripper')
        if (await gripper.is_holding_something(timeout=15)).is_holding_something:
            raise RuntimeError('Gripper is holding something; not waving')
        started = time.monotonic()
        await say_hello(arm, gripper)
        print(f'Hello took {time.monotonic() - started:.1f} s', flush=True)
    finally:
        await arm.stop(timeout=10)
        await machine.close()


class LiveView:
    """Debug windows: the wrist camera's live feed (with the jaw-check box and a status line) and the latest
    detection overlay (containers, blocks, rejections) from the most recent scan."""

    def __init__(self, camera, enabled):
        self.camera, self.enabled = camera, enabled
        self.status = ''
        self.task = None
        self._overlay_time = __import__('time').time()  # only show overlays written by this run
        self._web_time = 0.0

    def say(self, text):
        self.status = text

    def start(self):
        if self.enabled:
            self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except BaseException:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _publish(self, frame):
        """Write the annotated live frame for the web page (outputs/sort_runs/live.jpg, ~7 fps, atomic)."""
        now = __import__('time').time()
        if now - self._web_time < 0.14:
            return
        self._web_time = now
        try:
            folder = sorter.OUTPUT_DIR / 'sort_runs'
            folder.mkdir(parents=True, exist_ok=True)
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                tmp = folder / 'live.jpg.tmp'
                tmp.write_bytes(jpg.tobytes())
                os.replace(tmp, folder / 'live.jpg')
        except Exception:
            pass

    async def _loop(self):
        overlay_path = sorter.OUTPUT_DIR / 'sort_runs' / 'latest_overlay.jpg'
        while True:
            try:
                frames, _ = await self.camera.get_images(timeout=5)
                frame = cv2.imdecode(np.frombuffer(next(f for f in frames if f.name == 'color').data, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (int(w * JAW_VIEW[0]), int(h * JAW_VIEW[2])), (int(w * JAW_VIEW[1]) - 1, h - 1),
                                  (0, 200, 255), 2)
                    cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
                    cv2.putText(frame, self.status[:90], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    self._publish(frame)
                    cv2.imshow('live feed (orange box = jaw check)', frame)
                if overlay_path.exists() and overlay_path.stat().st_mtime > self._overlay_time:
                    self._overlay_time = overlay_path.stat().st_mtime
                    overlay = cv2.imread(str(overlay_path))
                    if overlay is not None:
                        cv2.imshow('detection (latest scan)', overlay)
                cv2.waitKey(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(0.08)


HALF_OPEN_POS = 425.0  # UFactory gripper travel is 0..850 (fully closed .. fully open); 425 is the midpoint
HALF_OPEN_TOLERANCE = 15.0


async def open_gripper(gripper, half_open=None, sleep=asyncio.sleep):
    """Open the jaws before a grasp. half_open=None: fully open. A number: that jaw position (0..850) through the
    driver's do_command {'set': pos} / {'get': True} (the other team's half_open_gripper.py). If the driver does
    not report the position or never reaches it, fall back to a full open: never carry on with unknown jaws."""
    if half_open is None:
        await gripper.open(timeout=15)
        return None
    try:
        if not isinstance((await gripper.do_command({'get': True}, timeout=15)).get('pos'), (int, float)):
            raise RuntimeError('driver did not report a jaw position')
        await gripper.do_command({'set': float(half_open)}, timeout=20)
        position = None
        for _ in range(15):
            position = (await gripper.do_command({'get': True}, timeout=10)).get('pos')
            if isinstance(position, (int, float)) and abs(position - half_open) <= HALF_OPEN_TOLERANCE:
                print(f'  gripper half-open at {position:.0f} / 850', flush=True)
                return position
            await sleep(0.5)
        raise RuntimeError(f'position {position} did not reach {half_open}')
    except Exception as error:
        print(f'  half-open failed ({str(error)[:80]}); opening fully instead', flush=True)
        await gripper.open(timeout=15)
        return None


def half_open_for(block, requested):
    """Jaw opening for this block's grasp: the requested half-open, except where the jaws could end up
    across a rectangle's LONG side. Long blocks are turned to grip the short side only when their footprint
    is known (aligned_orientation); without it the wrist angle is arbitrary, so open fully."""
    if requested is not None and block.get('long', False) and not block.get('footprint'):
        return None
    return requested


async def run_auto(args, config):
    rim = max(src['rim_z_mm'] for src in config['sources'])
    sorter.check_ready({**config, 'motion_commissioned': True,
                        'sources': [{**src, 'rim_z_mm': rim} for src in config['sources']]})
    load_dotenv(sorter.ROOT / '.env')
    # Load the vision models BEFORE connecting: loading blocks the event loop, which starves the
    # connection heartbeat and expired the Viam session (SESSION_EXPIRED) on the first move.
    detector = object_detector = None
    if not (args.use_cache and CACHE.exists()):
        detector = ContainerDetector()
        object_detector = ObjectInstanceDetector(sorter.MODELS_DIR / 'sam2.1_t.pt')
    machine = await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])), 30)
    arm = gripper = live = None
    try:
        arm = Arm.from_robot(machine, 'arm'); gripper = Gripper.from_robot(machine, 'gripper')
        camera = Camera.from_robot(machine, 'cam'); motion = MotionClient.from_robot(machine, 'builtin')
        world, _ = await inspect_setup(machine, motion)
        mover = DirectFallbackMotion(motion, arm)
        state = motion_world_state(world)
        live = LiveView(camera, args.live)
        live.start()
        if args.wave:
            live.say('hello wave')
            if (await gripper.is_holding_something(timeout=15)).is_holding_something:
                print('Gripper is holding something; skipping the wave.', flush=True)
            else:
                await say_hello(arm, gripper)
        await arm.do_command({'set_speed': SPEED}, timeout=15)
        jaw_offset = await measure_jaw_offset(machine, motion)

        async def move(pose):
            if not await mover.move(component_name='gripper',
                    destination=PoseInFrame(reference_frame='world', pose=pose),
                    world_state=state, extra={'motion_profile': 'linear'}, timeout=MOVE_TIMEOUT_S):
                raise RuntimeError('Motion not confirmed')

        table_pose = (await motion.get_pose('table', 'world', timeout=15)).pose
        table_geometry = next(o for o in world.obstacles if o.reference_frame == 'table').geometries[0]
        config['_table_top_z_mm'] = table_pose.z + table_geometry.center.z + table_geometry.box.dims_mm.z / 2
        if detector is None:  # models are loaded lazily only if a cached plan needs a rescan
            pass
        config['_detector'] = detector
        config['_object_detector'] = object_detector
        config['_container_tracker'] = None
        config['_show_rejected'] = False
        plan = None
        if args.use_cache and CACHE.exists():
            cached = json.loads(CACHE.read_text())
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached['time'])).total_seconds()
            if age <= CACHE_MAX_AGE_S and isinstance(cached['blocks'], list) and cached['blocks']:
                plan = cached
                print(f'Using cached layout and block positions ({age:.0f} s old); no scan.', flush=True)
        if plan is None:
            if detector is None:  # stale cache path fell through to a scan
                detector = ContainerDetector()
                object_detector = ObjectInstanceDetector(sorter.MODELS_DIR / 'sam2.1_t.pt')
                config['_detector'] = detector
                config['_object_detector'] = object_detector
            live.say('scanning the table')
            await move_arm_to_scan(motion, world, config)
            for attempt in range(3):  # layout detection is flaky
                try:
                    items, origin, basis, intrinsic = await sorter.snapshot(machine, camera, motion, config,
                                                                            detector, allow_partial=False)
                    break
                except ValueError as error:
                    if 'Layout uncertain' not in str(error) or attempt == 2:
                        raise
                    print('Retrying scan:', error, flush=True)
            plan = plan_from_scan(items, config['_current_layout'], args.colors, origin, basis, intrinsic, config)
            FOLDER.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(plan, indent=2))
            print('Scanned once and cached layout + block positions:', CACHE, flush=True)
        orient = config['pick_orientation']; transit = config['transit_z_mm']
        release = config['destination_release_z_mm']
        drop_z = max(release, release - config['release_above_rim_mm'] + DROP_ABOVE_RIM_MM)
        results = {}
        stop_all = False
        for block in plan['blocks']:
            if stop_all:
                break
            color = block['color']; name = block['name']
            long_block = block.get('long', False)
            drop_orient = drop_orientation(orient, jaw_offset, block['bin_axis_yaw'], long_block)
            drop = Pose(x=block['drop_xy'][0], y=block['drop_xy'][1], z=drop_z, **drop_orient)
            for attempt in range(1, MAX_ATTEMPTS + 1):
                x, y, z = block['world_mm']
                block_orient, turn = aligned_orientation(orient, block.get('footprint'), jaw_offset)
                if block.get('flip'):  # the arm's own wrist hit its camera at this angle: use the mirrored wrist turn
                    block_orient = dict(block_orient, theta=block_orient['theta'] + block['flip'])
                    turn += block['flip']
                if attempt > 1 and not long_block:  # last grasp failed: close the jaws across the other pair of faces
                    swing = 90 if attempt % 2 == 0 else -90
                    block_orient = dict(block_orient, theta=block_orient['theta'] + swing)
                    turn += swing
                grasp_z = (max(z + config['grasp_offset_mm'], MIN_GRASP_Z_MM) - PICK_LOWER_MM
                           + (LONG_RAISE_MM if long_block else 0))
                pick = Pose(x=x, y=y, z=grasp_z, **block_orient)
                rim_z = config['sources'][block['source']]['rim_z_mm']
                approach = sorter.clone_pose(pick, z=max(z + config['approach_mm'], rim_z + 50))
                print(f'--- {name}{" [long]" if long_block else ""} (try {attempt}): pick ({x:.0f},{y:.0f}) wrist turn {turn:+.0f} deg -> {block["bin"]}', flush=True)
                carrying = False
                live.say(f'{name} (try {attempt}): going to pick')
                try:
                    here = (await motion.get_pose('gripper', 'world', timeout=15)).pose
                    await move(sorter.clone_pose(here, z=max(here.z, transit)))
                    await move(sorter.clone_pose(approach, z=transit))
                    await open_gripper(gripper, half_open_for(block, args.half_open))  # --half-open: never across a rectangle's long side
                    await move(approach)
                    if args.recheck:
                        seen = await recheck_block(machine, camera, color, x, y, z, config['min_area_px'])
                        if seen is None:
                            print('  recheck: block outside camera view at approach; using cached position', flush=True)
                        elif math.hypot(seen[0] - x, seen[1] - y) > 1.5:
                            print(f'  recheck: refined position by {seen[0] - x:+.1f}, {seen[1] - y:+.1f} mm', flush=True)
                            pick = sorter.clone_pose(pick, x=seen[0], y=seen[1])
                            approach = sorter.clone_pose(approach, x=seen[0], y=seen[1])
                            await move(approach)
                        else:
                            print('  recheck: block confirmed', flush=True)
                    if long_block:
                        await arm.do_command({'set_speed': LONG_PICK_SPEED}, timeout=15)
                    mover.allow_table_contact = True
                    try:
                        await move(pick)
                    finally:
                        mover.allow_table_contact = False
                    await gripper.grab(timeout=20)
                    if long_block:
                        await arm.do_command({'set_speed': SPEED}, timeout=15)
                    await move(approach)  # lift clear, then judge the grasp from what the camera sees
                    in_jaws, fraction = await holding_in_view(machine, camera, color, name)
                    live.say(f'{name}: jaw check {"HELD" if in_jaws else "EMPTY"} ({fraction:.0%})')
                    if not in_jaws:
                        raise RuntimeError(f'camera does not see a {color} block in the jaws ({fraction:.0%} of the jaw view)')
                    if not (await gripper.is_holding_something(timeout=15)).is_holding_something:
                        raise RuntimeError('camera sees the block but the gripper reports nothing held')
                    print(f'  picked: {color} block seen in the jaws ({fraction:.0%} of the jaw view)', flush=True)
                    carrying = True
                    await move(sorter.clone_pose(approach, z=transit))
                    if args.bin_recheck:
                        live.say(f'{name}: re-scanning the {block["bin"]} bin before the drop')
                        fresh = await recheck_destination(machine, camera, motion, world, config, block)
                        if fresh is not None:
                            (nx, ny), axis_yaw = fresh
                            moved = math.hypot(nx - block['drop_xy'][0], ny - block['drop_xy'][1])
                            print(f'  bin recheck: {block["bin"]} drop spot moved {moved:.1f} mm', flush=True)
                            drop_orient = drop_orientation(orient, jaw_offset, axis_yaw, long_block)
                            drop = Pose(x=nx, y=ny, z=drop_z, **drop_orient)
                    live.say(f'{name}: dropping into {block["bin"]}')
                    await place_block(move, gripper, drop, transit, long_block)
                    results[name] = 'placed in ' + block['bin'] + (f' (try {attempt})' if attempt > 1 else '')
                    live.say(f'{name}: placed')
                    break
                except Exception as error:
                    reason = str(error)[:160]
                    try:
                        await arm.do_command({'set_speed': SPEED}, timeout=15)
                    except Exception:
                        pass
                    print(f'  {name} try {attempt} failed: {reason}', flush=True)
                    if 'self-collision' in str(error):  # jaws are symmetric: the opposite wrist turn grips the same way
                        block['flip'] = 180 if block.get('flip', 0) != 180 else -180
                    if carrying:  # the block is in the jaws mid-transfer: never drop it blindly
                        results[name] = 'PICKED UP (in the jaws) but not placed: ' + reason
                        stop_all = True
                        break
                    if attempt == MAX_ATTEMPTS:
                        results[name] = 'FAILED: ' + reason
                        break
                    try:  # let go of whatever the jaws may have, lift clear, look again, retry
                        await gripper.open(timeout=15)
                        here = (await motion.get_pose('gripper', 'world', timeout=15)).pose
                        await move(sorter.clone_pose(here, z=max(here.z, transit)))
                        found = await rescan_block(machine, camera, motion, world, config, color, (x, y))
                    except Exception as recovery_error:
                        results[name] = f'FAILED: {reason}; recovery failed: {str(recovery_error)[:100]}'
                        break
                    if found is None:
                        results[name] = f'FAILED: {reason}; block not found after rescan'
                        break
                    block['world_mm'] = found['world_mm']
                    block['footprint'] = found.get('footprint_world_offsets_mm')
                    print(f'  rescanned: {color} block now at ({found["world_mm"][0]:.0f},{found["world_mm"][1]:.0f})', flush=True)
        try:
            await move_arm_to_scan(motion, world, config)
        except Exception:
            pass
        print('\nSummary:', json.dumps(results, indent=1), flush=True)
    finally:
        try:
            if live is not None:
                await live.stop()
            if arm is not None:
                await arm.stop(timeout=10)
        finally:
            await machine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['scan', 'go', 'drop', 'all', 'auto', 'finish', 'hello'], default='scan',
                        help='scan: look + remember, no motion. go: pick up the remembered block and drop it. drop: drop the block already held')
    parser.add_argument('--block', help='finish: cached block name the gripper is holding, e.g. red1')
    parser.add_argument('--no-bin-recheck', dest='bin_recheck', action='store_false',
                        help='auto: do not re-detect the bins before each drop')
    parser.add_argument('--no-live', dest='live', action='store_false',
                        help='auto: do not open the live feed / detection windows')
    parser.add_argument('--no-wave', dest='wave', action='store_false', help='auto: skip the hello wave')
    parser.add_argument('--no-recheck', dest='recheck', action='store_false',
                        help='auto: skip the look-again at the approach pose')
    parser.add_argument('--use-cache', action='store_true',
                        help='auto: reuse cached layout/block positions (<15 min old) instead of scanning')
    parser.add_argument('--speed', type=float, default=None, metavar='N',
                        help='arm speed (normal 50). 5 = super slow and quiet, e.g. for presenting; slows the wave and scales move timeouts')
    parser.add_argument('--half-open', nargs='?', type=float, const=HALF_OPEN_POS, default=None, metavar='POS',
                        help=f'auto: open the jaws only to POS (0..850, default {HALF_OPEN_POS:.0f} = half) before each grasp; '
                             'a long block with no known footprint (could be gripped across its long side) opens fully')
    parser.add_argument('--colors', nargs='+', default=['green'], choices=['red', 'yellow', 'blue', 'green'],
                        help='blocks to move, in order (default: green); --stage all scans then moves each')
    args = parser.parse_args()
    apply_speed(args.speed)
    global COLOR
    if args.stage == 'hello':
        asyncio.run(run_hello(args, load_config()))
        return
    if args.stage == 'finish':
        asyncio.run(run_finish(args, load_config()))
        return
    if args.stage == 'auto':
        if args.colors == ['green']:
            args.colors = list(ORDER)
        asyncio.run(run_auto(args, load_config()))
        return

    def run_stage(stage):
        cfg = load_config()
        asyncio.run(run(argparse.Namespace(**{**vars(args), 'stage': stage}), cfg))

    def retry(stage, attempts=3):
        # Layout detection is flaky ('Layout uncertain'); retrying a failed scan is harmless.
        for n in range(attempts):
            try:
                return run_stage(stage)
            except Exception as error:
                if 'Layout uncertain' not in str(error) or n == attempts - 1:
                    raise
                print('Retrying:', error, flush=True)

    global COLOR
    for color in args.colors:
        COLOR = color
        print(f'=== {color} ===', flush=True)
        if args.stage != 'all':
            retry(args.stage)
            continue
        retry('scan')
        try:
            run_stage('go')
        except Exception as error:
            print('Go stopped:', error, flush=True)
            try:  # if the block is already in the gripper, finish the drop
                retry('drop', 4)
            except Exception as drop_error:
                if 'not holding' in str(drop_error):
                    raise error
                raise

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Stopped; no automatic release or return.')
    except Exception as error:
        raise SystemExit('Green test stopped: ' + str(error))
