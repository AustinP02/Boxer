import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import math
import unittest
import cv2
import numpy as np
import color_sorter as sorter
import green_to_container as g
import pill_to_container as p

# Camera 600 mm above world z=0 looking straight down; image x -> world x, image y -> world -y.
INTRINSIC = (900.0, 900.0, 640.0, 360.0)
ORIGIN = np.array([100.0, 200.0, 600.0])
BASIS = np.column_stack(([1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]))  # camera x, y, z axes in world
TABLE_Z = -23.0
W, H = 1280, 720


def scene(background=(120, 120, 120), table_white=False):
    frame = np.full((H, W, 3), 255 if table_white else background, np.uint8)
    depth = np.full((H, W), ORIGIN[2] - TABLE_Z, np.float32)
    return frame, depth


def world_px(x, y, z):
    return p.world_to_pixel([x, y, z], INTRINSIC, ORIGIN, BASIS)


def draw_disc(frame, depth, x, y, top_z, radius_mm):
    u, v = world_px(x, y, top_z)
    r = int(round(radius_mm * INTRINSIC[0] / (ORIGIN[2] - top_z)))
    cv2.circle(frame, (int(u), int(v)), r, (250, 250, 250), -1)
    cv2.circle(depth, (int(u), int(v)), r, float(ORIGIN[2] - top_z), -1)


def heights_of(depth):
    return p.height_map(depth, INTRINSIC, ORIGIN, BASIS, TABLE_Z)


class ProjectionTests(unittest.TestCase):
    def test_pixel_round_trip(self):
        for xyz in ((100, 200, 30), (250, 120, -23), (-40, 310, 80)):
            u, v = p.world_to_pixel(xyz, INTRINSIC, ORIGIN, BASIS)
            back = sorter.ray_plane((u, v), xyz[2], INTRINSIC, ORIGIN, BASIS)
            np.testing.assert_allclose(back, xyz, atol=1e-6)

    def test_container_at_rim_not_floor(self):
        u, v = p.world_to_pixel([160, 150, 47], INTRINSIC, ORIGIN, BASIS)
        at_rim = p.container_from_pixel(u, v, 47, INTRINSIC, ORIGIN, BASIS)
        at_floor = p.container_from_pixel(u, v, -23, INTRINSIC, ORIGIN, BASIS)
        np.testing.assert_allclose(at_rim, [160, 150], atol=1e-6)
        self.assertGreater(np.linalg.norm(at_rim - at_floor), 5)  # projecting at the floor would shift it

    def test_container_polygon_centre_and_extent(self):
        corners = [(100 - 75, 200 - 55), (100 + 75, 200 - 55), (100 + 75, 200 + 55), (100 - 75, 200 + 55)]
        polygon = [world_px(x, y, 47) for x, y in corners]
        centre, extent = p.container_from_polygon(polygon, 47, INTRINSIC, ORIGIN, BASIS)
        np.testing.assert_allclose(centre, [100, 200], atol=1)
        self.assertAlmostEqual(extent[0], 110, delta=2)
        self.assertAlmostEqual(extent[1], 150, delta=2)

    def test_pick_big_container_ignores_bins(self):
        def box(cx, cy, w, h):
            return dict(roi=[0, 0, 0, 0], confidence=.9, polygon=[world_px(cx + dx, cy + dy, 47) for dx, dy in
                        ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))])
        bin_ = box(0, 200, 138, 52)
        big = box(120, 220, 160, 120)
        best, seen = p.pick_big_container([bin_, big], 47, INTRINSIC, ORIGIN, BASIS, 80)
        self.assertEqual(len(seen), 2)
        np.testing.assert_allclose(best['centre'], [120, 220], atol=1)
        self.assertIsNone(p.pick_big_container([bin_], 47, INTRINSIC, ORIGIN, BASIS, 80)[0])


class TableMeasureTests(unittest.TestCase):
    def test_wrong_model_table_height_is_corrected(self):
        """Real table at z=+3 while the model says -23: the white table must not look 26 mm tall."""
        real = 3.0
        frame = np.full((H, W, 3), 235, np.uint8)  # white table
        depth = np.full((H, W), ORIGIN[2] - real, np.float32)
        draw_disc(frame, depth, 130, 180, real + 35, 17)  # 35 mm tall, 34 mm wide bottle cap
        table_z, heights, offset = p.measure_table(depth, INTRINSIC, ORIGIN, BASIS, TABLE_Z)
        self.assertAlmostEqual(offset, 26, delta=1.5)
        self.assertAlmostEqual(table_z, real, delta=1.5)
        cands = p.find_pill(frame, heights, INTRINSIC, ORIGIN, BASIS, table_z)
        self.assertEqual(len(p.eligible(cands)), 1, [c['rejected'] for c in cands])
        c = p.eligible(cands)[0]
        self.assertAlmostEqual(c['height_mm'], 35, delta=2)
        self.assertAlmostEqual(c['top_z'], real + 35, delta=2)

    def test_too_little_depth_leaves_model_height(self):
        self.assertEqual(p.estimate_table_offset(np.full((20, 20), np.nan)), 0.0)


class WhiteDetectionTests(unittest.TestCase):
    def test_white_mask_thresholds(self):
        img = np.zeros((2, 3, 3), np.uint8)
        img[0, 0] = (250, 250, 250)  # white
        img[0, 1] = (40, 40, 40)  # dark
        img[0, 2] = (60, 60, 250)  # saturated red
        img[1, 0] = (130, 135, 132)  # dim white, as in the real wrist frame
        m = p.white_mask(img)
        self.assertEqual([bool(m[0, 0]), bool(m[0, 1]), bool(m[0, 2]), bool(m[1, 0])], [True, False, False, True])

    def test_standing_pill_found(self):
        frame, depth = scene()
        draw_disc(frame, depth, 130, 180, 60, 25)  # 50 mm cap, top 83 mm above the table
        cands = p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z)
        good = p.eligible(cands)
        self.assertEqual(len(good), 1, [c['rejected'] for c in cands])
        c = good[0]
        self.assertEqual(c['shape'], 'standing')
        self.assertAlmostEqual(c['world_xy'][0], 130, delta=1.5)
        self.assertAlmostEqual(c['world_xy'][1], 180, delta=1.5)
        self.assertAlmostEqual(c['top_z'], 60, delta=2)
        self.assertAlmostEqual(c['height_mm'], 83, delta=2)
        self.assertAlmostEqual(c['size_mm'][0], 50, delta=3)

    def test_white_table_is_not_a_pill(self):
        frame, depth = scene(table_white=True)
        self.assertEqual(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z), [])

    def test_pill_on_white_table_still_found(self):
        frame, depth = scene(table_white=True)
        draw_disc(frame, depth, 130, 180, 40, 25)
        cands = p.eligible(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z))
        self.assertEqual(len(cands), 1)

    def test_big_raised_white_wall_rejected(self):
        frame, depth = scene()
        draw_disc(frame, depth, 100, 200, 40, 90)  # 180 mm white thing, e.g. a container
        cands = p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z)
        self.assertTrue(cands)
        self.assertFalse(p.eligible(cands))

    def test_thin_white_ring_rejected(self):
        frame, depth = scene()
        u, v = world_px(100, 200, 40)
        r = int(30 * 900 / 640)
        cv2.circle(frame, (int(u), int(v)), r, (250, 250, 250), 4)
        cv2.circle(depth, (int(u), int(v)), r, float(ORIGIN[2] - 40), 4)
        self.assertFalse(p.eligible(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z)))

    def test_flat_white_paper_rejected(self):
        frame, depth = scene()
        draw_disc(frame, depth, 130, 180, -20, 25)  # 3 mm above the table
        self.assertEqual(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z), [])

    def test_too_wide_for_jaws_rejected(self):
        frame, depth = scene()
        draw_disc(frame, depth, 130, 180, 60, 34)  # 68 mm
        cands = p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z)
        self.assertFalse(p.eligible(cands))

    def test_exclude_polygon_blanks_container(self):
        frame, depth = scene()
        draw_disc(frame, depth, 130, 180, 60, 25)
        u, v = world_px(130, 180, 60)
        box = [[u - 80, v - 80], [u + 80, v - 80], [u + 80, v + 80], [u - 80, v + 80]]
        self.assertEqual(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z, exclude=[box]), [])

    def test_missing_depth_on_cap_rejected(self):
        frame, depth = scene()
        draw_disc(frame, depth, 130, 180, 60, 25)
        h = heights_of(depth)
        h[:] = np.nan  # no depth anywhere: nothing counts as raised
        self.assertEqual(p.find_pill(frame, h, INTRINSIC, ORIGIN, BASIS, TABLE_Z), [])


class LyingPillTests(unittest.TestCase):
    AMBER = (40, 140, 230)  # BGR, hue ~16 in OpenCV

    def build(self, with_cap=True):
        """Bottle lying along world y at x=130: white cap y 170-190, amber body y 190-230, 35 mm across, top 35 mm."""
        frame, depth = scene()
        parts = [(112.5, 147.5, 190, 230, self.AMBER)]
        if with_cap:
            parts.append((112.5, 147.5, 170, 190, (250, 250, 250)))
        for x0, x1, y0, y1, colour in parts:
            a, b = world_px(x0, y1, 12), world_px(x1, y0, 12)  # top surface z = 12 (35 mm above the table)
            cv2.rectangle(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, -1)
            cv2.rectangle(depth, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), float(ORIGIN[2] - 12), -1)
        return frame, depth

    def test_lying_bottle_cap_plus_body(self):
        frame, depth = self.build()
        good = p.eligible(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z))
        self.assertEqual(len(good), 1)
        c = good[0]
        self.assertEqual(c['shape'], 'lying')
        self.assertAlmostEqual(c['size_mm'][0], 35, delta=3)  # diameter
        self.assertAlmostEqual(c['size_mm'][1], 60, delta=3)  # length: cap + body, not the cap alone
        self.assertAlmostEqual(abs(c['long_yaw'] - 90), 0, delta=3)  # long axis along world y
        x, y, note = p.grasp_point(c)
        self.assertAlmostEqual(x, 130, delta=2)
        self.assertAlmostEqual(y, 200, delta=2)  # middle of the whole bottle

    def test_amber_without_white_cap_is_ignored(self):
        frame, depth = self.build(with_cap=False)  # e.g. a yellow/red block
        self.assertEqual(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z), [])

    def test_white_glare_without_amber_body_is_rejected(self):
        frame, depth = self.build()
        a, b = world_px(112.5, 230, 12), world_px(147.5, 170, 12)  # whole bottle white, nothing amber
        cv2.rectangle(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (250, 250, 250), -1)
        cands = p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z)
        self.assertTrue(cands)
        self.assertFalse(p.eligible(cands))

    def test_block_with_a_white_patch_is_not_a_standing_pill(self):
        frame, depth = scene()
        u, v = world_px(130, 180, 12)
        r = int(18 * 900 / 600)
        cv2.rectangle(frame, (int(u - r), int(v - r)), (int(u + r), int(v + r)), self.AMBER, -1)  # 36 mm block
        cv2.rectangle(depth, (int(u - r), int(v - r)), (int(u + r), int(v + r)), float(ORIGIN[2] - 12), -1)
        cv2.rectangle(frame, (int(u - 8), int(v - 8)), (int(u + 8), int(v + 8)), (250, 250, 250), -1)  # small glare patch
        cands = p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z)
        self.assertFalse(p.eligible(cands))

    def test_block_next_to_pill_does_not_merge_when_apart(self):
        frame, depth = self.build()
        a, b = world_px(30, 190, 12), world_px(60, 160, 12)  # an amber block 50+ mm away
        cv2.rectangle(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), self.AMBER, -1)
        cv2.rectangle(depth, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), float(ORIGIN[2] - 12), -1)
        self.assertEqual(len(p.eligible(p.find_pill(frame, heights_of(depth), INTRINSIC, ORIGIN, BASIS, TABLE_Z))), 1)

    def test_lying_jaws_close_across_and_standing_keep_orientation(self):
        orient = dict(o_x=-0.0428, o_y=-0.0802, o_z=-0.9959, theta=62.06)
        lying = dict(shape='lying', long_yaw=37.0)
        turned = p.grasp_orientation(orient, 5.0, lying)
        jaw = g.tool_x_yaw(**turned) + 5.0
        self.assertAlmostEqual(((jaw - (37.0 + 90) + 90) % 180) - 90, 0, delta=0.5)  # jaws perpendicular to the long axis
        self.assertEqual(p.grasp_orientation(orient, 5.0, dict(shape='standing', long_yaw=None)), orient)
        self.assertAlmostEqual(p.grasp_orientation(orient, 0, dict(shape='standing'), 90)['theta'], orient['theta'] + 90)


class RetryTests(unittest.TestCase):
    def test_swings(self):
        standing, lying = dict(shape='standing'), dict(shape='lying')
        self.assertEqual([p.retry_swing(standing, n) for n in (1, 2, 3)], [0, 90, -90])
        self.assertEqual([p.retry_swing(lying, n) for n in (1, 2, 3)], [0, 180, -180])


class JawViewTests(unittest.TestCase):
    def test_white_in_jaws(self):
        frame = np.full((H, W, 3), 60, np.uint8)
        x0, x1, y0 = int(W * g.JAW_VIEW[0]), int(W * g.JAW_VIEW[1]), int(H * g.JAW_VIEW[2])
        self.assertLess(p.white_in_jaw_view(frame), 0.01)
        frame[y0:, x0:x1] = 250
        self.assertGreater(p.white_in_jaw_view(frame), 0.95)
        far = np.full((H, W), 600.0, np.float32)  # white table far below the camera: not in the jaws
        self.assertLess(p.white_in_jaw_view(frame, far), 0.01)
        near = np.full((H, W), 60.0, np.float32)
        self.assertGreater(p.white_in_jaw_view(frame, near), 0.95)


class FakeGripper:
    def __init__(self, reports=True, follows=True):
        self.pos, self.reports, self.follows, self.calls = 850.0, reports, follows, []

    async def do_command(self, command, timeout=None):
        self.calls.append(command)
        if 'set' in command:
            self.target = command['set']
            if self.follows:
                self.pos = command['set']
            return {}
        return {'pos': self.pos} if self.reports else {}

    async def open(self, timeout=None):
        self.calls.append('open')
        self.pos = 850.0


class HalfOpenTests(unittest.TestCase):
    @staticmethod
    def run_open(gripper, half):
        import asyncio

        async def no_sleep(_):
            return None
        return asyncio.run(p.open_gripper(gripper, half, sleep=no_sleep))

    def test_full_open_by_default(self):
        gripper = FakeGripper()
        self.assertIsNone(self.run_open(gripper, None))
        self.assertEqual(gripper.calls, ['open'])

    def test_half_open_sets_and_verifies_position(self):
        gripper = FakeGripper()
        self.assertEqual(self.run_open(gripper, 425.0), 425.0)
        self.assertIn({'set': 425.0}, gripper.calls)
        self.assertNotIn('open', gripper.calls)

    def test_falls_back_to_full_open_when_driver_does_not_report(self):
        gripper = FakeGripper(reports=False)
        self.assertIsNone(self.run_open(gripper, 425.0))
        self.assertEqual(gripper.calls[-1], 'open')
        self.assertNotIn({'set': 425.0}, gripper.calls)  # no jaw command sent blind

    def test_falls_back_to_full_open_when_position_never_reached(self):
        gripper = FakeGripper(follows=False)
        self.assertIsNone(self.run_open(gripper, 425.0))
        self.assertEqual(gripper.calls[-1], 'open')


class CliTests(unittest.TestCase):
    def test_half_open_option(self):
        self.assertIsNone(p.parse_args([]).half_open)
        self.assertEqual(p.parse_args(['--half-open']).half_open, 425.0)
        self.assertEqual(p.parse_args(['--half-open', '300']).half_open, 300.0)

    def test_defaults_and_overrides(self):
        a = p.parse_args([])
        self.assertEqual(a.stage, 'scan')
        a = p.parse_args(['--stage', 'go', '--container-xy', '300', '-50', '--container-height', '90'])
        self.assertEqual((a.stage, a.container_xy, a.container_height), ('go', [300.0, -50.0], 90.0))


if __name__ == '__main__':
    unittest.main()
