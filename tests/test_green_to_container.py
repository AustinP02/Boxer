import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import green_to_container as g


def item(color='green', rejected=(), world=True):
    d = dict(color=color, rejected=list(rejected), source=0)
    if world:
        d['world_mm'] = [1, 2, 3]
    return d


class GreenTargetTests(unittest.TestCase):
    def test_single_green_selected(self):
        target = item()
        self.assertIs(g.green_target([item('red'), target]), target)

    def test_none_or_rejected_or_multiple_refuse(self):
        for items in ([item('red')], [item(rejected=['x'])], [item(world=False)], [item(), item()]):
            with self.assertRaises(ValueError):
                g.green_target(items)


if __name__ == '__main__':
    unittest.main()


class WristAlignmentTests(unittest.TestCase):
    orient = dict(o_x=-0.0428, o_y=-0.0802, o_z=-0.9959, theta=62.06)

    def test_yaw_decreases_with_theta(self):
        a = g.tool_x_yaw(**self.orient)
        b = g.tool_x_yaw(**{**self.orient, 'theta': self.orient['theta'] + 10})
        self.assertAlmostEqual((a - b) % 360, 10, places=1)

    @staticmethod
    def rect(deg, long_side=10, short_side=10):
        import math
        r = math.radians(deg)
        u, v = (math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))
        c0 = (0.0, 0.0)
        c1 = (long_side * u[0], long_side * u[1])
        c2 = (c1[0] + short_side * v[0], c1[1] + short_side * v[1])
        c3 = (short_side * v[0], short_side * v[1])
        return [c0, c1, c2, c3]

    def test_aligned_jaws_match_square_faces(self):
        for deg in (3, 20, 44, 60, 80):
            for offset in (0.0, 17.0, -63.0):
                turned, delta = g.aligned_orientation(self.orient, self.rect(deg), offset)
                jaw = g.tool_x_yaw(**turned) + offset
                self.assertAlmostEqual(((jaw - deg) + 45) % 90 - 45, 0, places=1)

    def test_long_block_jaws_close_across_the_short_side(self):
        for deg in (3, 20, 44, 60, 80, 130):
            for offset in (0.0, 25.0):
                turned, _ = g.aligned_orientation(self.orient, self.rect(deg, 20, 10), offset)
                jaw = g.tool_x_yaw(**turned) + offset
                # jaw axis must be perpendicular to the long edge (mod 180)
                self.assertAlmostEqual(((jaw - (deg + 90)) + 90) % 180 - 90, 0, places=1)

    def test_no_footprint_keeps_orientation(self):
        turned, delta = g.aligned_orientation(self.orient, None)
        self.assertEqual(turned, self.orient)
        self.assertEqual(delta, 0.0)


class PlanTests(unittest.TestCase):
    def plan(self, colors):
        import numpy as np
        from unittest.mock import patch
        items = [dict(color=c, rejected=[], world_mm=[300 + 10 * i, 50, 30], source=0, footprint_world_offsets_mm=None, center=[640, 100], size_mm=[37, 37])
                 for i, c in enumerate(colors)]
        poly = lambda x: [[x, 300], [x + 20, 300], [x + 20, 400], [x, 400]]
        layout = dict(destinations=[dict(id=f'bin-{n + 1}', colors=[], polygon=poly(100 * n), center=[100 * n + 10, 350]) for n in range(4)])
        ray = lambda pixel, z, intr, o, b: np.array([pixel[0], pixel[1], z], dtype=float)
        with patch.object(g.sorter, 'ray_plane', ray):
            return g.plan_from_scan(items, layout, list(g.ORDER), None, None, None, dict(destination_floor_z_mm=0, container_dimensions_mm={'destination': [138, 52, 35]}))

    def test_each_colour_gets_one_bin(self):
        blocks = self.plan(['green', 'red', 'yellow', 'red', 'yellow', 'yellow', 'blue'])['blocks']
        bin_of = {}
        for b in blocks:
            bin_of.setdefault(b['color'], set()).add(b['bin'])
        self.assertTrue(all(len(v) == 1 for v in bin_of.values()))
        self.assertEqual(len({next(iter(v)) for v in bin_of.values()}), 4)

    def test_same_colour_blocks_have_distinct_slots(self):
        blocks = [b for b in self.plan(['yellow'] * 3 + ['red'])['blocks'] if b['color'] == 'yellow']
        self.assertEqual(len({tuple(b['drop_xy']) for b in blocks}), 3)

    def test_overflow_blocks_are_skipped_not_fatal(self):
        plan = self.plan(['red'] * 4)
        self.assertEqual(len(plan['blocks']), 3)
        self.assertEqual(len(plan['skipped']), 1)


class BinFrameTests(unittest.TestCase):
    CONFIG = dict(destination_floor_z_mm=2.0, container_dimensions_mm={'destination': [138, 52, 35]})

    def frame(self, polygon):
        import numpy as np
        from unittest.mock import patch
        seen = []

        def ray(pixel, z, intrinsic, origin, basis):
            seen.append(z)
            return np.array([pixel[0], pixel[1], z], dtype=float)

        with patch.object(g.sorter, 'ray_plane', ray):
            return g.bin_frame(dict(polygon=polygon), self.CONFIG, None, None, None), seen

    def test_bin_outline_is_projected_at_rim_height(self):
        (centre, axis, yaw), seen = self.frame([[0, 0], [150, 0], [150, 50], [0, 50]])
        self.assertEqual(set(seen), {37.0})  # floor 2 + bin height 35
        self.assertAlmostEqual(yaw % 180, 0, places=3)  # the long axis runs along image x
        self.assertAlmostEqual(float(centre[0]), 75, places=3)

    def test_blocks_stay_between_the_end_walls(self):
        inner_half = 138 / 2 - g.BIN_END_MARGIN_MM
        taken, spots = [], []
        for _ in range(3):
            t = g.fit_in_bin(38.0, taken, inner_half)
            self.assertIsNotNone(t)
            taken.append((t - 19 - g.BLOCK_GAP_MM, t + 19 + g.BLOCK_GAP_MM))
            spots.append(t)
            self.assertLessEqual(abs(t) + 19, inner_half + 1e-6)
        self.assertEqual(len(set(spots)), 3)
        self.assertIsNone(g.fit_in_bin(38.0, taken, inner_half))  # a fourth does not fit

    def test_long_block_leaves_room_for_one_square_only(self):
        inner_half = 138 / 2 - g.BIN_END_MARGIN_MM
        t = g.fit_in_bin(70.0, [], inner_half)
        taken = [(t - 35 - g.BLOCK_GAP_MM, t + 35 + g.BLOCK_GAP_MM)]
        self.assertIsNotNone(g.fit_in_bin(38.0, taken, inner_half))
        taken.append((-1e9, 1e9)) if g.fit_in_bin(38.0, taken, inner_half) is None else None
        t2 = g.fit_in_bin(38.0, taken, inner_half)
        if t2 is not None:
            taken.append((t2 - 19 - g.BLOCK_GAP_MM, t2 + 19 + g.BLOCK_GAP_MM))
        self.assertIsNone(g.fit_in_bin(38.0, taken, inner_half))
