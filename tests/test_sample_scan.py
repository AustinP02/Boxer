"""Exercise execution boundaries using fake hardware; never connects to Viam."""
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]/"src"))

import argparse
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import numpy as np
from viam.proto.common import Pose, PoseInFrame, WorldState
import sample_scan


class ScanBoundaryTests(unittest.TestCase):
    def exercise(self, execute, fail_move=False):
        machine = AsyncMock()
        arm = AsyncMock()
        motion = AsyncMock()
        motion.get_pose.return_value = PoseInFrame(reference_frame='world',
                                                  pose=Pose(x=200, y=0, z=300, o_z=-1))
        motion.move.return_value = True
        if fail_move:
            motion.move.side_effect = RuntimeError('simulated planner failure')
        args = argparse.Namespace(execute=execute, travel_y=30, drop_z=15,
                                  side_x=5, clearance_mm=100)
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(sample_scan, 'ROOT', Path(directory)),
                  patch.object(sample_scan, 'OUTPUT_DIR', Path(directory)),
                  patch.dict(os.environ, {'VIAM_API_KEY':'fake', 'VIAM_API_KEY_ID':'00000000-0000-4000-8000-000000000000', 'VIAM_ADDRESS':'fake'}),
                  patch.object(sample_scan.RobotClient, 'at_address', AsyncMock(return_value=machine)),
                  patch.object(sample_scan.Arm, 'from_robot', return_value=arm),
                  patch.object(sample_scan.Camera, 'from_robot', return_value=object()),
                  patch.object(sample_scan.MotionClient, 'from_robot', return_value=motion),
                  patch.object(sample_scan, 'inspect_setup', AsyncMock(return_value=(WorldState(), {}))),
                  patch.object(sample_scan, 'get_color_frame', AsyncMock(return_value=np.zeros((100,100,3),np.uint8))),
                  patch('builtins.print')):
                if fail_move:
                    with self.assertRaisesRegex(RuntimeError, 'planner failure'):
                        asyncio.run(sample_scan.run(args))
                else:
                    asyncio.run(sample_scan.run(args))
        return machine, arm, motion

    def test_inspect_never_issues_arm_commands(self):
        machine, arm, motion = self.exercise(False)
        arm.do_command.assert_not_awaited()
        arm.stop.assert_not_awaited()
        motion.move.assert_not_awaited()
        machine.close.assert_awaited_once()

    def test_failed_move_stops_and_does_not_return_home(self):
        machine, arm, motion = self.exercise(True, True)
        self.assertEqual(motion.move.await_count, 1)
        arm.stop.assert_awaited_once()
        machine.close.assert_awaited_once()

    def test_reject_large_sample_before_connect(self):
        with self.assertRaises(ValueError):
            sample_scan.scan_poses(Pose(), travel_y=500)


if __name__ == '__main__':
    unittest.main()
