import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import asyncio
from datetime import datetime,timezone
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch
import numpy as np
from viam.proto.common import Geometry,Pose,PoseInFrame,GeometriesInFrame,WorldState,Transform
import pickup_demo as demo


class PickupTests(unittest.TestCase):
    def config(self):
        return dict(sources=[dict(roi=[0,0,100,100],rim_z_mm=67),dict(roi=[110,0,100,100],rim_z_mm=64)],
            container_dimensions_mm={'source':[138,108,60]},grasp_offset_mm=-15,
            transit_z_mm=410,pick_orientation=dict(o_x=0,o_y=0,o_z=-1,theta=60),
            _table_top_z_mm=-23)

    def item(self):
        return dict(color='green',source=0,world_mm=[390,275,32],
                    rejected=['no matching-color or verified-empty destination'])

    def exercise(self,stage='approach',execute=True,fail_move=False,fail_observe=False,in_place=False):
        config=self.config(); item=self.item()
        _,pick,approach,lift=demo.pickup_target([item],config,'green')
        machine=SimpleNamespace(close=AsyncMock(),get_frame_system_config=AsyncMock(return_value=[
            SimpleNamespace(frame=Transform(reference_frame='gripper',pose_in_observer_frame=
                PoseInFrame(reference_frame='arm',pose=Pose(z=160,o_z=1,theta=180))))]))
        arm=SimpleNamespace(do_command=AsyncMock(),stop=AsyncMock())
        gripper=SimpleNamespace(open=AsyncMock(),grab=AsyncMock(return_value=True),stop=AsyncMock(),
            is_holding_something=AsyncMock(return_value=SimpleNamespace(is_holding_something=True)))
        async def pose(name,*args,**kwargs):
            return PoseInFrame(reference_frame='world',pose=(Pose(z=-123,o_z=1) if name=='table' else approach))
        motion=SimpleNamespace(get_pose=AsyncMock(side_effect=pose),move=AsyncMock(return_value=True))
        if fail_move:
            motion.move.side_effect=RuntimeError('planner stopped')
        geometry=Geometry(center=Pose(o_z=1),box=demo.sorter.RectangularPrism(dims_mm=demo.sorter.Vector3(x=3000,y=3000,z=200)))
        world=WorldState(obstacles=[GeometriesInFrame(reference_frame='table',geometries=[geometry])])
        observe=AsyncMock(return_value={'ok':True})
        if fail_observe:
            observe.side_effect=ValueError('target moved')
        env={'VIAM_ADDRESS':'test.invalid','VIAM_API_KEY':'fake','VIAM_API_KEY_ID':'00000000-0000-0000-0000-000000000001'}
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            if stage=='finish':
                state=dict(time=datetime.now(timezone.utc).isoformat(),item=item,
                    pick=demo.sorter.pose_dict(pick),approach=demo.sorter.pose_dict(approach),
                    lift=demo.sorter.pose_dict(lift),world=world.SerializeToString().hex())
                (folder/'target.json').write_text(json.dumps(state))
            with patch.object(demo,'FOLDER',folder),patch.dict(demo.os.environ,env), \
                 patch.object(demo,'load_dotenv'), \
                 patch.object(demo.RobotClient,'at_address',AsyncMock(return_value=machine)), \
                 patch.object(demo.Arm,'from_robot',return_value=arm), \
                 patch.object(demo.Gripper,'from_robot',return_value=gripper), \
                 patch.object(demo.Camera,'from_robot',return_value=object()), \
                 patch.object(demo.MotionClient,'from_robot',return_value=motion), \
                 patch.object(demo,'inspect_setup',AsyncMock(return_value=(world,{}))), \
                 patch.object(demo.sorter,'ContainerDetector'), \
                 patch.object(demo.sorter,'ObjectInstanceDetector'), \
                 patch.object(demo.sorter,'snapshot',AsyncMock(return_value=([item],np.array([0,0,500]),np.diag([1,1,-1]),(500,500,100,100)))), \
                 patch.object(demo,'observe_target',observe),patch('builtins.print'):
                args=argparse.Namespace(stage=stage,color='green',execute=execute,in_place=in_place)
                if fail_move or fail_observe:
                    with self.assertRaisesRegex((RuntimeError,ValueError),'planner stopped|target moved'):
                        asyncio.run(demo.run(args,config))
                else:
                    asyncio.run(demo.run(args,config))
        machine.close.assert_awaited_once()
        return arm,gripper,motion

    def test_destination_rejection_does_not_block_pickup_but_depth_does(self):
        item=self.item()
        self.assertEqual(demo.pickup_target([item],self.config(),'green')[1].z,17)
        item['rejected'].append('uneven depth / occlusion')
        with self.assertRaises(ValueError):
            demo.pickup_target([item],self.config(),'green')

    def test_observation_never_moves_or_opens(self):
        arm,gripper,motion=self.exercise(execute=False)
        arm.do_command.assert_not_awaited(); motion.move.assert_not_awaited()
        gripper.open.assert_not_awaited(); gripper.grab.assert_not_awaited()

    def test_approach_stops_before_grasp(self):
        arm,gripper,motion=self.exercise()
        self.assertEqual(motion.move.await_count,3)
        gripper.open.assert_awaited_once(); gripper.grab.assert_not_awaited()
        arm.stop.assert_awaited_once()

    def test_current_view_approach_uses_live_downward_orientation(self):
        arm,gripper,motion=self.exercise(in_place=True)
        self.assertEqual(motion.move.await_count,3)
        gripper.open.assert_awaited_once()
        gripper.grab.assert_not_awaited()

    def test_planner_failure_stops_without_grasp_or_retry(self):
        arm,gripper,motion=self.exercise(fail_move=True)
        self.assertEqual(motion.move.await_count,1)
        arm.stop.assert_awaited_once(); gripper.grab.assert_not_awaited()

    def test_finish_rechecks_target_before_descent(self):
        arm,gripper,motion=self.exercise(stage='finish',fail_observe=True)
        motion.move.assert_not_awaited(); gripper.grab.assert_not_awaited()

    def test_finish_grasps_once_lifts_and_never_releases(self):
        arm,gripper,motion=self.exercise(stage='finish')
        self.assertEqual(motion.move.await_count,2)
        gripper.grab.assert_awaited_once(); gripper.open.assert_not_awaited()
        gripper.is_holding_something.assert_awaited_once()


if __name__=='__main__':
    unittest.main()
