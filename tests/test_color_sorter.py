import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]/"src"))
import argparse
import asyncio
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
import cv2
import numpy as np
from sort_vision import decode_depth,camera_point,container_contents,choose_destination,detect_source_items
from container_detector import remove_group_boxes,validate_layout
import color_sorter


class SorterTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((_ProjectPath(__file__).resolve().parents[1]/'config/sort_config.json').read_text())

    def test_depth_endianness_and_corrupt_payload(self):
        data=b'DEPTHMAP'+(2).to_bytes(8,'big')+(2).to_bytes(8,'big')+np.array([[250,500],[1000,0]],dtype='>u2').tobytes()
        np.testing.assert_array_equal(decode_depth(data),[[250,500],[1000,0]])
        with self.assertRaises(ValueError):
            decode_depth(data[:-1])

    def test_projection_units(self):
        np.testing.assert_allclose(camera_point((150,100),500,(500,500,100,100)),[50,0,500])

    def test_existing_color_precedes_empty(self):
        empty=dict(id='bin-1',colors=[],empty_verified=True)
        yellow=dict(id='bin-4',colors=['yellow'],empty_verified=False)
        self.assertEqual(choose_destination('yellow',[empty,yellow])['id'],'bin-4')
        self.assertEqual(choose_destination('green',[empty,yellow])['id'],'bin-1')

    def test_unknown_and_mixed_bins_are_not_empty(self):
        bins=[dict(id='a',colors=[],empty_verified=False),dict(id='b',colors=['red','blue'],empty_verified=False)]
        self.assertIsNone(choose_destination('red',bins))

    def test_matching_bin_without_room_does_not_claim_capacity(self):
        bins=[dict(id='red',colors=['red'],empty_verified=False,free_radius_mm=5),
              dict(id='empty',colors=[],empty_verified=True,free_radius_mm=50)]
        self.assertIsNone(choose_destination('red',bins,required_radius_mm=20))
        self.assertEqual(choose_destination('blue',bins,required_radius_mm=20)['id'],'empty')

    def test_measured_dimensions_reject_merged_destination(self):
        self.config['_table_top_z_mm']=0
        def box(width):
            return dict(role='destination',roi=[20,20,width,56],center=[20+width/2,48])
        kept=color_sorter.dimension_filter([box(148),box(296)],self.config,
                                            (500,500,100,100),np.array([0,0,500]),np.diag([1,1,-1]))
        self.assertEqual(len(kept),1)
        self.assertEqual(kept[0]['roi'][2],148)

    def test_color_content_and_depth_verified_empty(self):
        f=np.full((200,200,3),255,np.uint8)
        container=dict(id='bin-1',roi=[20,20,160,160])
        self.config['destination_floor_z_mm']=-500
        depth=np.full((200,200),500,np.float32)
        kw=dict(depth=depth,intrinsics=(500,500,100,100),origin=np.zeros(3),basis=np.diag([1,1,-1]))
        self.assertTrue(container_contents(f,container,self.config,**kw)['empty_verified'])
        depth[40:160,40:160]=450
        self.assertFalse(container_contents(f,container,self.config,**kw)['empty_verified'])
        cv2.rectangle(f,(70,70),(120,120),(0,255,255),-1)
        self.assertEqual(container_contents(f,container,self.config)['colors'],['yellow'])

    def test_reject_group_box_and_incomplete_layout(self):
        def box(roi):
            x,y,w,h=roi
            return dict(role='source',roi=roi,center=[x+w/2,y+h/2])
        proposals=[box([0,0,220,100]),box([0,0,100,100]),box([120,0,100,100])]
        self.assertEqual(len(remove_group_boxes(proposals)),2)
        with self.assertRaisesRegex(ValueError,'require 2 / 4'):
            validate_layout(proposals)

    def test_source_region_moves_with_detection(self):
        frame=np.full((300,500,3),255,np.uint8)
        cv2.rectangle(frame,(260,100),(300,150),(0,0,255),-1)
        old=[dict(roi=[20,50,130,180])]
        new=[dict(roi=[220,50,130,180])]
        self.assertEqual(detect_source_items(frame,old,self.config),[])
        self.assertEqual(detect_source_items(frame,new,self.config)[0]['color'],'red')

    def test_uncommissioned_run_never_connects(self):
        args=argparse.Namespace(command='run',execute=True)
        with patch.object(color_sorter.RobotClient,'at_address',AsyncMock()) as connect:
            with self.assertRaisesRegex(ValueError,'Calibration needed'):
                asyncio.run(color_sorter.run(args,self.config))
            connect.assert_not_awaited()


    def test_destination_teach_without_camera_or_actuator_clients(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        args=argparse.Namespace(command='teach',name='destination')
        pose=SimpleNamespace(x=10,y=20,z=90,o_x=0,o_y=0,o_z=-1,theta=0)
        machine=SimpleNamespace(close=AsyncMock())
        motion=SimpleNamespace(get_pose=AsyncMock(return_value=SimpleNamespace(pose=pose)))
        env={'VIAM_ADDRESS':'test.invalid','VIAM_API_KEY':'test','VIAM_API_KEY_ID':'00000000-0000-0000-0000-000000000001'}
        with patch.dict(color_sorter.os.environ,env), \
             patch.object(color_sorter,'load_dotenv'), \
             patch.object(color_sorter.RobotClient,'at_address',AsyncMock(return_value=machine)), \
             patch.object(color_sorter.MotionClient,'from_robot',return_value=motion), \
             patch.object(color_sorter.Camera,'from_robot',Mock(side_effect=KeyError(('camera','cam')))) as camera, \
             patch.object(color_sorter.Arm,'from_robot') as arm, \
             patch.object(color_sorter.Gripper,'from_robot') as gripper, \
             patch.object(color_sorter,'save_config') as save:
            asyncio.run(color_sorter.run(args,self.config))
        self.assertEqual(self.config['destination_release_z_mm'],90)
        self.assertEqual(self.config['destination_floor_z_mm'],35)
        save.assert_called_once_with(self.config)
        motion.get_pose.assert_awaited_once_with('gripper','world',timeout=15)
        machine.close.assert_awaited_once()
        camera.assert_not_called()
        arm.assert_not_called()
        gripper.assert_not_called()



    def test_height_disagreement_is_rejected_before_grasp(self):
        config={'container_dimensions_mm':{'source':[138,108,60]},'grasp_offset_mm':-15}
        source={'rim_z_mm':122.3}
        self.assertIn('below taught source floor',color_sorter.source_height_error(
            {'world_mm':[384,137,32.8]},source,config))
        self.assertIn('too close',color_sorter.source_height_error(
            {'world_mm':[384,137,72]},source,config))
        self.assertIsNone(color_sorter.source_height_error(
            {'world_mm':[384,137,95]},source,config))


if __name__=='__main__':
    unittest.main()
