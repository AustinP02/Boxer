import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import cv2
import numpy as np
from instance_analysis import analyze_instances
from object_instances import dedupe_masks


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.full((160,240,3),255,np.uint8)
        self.a=np.zeros((160,240),bool); self.a[40:100,40:90]=True
        self.b=np.zeros((160,240),bool); self.b[40:100,90:140]=True
        self.frame[self.a|self.b]=(0,255,255)
        self.config=dict(min_area_px=100,jaw_margin_px=5,max_depth_spread_mm=20,
                         max_grasp_width_mm=60,max_object_span_mm=120)
        self.sources=[dict(roi=[10,10,200,140])]

    def test_two_touching_yellow_instances_remain_two(self):
        masks=[dict(mask=m,quality=.9,backend='test') for m in (self.a,self.b)]
        items=analyze_instances(self.frame,self.sources,self.config,masks)
        self.assertEqual(len(items),2)
        self.assertEqual([i['color'] for i in items],['yellow','yellow'])
        self.assertTrue(all('crowded: adjacent instance in jaw clearance' in i['rejected'] for i in items))

    def test_no_model_instances_does_not_fall_back_to_color_blob(self):
        self.assertEqual(analyze_instances(self.frame,self.sources,self.config,[]),[])

    def test_duplicate_seed_masks_do_not_become_two_objects(self):
        masks=[dict(mask=self.a,quality=.9),dict(mask=self.a.copy(),quality=.8)]
        self.assertEqual(len(dedupe_masks(masks)),1)


if __name__=='__main__':
    unittest.main()
