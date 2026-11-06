import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import cv2
import numpy as np
from rim_fusion import fuse_rims, standalone_black_rims, cue_maps, resolve_geometric_roles


class RimTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.full((300,400,3),230,np.uint8)
        self.depth=np.full((300,400),500,np.float32)
        self.proposal=dict(role='destination',roi=[85,75,170,120],center=[170,135],polygon=None,confidence=0.8)

    def test_visible_black_rim_refines_ml_box(self):
        cv2.rectangle(self.frame,(90,80),(250,190),(0,0,0),5)
        result,_=fuse_rims(self.frame,[self.proposal])
        self.assertEqual(result[0]['rim_method'],'ML + visible black rim')
        self.assertEqual(result[0]['confidence'],0.8)

    def test_depth_holes_without_visual_evidence_are_not_containers(self):
        cv2.rectangle(self.depth,(90,80),(250,190),0,8)
        before=self.depth.copy()
        result,_=fuse_rims(self.frame,[self.proposal],self.depth,registered=True)
        self.assertEqual(result[0]['rim_method'],'ML only / rim unconfirmed')
        np.testing.assert_array_equal(before,self.depth)
        self.assertEqual(fuse_rims(self.frame,[],self.depth,registered=True)[0],[])

    def test_black_rim_is_proposed_without_ml(self):
        cv2.rectangle(self.frame,(90,80),(250,190),(105,105,105),3)
        candidates=standalone_black_rims(self.frame,cue_maps(self.frame))
        self.assertTrue(candidates)
        self.assertTrue(all(c['confidence'] is None for c in candidates))
        self.assertEqual(resolve_geometric_roles(candidates),[])

    def test_geometric_role_requires_unique_size_match(self):
        a=dict(proposal_origin='black-rim',geometry_group=1,dimension_error=.1,
               role='destination',roi=[1,1,100,50])
        b=dict(a,role='source',dimension_error=.8)
        self.assertEqual(resolve_geometric_roles([a,b])[0]['role'],'destination')
        self.assertEqual(resolve_geometric_roles([a,dict(b,dimension_error=.12)]),[])

    def test_unregistered_depth_is_excluded(self):
        cv2.rectangle(self.depth,(90,80),(250,190),0,8)
        result,cues=fuse_rims(self.frame,[self.proposal],self.depth,registered=False)
        self.assertEqual(np.count_nonzero(cues['joint']),0)
        self.assertEqual(result[0]['roi'],self.proposal['roi'])


if __name__=='__main__':
    unittest.main()
