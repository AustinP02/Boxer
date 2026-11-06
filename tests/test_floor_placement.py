import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from floor_placement import fit_footprint, choose_placement, floor_context


class PlacementTests(unittest.TestCase):
    def context(self):
        return dict(available=np.full((140,52),255,np.uint8),lower=np.array([0,0]),
                    floor_z=0,origin=np.array([0,0,500]),
                    basis=np.diag([1,1,-1]),intrinsics=(500,500,100,100))

    def offsets(self,width=32,length=34):
        return [[-width/2,-length/2],[width/2,-length/2],
                [width/2,length/2],[-width/2,length/2]]

    def test_rectangular_block_fits_narrow_bin_with_margin(self):
        self.assertIsNotNone(fit_footprint(self.offsets(),self.context()))

    def test_wide_block_does_not_fit(self):
        self.assertIsNone(fit_footprint(self.offsets(width=46),self.context()))

    def test_invalid_depth_patch_does_not_count_as_free(self):
        context=self.context()
        context['available'][:]=0
        self.assertIsNone(fit_footprint(self.offsets(),context))
        context['available'][:]=255
        context['available'][:,24:28]=0
        self.assertIsNone(fit_footprint(self.offsets(),context))

    def test_unknown_empty_bin_is_not_used(self):
        bins=[dict(id='a',colors=[],empty_verified=False)]
        self.assertIsNone(choose_placement('green',bins,self.offsets(),{'a':self.context()}))

    def test_full_matching_bin_keeps_assignment(self):
        full=self.context(); full['available'][:]=0
        bins=[dict(id='a',colors=['green'],empty_verified=False),
              dict(id='b',colors=[],empty_verified=True)]
        self.assertIsNone(choose_placement('green',bins,self.offsets(),{'a':full,'b':self.context()}))
        self.assertEqual(choose_placement('yellow',bins,self.offsets(),{'a':full,'b':self.context()})['id'],'b')

    def test_raw_depth_holes_remain_unavailable(self):
        container=dict(roi=[20,20,60,150])
        context=floor_context(container,np.zeros((200,200),np.float32),0,
                              np.array([0,0,500]),np.diag([1,1,-1]),(500,500,100,100))
        self.assertIsNotNone(context)
        self.assertFalse(np.any(context['available']))


if __name__=='__main__':
    unittest.main()
