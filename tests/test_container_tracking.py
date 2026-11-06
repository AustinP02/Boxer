import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from container_tracking import ContainerTracker,clean_candidates


def box(x=10,y=10,w=100,h=50,role='destination',**extras):
    return dict(role=role,roi=[x,y,w,h],center=[x+w/2,y+h/2],confidence=.8,**extras)


class TrackingTests(unittest.TestCase):
    def test_brief_miss_is_visible_but_not_fresh_or_pickable(self):
        tracker=ContainerTracker()
        a,_,_=tracker.update([box()],now=0)
        b,ghosts,_=tracker.update([],now=.5)
        self.assertEqual(b,[])
        self.assertEqual(ghosts[0]['track_id'],a[0]['track_id'])
        self.assertFalse(ghosts[0]['pick_eligible'])
        c,_,_=tracker.update([box(x=14)],now=1)
        self.assertEqual(c[0]['track_id'],a[0]['track_id'])

    def test_long_miss_expires(self):
        tracker=ContainerTracker()
        a,_,_=tracker.update([box()],now=0)
        _,ghosts,_=tracker.update([],now=4)
        self.assertEqual(ghosts,[])
        b,_,_=tracker.update([box()],now=4.1)
        self.assertNotEqual(a[0]['track_id'],b[0]['track_id'])

    def test_merged_box_does_not_overwrite_two_tracks(self):
        tracker=ContainerTracker()
        a,_,_=tracker.update([box(),box(x=125)],now=0)
        b,ghosts,rejected=tracker.update([box(w=215)],now=.2)
        self.assertEqual(b,[])
        self.assertEqual(len(ghosts),2)
        self.assertEqual(len(rejected),1)
        c,_,_=tracker.update([box(x=13),box(x=128)],now=.4)
        self.assertEqual([d['track_id'] for d in a],[d['track_id'] for d in c])

    def test_fresh_coordinates_are_not_smoothed_for_planning(self):
        tracker=ContainerTracker()
        tracker.update([box()],now=0)
        fresh,_,_=tracker.update([box(x=16)],now=.2)
        self.assertEqual(fresh[0]['roi'],[16,10,100,50])
        self.assertNotEqual(fresh[0]['display_polygon'][0][0],16)

    def test_camera_motion_clears_old_tracks(self):
        tracker=ContainerTracker()
        context=(1280,720,0,0,500,0,0,-1,0)
        a,_,_=tracker.update([box()],now=0,context=context)
        b,ghosts,_=tracker.update([],now=.2,context=(1280,720,30,0,500,0,0,-1,0))
        self.assertEqual(ghosts,[])

    def test_two_destination_rims_are_not_one_source(self):
        pair=box(x=0,y=0,w=210,h=150,role='source')
        a=box(x=0,y=0,w=100,h=150,proposal_origin='black-rim',dimension_error=.1)
        b=box(x=110,y=0,w=100,h=150,proposal_origin='black-rim',dimension_error=.1)
        clean,rejected=clean_candidates([pair,a,b])
        self.assertEqual(len(clean),2)
        self.assertTrue(all(d['role']=='destination' for d in clean))
        self.assertEqual(rejected[0]['rejection_reason'],'encloses separate container rims')

    def test_clipped_box_and_nested_ml_merge_are_rejected(self):
        clean,rejected=clean_candidates([box(clipped=True)])
        self.assertEqual(clean,[])
        rim=box(proposal_origin='black-rim',dimension_error=.1)
        clean,_=clean_candidates([box(x=0,y=0,w=200,h=100),rim])
        self.assertEqual(len(clean),1)
        self.assertEqual(clean[0]['proposal_origin'],'black-rim')


if __name__=='__main__':
    unittest.main()
