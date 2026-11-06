"""Temporal container IDs and UI-only smoothing. Stale boxes are NEVER targets."""
from copy import deepcopy
import math
import time
import cv2
import numpy as np


def overlap_fraction(a,b):
    x,y,w,h=a; xx,yy,ww,hh=b
    area=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return area/max(1,min(w*h,ww*hh))


def quad(detection):
    polygon=detection.get('polygon')
    if polygon:
        return cv2.boxPoints(cv2.minAreaRect(np.array(polygon,np.float32)))
    x,y,w,h=detection['roi']
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)


def align_quad(current,previous):
    choices=[np.roll(q,k,axis=0) for q in (current,current[::-1]) for k in range(4)]
    return min(choices,key=lambda q:float(np.sum((q-previous)**2)))


def clean_candidates(detections):
    """Suppress incomplete, nested duplicate, and spanning same-role boxes."""
    valid=[]; rejected=[]
    for item in detections:
        if item.get('clipped'):
            rejected.append(dict(item,rejection_reason='partial / clipped'))
        else:
            valid.append(deepcopy(item))
    # Two 52 mm destinations form a ~104 mm pair: nearly the 108 mm source
    # width. Interior independent rims must win over the outer size hypothesis,
    # including when the outer candidate has the OTHER role.
    not_groups=[]
    for item in valid:
        area=item['roi'][2]*item['roi'][3]
        children=[]
        for other in valid:
            other_area=other['roi'][2]*other['roi'][3]
            rim=other.get('proposal_origin')=='black-rim' or 'visible black rim' in other.get('rim_method','')
            if other is not item and rim and 0.15*area<other_area<0.75*area and overlap_fraction(item['roi'],other['roi'])>0.85:
                children.append(other)
        merged=any(overlap_fraction(a['roi'],b['roi'])<0.15
                   for i,a in enumerate(children) for b in children[i+1:])
        if merged:
            rejected.append(dict(item,rejection_reason='encloses separate container rims'))
        else:
            not_groups.append(item)
    valid=not_groups
    # Prefer measured-rim proposals over nested ML duplicates.
    valid.sort(key=lambda d:(d.get('proposal_origin')=='black-rim',
                             -d.get('dimension_error',10),d.get('confidence') or 0),reverse=True)
    selected=[]
    for item in valid:
        area=item['roi'][2]*item['roi'][3]
        suppress=False
        for other in selected:
            if other['role']!=item['role']:
                continue
            other_area=other['roi'][2]*other['roi'][3]
            coverage=overlap_fraction(item['roi'],other['roi'])
            if coverage>0.8:
                # Two nested same-role outlines cannot be separate side-by-side bins.
                # In particular, do not let a broad ML rectangle absorb a rim bin.
                if area>other_area*1.5 or max(area,other_area)/max(1,min(area,other_area))<1.5:
                    suppress=True; break
        if suppress:
            rejected.append(dict(item,rejection_reason='duplicate / spanning proposal'))
        else:
            selected.append(item)
    return selected,rejected


class ContainerTracker:
    def __init__(self,hold_seconds=3.0,max_misses=2,min_hits=3):
        self.hold_seconds=hold_seconds
        self.max_misses=max_misses
        self.min_hits=min_hits
        self.tracks={}
        self.next_id=1
        self.context=None

    def reset(self):
        self.tracks.clear()

    def update(self,detections,now=None,context=None):
        now=time.monotonic() if now is None else now
        if context is not None and self.context is not None:
            # Context is (image width,height,camera x,y,z,ox,oy,oz,theta).
            old=np.array(self.context[2:],float); new=np.array(context[2:],float)
            if context[:2]!=self.context[:2] or np.linalg.norm(new[:3]-old[:3])>2 or np.linalg.norm(new[3:6]-old[3:6])>0.02 or abs(new[-1]-old[-1])>1:
                self.reset()
        self.context=context
        current,rejected=clean_candidates(detections)
        for tid in list(self.tracks):
            if now-self.tracks[tid]['seen']>self.hold_seconds:
                del self.tracks[tid]
        # Reject boxes enclosing two recently observed separate tracks even when
        # one or both small boxes disappear from this particular frame.
        filtered=[]
        for item in current:
            covered=[]
            area=item['roi'][2]*item['roi'][3]
            for tid,track in self.tracks.items():
                previous=track['detection']
                previous_area=previous['roi'][2]*previous['roi'][3]
                if area>1.4*previous_area and overlap_fraction(item['roi'],previous['roi'])>0.8:
                    covered.append(tid)
            if len(covered)>=2:
                rejected.append(dict(item,rejection_reason='merged across existing tracks'))
            else:
                filtered.append(item)
        current=filtered
        matches=[]
        for index,item in enumerate(current):
            center=np.array(item['center'],float)
            area=item['roi'][2]*item['roi'][3]
            for tid,track in self.tracks.items():
                old=track['detection']
                if old['role']!=item['role']:
                    continue
                ratio=area/max(1,old['roi'][2]*old['roi'][3])
                distance=float(np.linalg.norm(center-np.array(old['center'])))
                diagonal=math.hypot(old['roi'][2],old['roi'][3])
                if 0.6<=ratio<=1.65 and distance<=max(35,diagonal*0.8):
                    cost=distance/max(diagonal,1)+(1-overlap_fraction(item['roi'],old['roi']))*0.4
                    matches.append((cost,index,tid))
        used_items=set(); used_tracks=set(); assigned={}
        for _,index,tid in sorted(matches):
            if index not in used_items and tid not in used_tracks:
                assigned[index]=tid; used_items.add(index); used_tracks.add(tid)
        fresh=[]
        for index,item in enumerate(current):
            tid=assigned.get(index)
            observed_quad=quad(item)
            if tid is None:
                tid=self.next_id; self.next_id+=1
                track=dict(seen=now,hits=0,misses=0,display=observed_quad)
                self.tracks[tid]=track
            else:
                track=self.tracks[tid]
                observed_quad=align_quad(observed_quad,track['display'])
                distance=float(np.linalg.norm(np.array(item['center'])-np.array(track['detection']['center'])))
                # Follow meaningful movement immediately. Smooth only tiny jitter.
                alpha=1.0 if distance>12 else 0.55
                track['display']=alpha*observed_quad+(1-alpha)*track['display']
            track.update(seen=now,hits=track['hits']+1,misses=0,detection=deepcopy(item))
            used_tracks.add(tid)
            item.update(track_id=tid,tracking_state='confirmed' if track['hits']>=self.min_hits else 'tentative',
                        fresh=True,pick_eligible=track['hits']>=self.min_hits,
                        display_polygon=track['display'].tolist())
            fresh.append(item)
        ghosts=[]
        for tid in list(self.tracks):
            if tid in used_tracks:
                continue
            track=self.tracks[tid]; track['misses']+=1
            age=now-track['seen']
            if track['misses']>self.max_misses or age>self.hold_seconds:
                del self.tracks[tid]; continue
            ghost=deepcopy(track['detection'])
            ghost.update(track_id=tid,tracking_state='stale',fresh=False,pick_eligible=False,
                         age_seconds=round(age,2),display_polygon=track['display'].tolist())
            # Do not reuse previous contents/color assignment as current evidence.
            for key in ('colors','status','empty_verified','free_center','free_radius_mm'):
                ghost.pop(key,None)
            ghosts.append(ghost)
        return fresh,ghosts,rejected
