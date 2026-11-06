"""Refine ML container proposals with visible rims and optional depth-hole clues.

Unknown depth is never converted into a distance or filled in by this module.
Only registered depth may contribute to proposal geometry.
"""
import cv2
import numpy as np


def cue_maps(frame,depth=None):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),30,90)
    # Thin marker lines appear gray under this camera's exposure. Use several
    # thresholds rather than merging their outlines into shadows at one level.
    dark_layers=[]
    for threshold in (85,100,115,130):
        dark=((hsv[:,:,2]<threshold)&(hsv[:,:,1]<140)).astype(np.uint8)*255
        dark_layers.append(cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)))
    dark=dark_layers[-1]
    holes=np.zeros(gray.shape,np.uint8)
    if depth is not None:
        if depth.shape!=gray.shape:
            raise ValueError('Rim fusion requires equal color/depth dimensions')
        holes=(~np.isfinite(depth)|(depth<=100)|(depth>=2000)).astype(np.uint8)*255
    nearby_holes=cv2.dilate(holes,np.ones((7,7),np.uint8))
    joint=cv2.bitwise_and(edges,nearby_holes)
    return dict(rgb_edges=edges,dark_rim=dark,dark_layers=dark_layers,holes=holes,joint=joint)


def support(polygon,cues):
    ring=np.zeros_like(cues['rgb_edges'])
    cv2.polylines(ring,[np.array(polygon,np.int32)],True,255,7)
    area=ring>0
    if np.count_nonzero(area)<20:
        return dict(black=0.0,rgb=0.0,joint=0.0)
    dilated=lambda image: cv2.dilate(image,np.ones((3,3),np.uint8))
    return dict(black=float(np.mean(cues['dark_rim'][area]>0)),
                rgb=float(np.mean(dilated(cues['rgb_edges'])[area]>0)),
                joint=float(np.mean(dilated(cues['joint'])[area]>0)))


def iou(a,b):
    x,y,w,h=a; xx,yy,ww,hh=b
    intersection=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return intersection/max(1,w*h+ww*hh-intersection)


def rim_candidates(frame,cues,registered=False):
    candidates=[]
    maps=list(cues['dark_layers'])
    if registered:
        maps.append(cv2.morphologyEx(cues['joint'],cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)))
    for mask in maps:
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<1500 or area>0.8*frame.shape[0]*frame.shape[1]:
                continue
            approx=cv2.approxPolyDP(contour,0.025*cv2.arcLength(contour,True),True)
            if len(approx)!=4 or not cv2.isContourConvex(approx):
                continue
            polygon=approx.reshape(-1,2).tolist()
            x,y,w,h=cv2.boundingRect(approx)
            if min(w,h)<25:
                continue
            evidence=support(polygon,cues)
            # A black visible rim can work without depth; depth holes alone cannot.
            if not (evidence['black']>=0.2 and evidence['rgb']>=0.15) and not (
                    registered and evidence['joint']>=0.18 and evidence['rgb']>=0.25):
                continue
            score=evidence['rgb']+evidence['black']+evidence['joint']
            candidates.append(dict(roi=[x,y,w,h],polygon=polygon,evidence=evidence,score=score))
    candidates.sort(key=lambda c:c['score'],reverse=True)
    unique=[]
    for candidate in candidates:
        if not any(iou(candidate['roi'],other['roi'])>0.75 for other in unique):
            unique.append(candidate)
    return unique


def fuse_rims(frame,proposals,depth=None,registered=False):
    cues=cue_maps(frame,depth if registered else None)
    unique=rim_candidates(frame,cues,registered)
    fused=[]
    for proposal in proposals:
        original=dict(proposal)
        x,y,w,h=proposal['roi']
        polygon=proposal.get('polygon') or [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
        original['rim_evidence']=support(polygon,cues)
        original['rim_method']='ML only / rim unconfirmed'
        matches=[c for c in unique if iou(c['roi'],proposal['roi'])>=0.55]
        if matches:
            candidate=max(matches,key=lambda c:c['score'])
            original.update(roi=candidate['roi'],polygon=candidate['polygon'],
                center=np.mean(np.array(candidate['polygon']),axis=0).tolist(),
                rim_evidence=candidate['evidence'],
                rim_method='ML + visible black rim' if candidate['evidence']['black']>=0.2 else 'ML + RGB + invalid-depth clue')
        fused.append(original)
    return fused,cues


def standalone_black_rims(frame,cues):
    """Unclassified geometric hypotheses; caller MUST apply metric size checks.

    Each rectangle gets both possible roles. Only a unique size-compatible role
    can survive resolve_geometric_roles. No neural probability is fabricated.
    """
    proposals=[]
    for index,candidate in enumerate(rim_candidates(frame,cues)):
        if candidate['evidence']['black']<0.2:
            continue
        x,y,w,h=candidate['roi']
        for role in ('source','destination'):
            proposals.append(dict(role=role,roi=list(candidate['roi']),
                polygon=candidate['polygon'],center=np.mean(candidate['polygon'],axis=0).tolist(),
                confidence=None,clipped=x<=2 or y<=2 or x+w>=frame.shape[1]-2 or y+h>=frame.shape[0]-2,
                rim_method='visible black rim + measured size',rim_evidence=candidate['evidence'],
                proposal_origin='black-rim',geometry_group=index))
    return proposals


def resolve_geometric_roles(proposals):
    selected=[d for d in proposals if d.get('proposal_origin')!='black-rim']
    groups={d['geometry_group'] for d in proposals if d.get('proposal_origin')=='black-rim'}
    for group in groups:
        choices=sorted([d for d in proposals if d.get('proposal_origin')=='black-rim'
                        and d['geometry_group']==group and 'dimension_error' in d],
                       key=lambda d:d['dimension_error'])
        if choices and (len(choices)==1 or choices[1]['dimension_error']-choices[0]['dimension_error']>0.15):
            candidate=choices[0]
            overlaps=[d for d in selected if d['role']==candidate['role'] and iou(d['roi'],candidate['roi'])>0.6]
            for old in overlaps:
                selected.remove(old)
            selected.append(candidate)
    return selected


def draw_cues(frame,cues):
    image=frame.copy()
    image[cues['dark_rim']>0]=(30,200,255)
    image[cv2.dilate(cues['joint'],np.ones((3,3),np.uint8))>0]=(0,255,0)
    return image
