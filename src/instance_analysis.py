"""Measure individual segmentation masks. Never merges them by their hue."""
import cv2
import numpy as np
from block_detector import COLORS
from sort_vision import camera_point


def analyze_instances(frame,sources,config,instances,depth=None,intrinsics=None):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    masks={}
    for color,(ranges,_) in COLORS.items():
        mask=np.zeros(frame.shape[:2],np.uint8)
        for lo,hi in ranges:
            mask|=cv2.inRange(hsv,(lo,90,60),(hi,255,255))
        masks[color]=mask>0
    regions=[]
    for source in sources:
        region=np.zeros(frame.shape[:2],np.uint8)
        if source.get('polygon'):
            cv2.fillPoly(region,[np.array(source['polygon'],np.int32)],255)
        else:
            x,y,w,h=source['roi']; region[y:y+h,x:x+w]=255
        regions.append(region>0)
    result=[]
    for number,instance in enumerate(instances):
        mask=instance['mask']
        area=np.count_nonzero(mask)
        if area<config['min_area_px'] or mask.shape!=frame.shape[:2]:
            continue
        cover=[np.count_nonzero(mask&r)/area for r in regions]
        if not cover or max(cover)<.85:
            continue
        source=int(np.argmax(cover))
        counts={color:np.count_nonzero(mask&region) for color,region in masks.items()}
        color=max(counts,key=counts.get)
        if counts[color]/max(sum(counts.values()),1)<.75 or counts[color]/area<.45:
            continue
        contours,_=cv2.findContours(mask.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(contour)
        selected=(mask&masks[color]).astype(np.uint8)*255
        _,radius,_,center=cv2.minMaxLoc(cv2.distanceTransform(selected,cv2.DIST_L2,5))
        outline=cv2.approxPolyDP(contour,1.5,True).reshape(-1,2).tolist()
        item=dict(source=source,color=color,destination=color,bbox=[x,y,w,h],center=list(center),
            area=float(area),shape='piece-mask hypothesis',instance_id=number+1,
            instance_outline=outline,mask_quality=instance['quality'],backend=instance['backend'],rejected=[])
        if sum(cv2.contourArea(c) for c in contours)>cv2.contourArea(contour)*1.15:
            item['rejected'].append('disconnected mask / possibly merged objects')
        if radius<5:
            item['rejected'].append('thin visible surface')
        source_distance=cv2.distanceTransform(regions[source].astype(np.uint8)*255,cv2.DIST_L2,5)
        if np.min(source_distance[mask])<config['jaw_margin_px']:
            item['rejected'].append('too close to tray boundary')
        if depth is not None:
            if depth.shape!=mask.shape:
                raise ValueError('Depth/color size mismatch')
            core=cv2.erode(selected,np.ones((5,5),np.uint8))>0
            valid=core&np.isfinite(depth)&(depth>100)&(depth<2000)
            values=depth[valid]
            if len(values)<25 or len(values)/max(1,np.count_nonzero(core))<.6:
                item['rejected'].append('missing depth')
            else:
                if np.percentile(values,90)-np.percentile(values,10)>config['max_depth_spread_mm']:
                    item['rejected'].append('uneven depth / occlusion')
                u,v=center
                patch=depth[max(0,v-3):v+4,max(0,u-3):u+4]
                good=patch[np.isfinite(patch)&(patch>100)&(patch<2000)]
                if len(good)<9:
                    item['rejected'].append('invalid grasp depth')
                else:
                    distance=float(np.median(good))
                    item['camera_mm']=camera_point(center,distance,intrinsics).tolist()
                    _,(rw,rh),_=cv2.minAreaRect(contour)
                    item['size_mm']=sorted([rw*distance/intrinsics[0],rh*distance/intrinsics[1]])
                    if item['size_mm'][0]>config['max_grasp_width_mm'] or item['size_mm'][1]>config['max_object_span_mm']:
                        item['rejected'].append('oversized / possibly merged mask')
        result.append(item)
    for i,item in enumerate(result):
        x,y,w,h=item['bbox']; gap=config['jaw_margin_px']
        for j,other in enumerate(result):
            if i==j or item['source']!=other['source']:
                continue
            a,b,c,d=other['bbox']
            if x-gap<a+c and a-gap<x+w and y-gap<b+d and b-gap<y+h:
                item['rejected'].append('crowded: adjacent instance in jaw clearance'); break
    return result
