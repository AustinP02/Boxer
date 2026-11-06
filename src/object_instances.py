"""SAM2 piece masks seeded on colored foreground; color is assigned AFTER masks.

Seed points are not object detections. Multiple seeds on one piece are deduped.
This model may still miss or merge touching pieces; no color-blob fallback is used.
"""
from pathlib import Path
import cv2
import numpy as np
from block_detector import COLORS


def foreground_seeds(frame,sources,max_points=24,spacing=20):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    colored=np.zeros(frame.shape[:2],np.uint8)
    for ranges,_ in COLORS.values():
        for lo,hi in ranges:
            colored|=cv2.inRange(hsv,(lo,90,60),(hi,255,255))
    source_mask=np.zeros_like(colored)
    for source in sources:
        if source.get('polygon'):
            cv2.fillPoly(source_mask,[np.array(source['polygon'],np.int32)],255)
        else:
            x,y,w,h=source['roi']; source_mask[y:y+h,x:x+w]=255
    colored &= cv2.erode(source_mask,np.ones((9,9),np.uint8))
    distance=cv2.distanceTransform(colored,cv2.DIST_L2,5)
    ys,xs=np.where(distance>=4)
    if not len(xs):
        return []
    # A sparse spatial sample covers all colors instead of using one seed per
    # connected hue region. Farthest-point sampling covers touching pieces too.
    sample=np.arange(0,len(xs),max(1,len(xs)//12000))
    xy=np.column_stack((xs[sample],ys[sample])).astype(np.float32)
    first=int(np.argmax(distance[xy[:,1].astype(int),xy[:,0].astype(int)]))
    chosen=[xy[first]]
    nearest=np.sum((xy-chosen[0])**2,axis=1)
    for _ in range(max_points-1):
        index=int(np.argmax(nearest))
        if nearest[index]<spacing**2:
            break
        chosen.append(xy[index])
        nearest=np.minimum(nearest,np.sum((xy-chosen[-1])**2,axis=1))
    return [p.astype(int).tolist() for p in chosen]


def dedupe_masks(instances):
    ordered=sorted(instances,key=lambda x:x['quality'],reverse=True)
    kept=[]
    for item in ordered:
        mask=item['mask']
        duplicate=False
        for previous in kept:
            other=previous['mask']
            overlap=np.count_nonzero(mask&other)
            if overlap/max(1,np.count_nonzero(mask|other))>0.65 or overlap/max(1,min(np.count_nonzero(mask),np.count_nonzero(other)))>0.9:
                duplicate=True; break
        if not duplicate:
            kept.append(item)
    return kept


class ObjectInstanceDetector:
    def __init__(self,weights):
        from ultralytics import SAM
        self.model=SAM(str(weights))

    def detect(self,frame,sources):
        seeds=foreground_seeds(frame,sources)
        if not seeds:
            return []
        # Each point is a DIFFERENT prompt/object hypothesis, not a set of
        # positive points defining one merged mask.
        points=[[[x,y]] for x,y in seeds]
        result=self.model.predict(frame,points=points,labels=[[1] for _ in points],
            imgsz=1024,verbose=False,device='cpu')[0]
        if result.masks is None:
            return []
        masks=result.masks.data.cpu().numpy()
        qualities=result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))
        instances=[]
        source_mask=np.zeros(frame.shape[:2],np.uint8)
        for source in sources:
            if source.get('polygon'):
                cv2.fillPoly(source_mask,[np.array(source['polygon'],np.int32)],255)
            else:
                x,y,w,h=source['roi']; source_mask[y:y+h,x:x+w]=255
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        color_masks=[]
        for ranges,_ in COLORS.values():
            color_mask=np.zeros(frame.shape[:2],np.uint8)
            for lo,hi in ranges:
                color_mask|=cv2.inRange(hsv,(lo,90,60),(hi,255,255))
            color_masks.append(color_mask>0)
        for index,mask in enumerate(masks):
            mask=cv2.resize(mask, (frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)>0.5
            area=np.count_nonzero(mask)
            if qualities[index]<.75 or area<150 or area>frame.shape[0]*frame.shape[1]*.12:
                continue
            if np.count_nonzero(mask&(source_mask>0))/area<.85:
                continue
            counts=[np.count_nonzero(mask&color) for color in color_masks]
            if max(counts)/max(1,sum(counts))<.75 or max(counts)/area<.45:
                continue
            instances.append(dict(mask=mask,quality=float(qualities[index]),backend='SAM2'))
        return dedupe_masks(instances)
