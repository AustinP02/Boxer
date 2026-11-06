"""Local YOLOE visual-prompt container reacquisition. No robot actuation."""
import os
from pathlib import Path
import cv2
import numpy as np

from project_paths import MODELS_DIR, CACHE_DIR
ROOT = MODELS_DIR
os.environ.setdefault('YOLO_CONFIG_DIR',str(CACHE_DIR/'ultralytics'))


def build_profile(frame, sources, destinations):
    from ultralytics import YOLOE
    model=YOLOE(str(ROOT/'yoloe-26s-seg.pt'))
    boxes=[]
    for x,y,w,h in sources+destinations:
        boxes.append([x,y,x+w,y+h])
    prompts={'bboxes':np.asarray(boxes,np.float32),
             'cls':np.asarray([0]*len(sources)+[1]*len(destinations),np.int32)}
    model.predict(frame,refer_image=frame,visual_prompts=prompts,
                  imgsz=960,conf=0.35,verbose=False,device='cpu')
    model.save(str(ROOT/'container_profile.pt'))


def validate_layout(detections):
    detections=remove_group_boxes(detections)
    sources=sorted([d for d in detections if d['role']=='source'],key=lambda d:d['center'][0])
    destinations=sorted([d for d in detections if d['role']=='destination'],key=lambda d:d['center'][0])
    if len(sources)!=2 or len(destinations)!=4:
        raise ValueError(f'Layout uncertain: detected {len(sources)} sources / {len(destinations)} destinations; require 2 / 4')
    if any(d.get('clipped',False) for d in sources+destinations):
        raise ValueError('A container is cut off by the image boundary; do not pick')
    all_boxes=sources+destinations
    for i,a in enumerate(all_boxes):
        x,y,w,h=a['roi']
        for b in all_boxes[i+1:]:
            bx,by,bw,bh=b['roi']
            overlap=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
            if overlap/max(1,min(w*h,bw*bh))>0.15:
                raise ValueError('Containers overlap or duplicate detection; do not pick')
    for i,detection in enumerate(destinations):
        detection['id']=f"bin-T{detection['track_id']}" if detection.get('track_id') else f'bin-{i+1}'
    return dict(sources=sources,destinations=destinations,complete=True)


def preview_layout(detections):
    detections=remove_group_boxes(detections)
    sources=sorted([d for d in detections if d['role']=='source'],key=lambda d:d['center'][0])
    bins=sorted([d for d in detections if d['role']=='destination'],key=lambda d:d['center'][0])
    if len(sources)>2 or len(bins)>4:
        raise ValueError(f'Uncertain preview: {len(sources)} sources, {len(bins)} destinations')
    for i,detection in enumerate(bins):
        detection['id']=f"bin-T{detection['track_id']}" if detection.get('track_id') else f'bin-{i+1}'
    return dict(sources=sources,destinations=bins,complete=False)


def remove_group_boxes(detections):
    """Reject a merged proposal enclosing two separate same-role proposals."""
    kept=[]
    for i,a in enumerate(detections):
        x,y,w,h=a['roi']; contained=[]
        for j,b in enumerate(detections):
            if i==j or a['role']!=b['role']:
                continue
            bx,by,bw,bh=b['roi']
            intersection=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
            if 0.2*w*h < bw*bh < 0.8*w*h and intersection/(bw*bh)>0.85:
                contained.append(b)
        group=False
        for j,b in enumerate(contained):
            for c in contained[j+1:]:
                bx,by,bw,bh=b['roi']; cx,cy,cw,ch=c['roi']
                overlap=max(0,min(bx+bw,cx+cw)-max(bx,cx))*max(0,min(by+bh,cy+ch)-max(by,cy))
                if overlap/min(bw*bh,cw*ch)<0.15:
                    group=True
        if not group:
            kept.append(a)
    return kept


class ContainerDetector:
    def __init__(self,profile=None):
        from ultralytics import YOLOE
        profile=Path(profile or ROOT/'container_profile.pt')
        if not profile.exists():
            raise ValueError('Run setup to give YOLOE container examples first')
        self.model=YOLOE(str(profile))

    def detect(self,frame,require_full=True):
        result=self.model.predict(frame,imgsz=960,conf=0.35,iou=0.45,
                                  verbose=False,device='cpu')[0]
        detections=[]
        if result.boxes is None:
            return validate_layout([]) if require_full else []
        for index,box in enumerate(result.boxes):
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy().tolist()
            x1,y1=max(0,int(x1)),max(0,int(y1))
            x2,y2=min(frame.shape[1],int(x2)),min(frame.shape[0],int(y2))
            if min(x2-x1,y2-y1)<20:
                continue
            polygon=None
            if result.masks is not None and len(result.masks.xy)>index:
                raw=result.masks.xy[index]
                if len(raw)>=4:
                    polygon=cv2.convexHull(raw.astype(np.float32)).reshape(-1,2).tolist()
            detections.append(dict(role='source' if int(box.cls.item())==0 else 'destination',
                roi=[x1,y1,x2-x1,y2-y1],center=[(x1+x2)/2,(y1+y2)/2],
                polygon=polygon,confidence=float(box.conf.item()),
                clipped=x1<=2 or y1<=2 or x2>=frame.shape[1]-2 or y2>=frame.shape[0]-2))
        return validate_layout(detections) if require_full else detections
