"""Container rendering; display smoothing and stale tracks never alter targets."""
import cv2
import numpy as np


def dashed(image,polygon,ink):
    points=np.array(polygon,np.float32).reshape(-1,2)
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        length=float(np.linalg.norm(b-a))
        if length<1:
            continue
        for start in np.arange(0,length,15):
            p=a+(b-a)*start/length
            q=a+(b-a)*min(start+8,length)/length
            cv2.line(image,tuple(p.astype(int)),tuple(q.astype(int)),ink,2)


def draw_layout(frame,detections,confirmed,ghosts=None):
    image=frame.copy()
    ghosts=ghosts or []
    for d in list(ghosts)+list(detections):
        x,y,w,h=d['roi']
        polygon=d.get('display_polygon') or d.get('polygon') or [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
        stale=d.get('tracking_state')=='stale'
        tentative=d.get('tracking_state')=='tentative'
        ink=(0,160,255) if stale else ((40,190,40) if d['role']=='source' else (230,130,20))
        if stale or tentative:
            dashed(image,polygon,ink)
        else:
            cv2.polylines(image,[np.array(polygon,np.int32)],True,ink,2)
        prefix='SRC' if d['role']=='source' else 'BIN'
        identity=f" T{d['track_id']}" if d.get('track_id') else ''
        label=prefix+identity
        if stale:
            label+=f" STALE {d.get('age_seconds',0):.1f}s - NO PICK"
        elif tentative:
            label+=' new'
        if not stale and d['role']=='destination':
            label+=' | '+('empty' if d.get('empty_verified') else '/'.join(d.get('colors',[])) or 'unverified')
        if not stale and (d.get('rim_method','').startswith('ML +') or d.get('proposal_origin')=='black-rim'):
            label+=' | rim'
        (tw,th),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.45,1)
        tx=max(0,min(x,image.shape[1]-tw-4)); ty=max(th+5,y-6)
        cv2.rectangle(image,(tx,ty-th-3),(tx+tw+4,ty+baseline+2),(20,20,20),-1)
        cv2.putText(image,label,(tx+2,ty),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    fresh_sources=sum(d['role']=='source' for d in detections)
    fresh_bins=sum(d['role']=='destination' for d in detections)
    banner='CURRENT LAYOUT CONFIRMED' if confirmed else f'FRESH {fresh_sources}/2 sources, {fresh_bins}/4 bins | {len(ghosts)} stale outlines | NO PICK'
    cv2.rectangle(image,(0,0),(image.shape[1],34),(20,20,20),-1)
    cv2.putText(image,banner,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return image
