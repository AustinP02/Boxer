"""Pure color/depth perception and routing. No robot connections or commands."""
import cv2
import numpy as np
from block_detector import COLORS, classify_shape


def decode_depth(data):
    if len(data) < 24 or data[:8] != b'DEPTHMAP':
        raise ValueError('Expected Viam big-endian DEPTHMAP image')
    w = int.from_bytes(data[8:16], 'big')
    h = int.from_bytes(data[16:24], 'big')
    if not 0 < w*h <= 20000000 or len(data) != 24+2*w*h:
        raise ValueError('Invalid depth dimensions/payload')
    return np.frombuffer(data[24:], dtype='>u2').reshape(h, w).astype(np.float32)


def camera_point(pixel, depth_mm, intrinsics):
    u, v = pixel
    fx, fy, cx, cy = intrinsics
    if min(fx, fy, depth_mm) <= 0:
        raise ValueError('Invalid intrinsics/depth')
    return np.array([(u-cx)*depth_mm/fx, (v-cy)*depth_mm/fy, depth_mm])


def detect_source_items(frame, sources, config, depth=None, intrinsics=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    items = []
    height, width = frame.shape[:2]
    for index, source in enumerate(sources):
        x, y, w, h = source['roi']
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
            raise ValueError('Source region is outside the image; repeat setup')
        for color, (ranges, _) in COLORS.items():
            mask = np.zeros((height,width), np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, (lo,90,60), (hi,255,255))
            crop = np.zeros_like(mask)
            crop[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
            if source.get('polygon'):
                interior=np.zeros_like(mask)
                cv2.fillPoly(interior,[np.array(source['polygon'],np.int32)],255)
                interior=cv2.erode(interior,np.ones((9,9),np.uint8))
                crop &= interior
            crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < config['min_area_px']:
                    continue
                bx,by,bw,bh = cv2.boundingRect(contour)
                mask_item = np.zeros_like(mask)
                cv2.drawContours(mask_item,[contour],-1,255,-1)
                distance = cv2.distanceTransform(mask_item, cv2.DIST_L2,5)
                _, radius, _, center = cv2.minMaxLoc(distance)
                shape, _ = classify_shape(contour)
                item = dict(source=index, color=color, destination=color,
                            bbox=[bx,by,bw,bh], center=list(center),
                            area=float(cv2.contourArea(contour)), shape=shape,
                            rejected=[])
                margin = config['jaw_margin_px']
                if bx <= x+margin or by <= y+margin or bx+bw >= x+w-margin or by+bh >= y+h-margin:
                    item['rejected'].append('too close to tray boundary')
                if radius < 5:
                    item['rejected'].append('thin or fragmented region')
                if depth is not None:
                    if depth.shape != frame.shape[:2]:
                        raise ValueError('Depth/color image sizes differ; verify registration')
                    core = cv2.erode(mask_item,np.ones((5,5),np.uint8)) > 0
                    valid = core & np.isfinite(depth) & (depth>100) & (depth<2000)
                    values = depth[valid]
                    if len(values) < 25 or len(values)/max(np.count_nonzero(core),1) < 0.6:
                        item['rejected'].append('missing depth')
                    else:
                        spread = float(np.percentile(values,90)-np.percentile(values,10))
                        if spread > config['max_depth_spread_mm']:
                            item['rejected'].append('uneven depth / overlapping pieces')
                        u,v = center
                        local = depth[max(0,v-3):v+4,max(0,u-3):u+4]
                        local = local[np.isfinite(local) & (local>100) & (local<2000)]
                        if len(local) < 9:
                            item['rejected'].append('invalid depth at pick center')
                        else:
                            d = float(np.median(local))
                            item['camera_mm'] = camera_point(center,d,intrinsics).tolist()
                            (_, _), (rw,rh), angle = cv2.minAreaRect(contour)
                            dims = sorted([rw*d/intrinsics[0],rh*d/intrinsics[1]])
                            item['size_mm'] = dims
                            if dims[0] > config['max_grasp_width_mm'] or dims[1] > config['max_object_span_mm']:
                                item['rejected'].append('oversized / possibly merged objects')
                items.append(item)
    # Nearby connected regions are not independent grasp targets in this MVP.
    for i,item in enumerate(items):
        x,y,w,h = item['bbox']
        for j,other in enumerate(items):
            if i==j or item['source']!=other['source']:
                continue
            a,b,c,d = other['bbox']
            gap = config['jaw_margin_px']
            if x-gap < a+c and a-gap < x+w and y-gap < b+d and b-gap < y+h:
                item['rejected'].append('crowded: adjacent object in jaw clearance')
                break
    return items


def overlay(frame, sources, items):
    result=frame.copy()
    for item in items:
        x,y,w,h=item['bbox']
        ink=COLORS[item['color']][1]
        if item.get('instance_outline'):
            cv2.polylines(result,[np.array(item['instance_outline'],np.int32)],True,ink,2)
        else:
            cv2.rectangle(result,(x,y),(x+w,y+h),ink,1)
        label=f"{item['color']} piece {item.get('instance_id','?')}"
        if item['rejected']:
            label+=' blocked'
        cv2.putText(result,label,(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    return result


def container_contents(frame,container,config,depth=None,intrinsics=None,origin=None,basis=None):
    x,y,w,h=container['roi']
    mask=np.zeros(frame.shape[:2],np.uint8)
    if container.get('polygon'):
        cv2.fillPoly(mask,[np.array(container['polygon'],np.int32)],255)
    else:
        mask[y:y+h,x:x+w]=255
    mask=cv2.erode(mask,np.ones((15,15),np.uint8))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    colors=[]
    for color,(ranges,_) in COLORS.items():
        selected=np.zeros_like(mask)
        for lo,hi in ranges:
            selected |= cv2.inRange(hsv,(lo,90,60),(hi,255,255))
        selected &= mask
        contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if any(cv2.contourArea(c)>=config['min_area_px'] for c in contours):
            colors.append(color)
    result=dict(container,colors=colors,empty_verified=False)
    if len(colors)>1:
        result['status']='mixed colors; exclude'
        return result
    if colors:
        result['status']=colors[0]
    else:
        result['status']='unassigned; emptiness not verified'
    floor=config.get('destination_floor_z_mm')
    if depth is not None and floor is not None:
        ys,xs=np.where(mask>0)
        ds=depth[ys,xs]
        valid=np.isfinite(ds)&(ds>100)&(ds<2000)
        if len(ds)>100 and np.mean(valid)>0.85:
            samples=np.flatnonzero(valid)[::max(1,int(np.sum(valid)//300))]
            heights=[]
            for i in samples:
                world=origin+basis@camera_point((xs[i],ys[i]),float(ds[i]),intrinsics)
                heights.append(world[2])
            # An unknown/white object above the calibrated floor must not count as empty.
            if not colors and len(heights)>30 and np.mean(np.abs(np.array(heights)-floor)<6)>0.95:
                result['empty_verified']=True
                result['status']='empty / unassigned'
            # Locate genuinely clear floor space rather than dropping repeatedly
            # at the bin center or assuming a matching-color bin has capacity.
            floor_mask=np.zeros_like(mask)
            valid_indices=np.flatnonzero(valid)
            us=xs[valid_indices]; vs=ys[valid_indices]; dz=ds[valid_indices]
            fx,fy,cx,cy=intrinsics
            camera_xyz=np.column_stack(((us-cx)*dz/fx,(vs-cy)*dz/fy,dz))
            z_world=origin[2]+camera_xyz@basis[2,:]
            free_indices=valid_indices[np.abs(z_world-floor)<6]
            floor_mask[ys[free_indices],xs[free_indices]]=255
            floor_mask=cv2.morphologyEx(floor_mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            distance=cv2.distanceTransform(floor_mask,cv2.DIST_L2,5)
            _,radius,_,center=cv2.minMaxLoc(distance)
            if radius>0:
                u,v=center
                d=float(depth[v,u])
                result['free_center']=list(center)
                result['free_radius_mm']=float(radius*d/max(fx,fy))
    return result


def choose_destination(color,bins,required_radius_mm=None):
    matching=[b for b in bins if b['colors']==[color]]
    def fits(bin):
        return required_radius_mm is None or bin.get('free_radius_mm',0)>=required_radius_mm
    if matching:
        return next((b for b in matching if fits(b)),None)
    empty=[b for b in bins if b['empty_verified'] and not b['colors'] and fits(b)]
    return empty[0] if empty else None
