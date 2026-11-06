"""Two-source, four-destination calibrated color sorting prototype.

scan --move: move to supplied scan pose and observe.
setup: mark two source interiors on latest scan image (no motion).
teach NAME: record current gripper pose (no motion).
plan: observe in place and show proposed routes (no motion).
run --execute: move to scan, pick ONE eligible object, then rescan.
"""
import argparse
import asyncio
import json
import math
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame, WorldState, GeometriesInFrame, Geometry, RectangularPrism, Vector3, Transform
from sample_scan import inspect_setup
from sort_vision import decode_depth, detect_source_items, overlay, container_contents, choose_destination
from container_detector import ContainerDetector, build_profile, validate_layout, preview_layout
from rim_fusion import fuse_rims, standalone_black_rims, resolve_geometric_roles
from container_tracking import ContainerTracker, clean_candidates
from object_instances import ObjectInstanceDetector
from instance_analysis import analyze_instances

from project_paths import PROJECT_ROOT, CONFIG_DIR, MODELS_DIR, OUTPUT_DIR
ROOT = PROJECT_ROOT
CONFIG = CONFIG_DIR/'sort_config.json'
ORDER = ['red','yellow','blue','green']


from tracking_render import draw_layout


def ruler_check(args):
    report=json.loads((OUTPUT_DIR/'sort_runs/latest.json').read_text())
    if not report.get('has_aligned_depth'):
        raise ValueError('Verify aligned depth and capture a new scan before checking the ruler')
    frame=cv2.imread(str(OUTPUT_DIR/'sort_runs/latest.jpg'))
    depth=np.load(OUTPUT_DIR/'sort_runs/latest_depth.npy')
    pixels=[]
    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(pixels)<2:
            pixels.append((x,y))
            cv2.circle(frame,(x,y),5,(0,0,255),-1)
    try:
        cv2.namedWindow('ruler check'); cv2.setMouseCallback('ruler check',click)
        print(f'Click two ruler tick marks {args.mm} mm apart. Q cancels.')
        while len(pixels)<2:
            cv2.imshow('ruler check',frame)
            if cv2.waitKey(20)&0xff==ord('q'):
                return
    finally:
        cv2.destroyAllWindows()
    from sort_vision import camera_point
    points=[]
    for u,v in pixels:
        values=depth[max(0,v-2):v+3,max(0,u-2):u+3]
        values=values[np.isfinite(values)&(values>100)&(values<2000)]
        if len(values)<9:
            raise ValueError('Insufficient depth at ruler tick')
        points.append(camera_point((u,v),float(np.median(values)),report['intrinsics']))
    measured=float(np.linalg.norm(points[1]-points[0]))
    print(f'Depth measurement: {measured:.2f} mm; ruler: {args.mm:.2f} mm; error: {100*(measured/args.mm-1):+.2f}%')
    print('This checks scale, not the full hand-eye calibration. No robot movement issued.')


def save_config(config):
    temp = CONFIG.with_suffix('.tmp')
    temp.write_text(json.dumps(config,indent=2))
    temp.replace(CONFIG)


def pose_dict(pose):
    return {name:float(getattr(pose,name)) for name in ('x','y','z','o_x','o_y','o_z','theta')}


def check_ready(config):
    missing = []
    if len(config['sources'])!=2:
        missing.append('setup: mark both source interiors')
    for i,source in enumerate(config['sources']):
        if source.get('rim_z_mm') is None:
            missing.append(f'teach source-{i+1}-rim')
    for field in ('destination_release_z_mm','destination_floor_z_mm'):
        if config.get(field) is None:
            missing.append(field)
    for key in ('pick_orientation','transit_z_mm','grasp_offset_mm'):
        if config.get(key) is None:
            missing.append(key)
    if not config['depth_aligned_verified']:
        missing.append('verify color/depth registration; calibrate --aligned-depth')
    if not config.get('motion_commissioned'):
        missing.append('motion commissioning: verify localization and grasp/drop geometry first')
    if not config.get('shared_bin_heights_verified'):
        missing.append('verify all four destinations have same rim/floor heights')
    if missing:
        raise ValueError('Calibration needed: '+ '; '.join(missing))
    numeric=[config['destination_release_z_mm'],config['destination_floor_z_mm'],
             config['transit_z_mm'],config['grasp_offset_mm'],*config['pick_orientation'].values(),
             *[s['rim_z_mm'] for s in config['sources']]]
    if not all(math.isfinite(v) for v in numeric):
        raise ValueError('Nonfinite calibration values')
    if abs(config['sources'][0]['rim_z_mm']-config['sources'][1]['rim_z_mm'])>3:
        raise ValueError('Different source heights require persistent container identity/markers')
    if config['destination_release_z_mm']<=config['destination_floor_z_mm']+20:
        raise ValueError('Release height is not above the destination floor/rim')
    if not -30 <= config['grasp_offset_mm'] <= 10:
        raise ValueError('Grasp offset must be in -30..10 mm')
    if config['pick_orientation']['o_z'] > -0.97:
        raise ValueError('Teach a near-vertical downward gripper orientation')
    floor = max([s['rim_z_mm']+150 for s in config['sources']] +
                [config['destination_release_z_mm']+80])
    if config['transit_z_mm'] < floor:
        raise ValueError(f'Taught transit Z must be at least {floor:.1f} mm for clearance')


def setup(config):
    frame = cv2.imread(str(OUTPUT_DIR/'sort_runs/latest.jpg'))
    if frame is None:
        raise ValueError('Run scan --move first')
    metadata = json.loads((OUTPUT_DIR/'sort_runs/latest.json').read_text())
    sources = []
    destinations=[]
    try:
        for i in range(2):
            print(f'Select entire SOURCE {i+1} container as an ML example. Enter confirms; C cancels.')
            x,y,w,h = map(int,cv2.selectROI('source setup',frame,showCrosshair=True))
            if min(w,h)<30:
                raise ValueError('Source selection cancelled or too small; nothing saved')
            sources.append(dict(roi=[x,y,w,h],rim_z_mm=None))
        for i in range(4):
            print(f'Select entire destination BIN {i+1}, initially left to right. Colors are assigned by contents.')
            box=list(map(int,cv2.selectROI('source setup',frame,showCrosshair=True)))
            if min(box[2:])<20:
                raise ValueError('Destination example cancelled or too small')
            destinations.append(box)
    finally:
        cv2.destroyAllWindows()
    a,b = sources[0]['roi'],sources[1]['roi']
    if a[0]<b[0]+b[2] and b[0]<a[0]+a[2] and a[1]<b[1]+b[3] and b[1]<a[1]+a[3]:
        raise ValueError('Source interiors overlap; repeat setup')
    build_profile(frame,[s['roi'] for s in sources],destinations)
    config['sources'] = sources
    config['destinations']={}
    config['destination_reference']={}
    config['motion_commissioned']=False
    config['setup_camera_pose'] = metadata['camera_pose']
    config['image_size'] = [frame.shape[1],frame.shape[0]]
    save_config(config)
    cv2.imwrite(str(MODELS_DIR/'container_reference.jpg'),frame)
    print('Saved YOLOE container examples. Run watch and verify 2 sources / 4 destinations before teaching.')


async def camera_basis(machine):
    positions = []
    for xyz in ((0,0,0),(1,0,0),(0,1,0),(0,0,1)):
        p = await machine.transform_pose(PoseInFrame(reference_frame='cam',pose=Pose(
            x=xyz[0],y=xyz[1],z=xyz[2])), 'world')
        positions.append(np.array([p.pose.x,p.pose.y,p.pose.z]))
    return positions[0], np.column_stack([p-positions[0] for p in positions[1:]])


async def snapshot(machine, camera, motion, config, detector=None,allow_partial=False):
    # All pixel regions belong to the supplied fixed scan view.
    current = await Arm.from_robot(machine,'arm').get_end_position(timeout=15)
    target = Pose(**config['scan_pose'])
    if not allow_partial and np.linalg.norm(np.array([current.x-target.x,current.y-target.y,current.z-target.z]))>2:
        raise ValueError('Arm is not at scan position; run scan --move')
    a=np.array([current.o_x,current.o_y,current.o_z]); b=np.array([target.o_x,target.o_y,target.o_z])
    if not allow_partial and (np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)) < math.cos(math.radians(1)) or abs(current.theta-target.theta)>1):
        raise ValueError('Arm orientation differs from scan view')
    cam_pose = (await motion.get_pose('cam','world',timeout=15)).pose
    if not allow_partial and config.get('setup_camera_pose'):
        old = config['setup_camera_pose']
        if np.linalg.norm(np.array([getattr(cam_pose,k)-old[k] for k in ('x','y','z')]))>3:
            raise ValueError('Camera mount/view changed; repeat setup')
        if abs(cam_pose.theta-old['theta'])>1 or np.linalg.norm(np.array([getattr(cam_pose,k)-old[k] for k in ('o_x','o_y','o_z')]))>0.02:
            raise ValueError('Camera rotation changed; repeat setup')
    images, _ = await camera.get_images(timeout=20)
    color = next(i for i in images if i.name=='color')
    frame = cv2.imdecode(np.frombuffer(color.data,np.uint8),cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Cannot decode color frame')
    if config.get('image_size') and config['image_size']!=[frame.shape[1],frame.shape[0]]:
        raise ValueError('Image resolution changed; repeat source setup')
    depth = None
    props = await camera.get_properties(timeout=15)
    p=props.intrinsic_parameters
    intrinsic=(p.focal_x_px,p.focal_y_px,p.center_x_px,p.center_y_px)
    if min(intrinsic[:2])<=0:
        raise ValueError('Camera has no usable intrinsics')
    if config['depth_aligned_verified']:
        raw = next(i for i in images if i.name=='depth')
        depth = decode_depth(raw.data)
        props = await camera.get_properties(timeout=15)
        p = props.intrinsic_parameters
        if (p.width_px,p.height_px)!=(frame.shape[1],frame.shape[0]):
            raise ValueError('Intrinsics do not match color resolution')
        if any(abs(v)>1e-8 for v in props.distortion_parameters.parameters):
            raise ValueError('Nonzero lens distortion requires calibrated undistortion support')
        intrinsic = (p.focal_x_px,p.focal_y_px,p.center_x_px,p.center_y_px)
    layout=None
    sources=config['sources']
    if detector is not None:
        raw=await asyncio.to_thread(detector.detect,frame,False)
        ml_count=len(raw)
        raw,cues=fuse_rims(frame,raw,depth,registered=config['depth_aligned_verified'])
        raw.extend(standalone_black_rims(frame,cues))
        all_proposals=list(raw)
        origin,basis=await camera_basis(machine)
        raw=resolve_geometric_roles(dimension_filter(raw,config,intrinsic,origin,basis))
        tracker=config.get('_container_tracker')
        if tracker is not None:
            context=(frame.shape[1],frame.shape[0],*[getattr(cam_pose,k) for k in ('x','y','z','o_x','o_y','o_z','theta')])
            raw,ghosts,tracking_rejections=tracker.update(raw,context=context)
        else:
            raw,tracking_rejections=clean_candidates(raw)
            ghosts=[]
        config['_tracking_ghosts']=ghosts
        config['_tracking_rejections']=tracking_rejections
        rejected=[d for d in all_proposals if d.get('proposal_origin')!='black-rim' and not any(d is kept for kept in raw)]
        config['_detection_debug']=dict(ml_proposals=ml_count,accepted=len(raw),rejected=rejected)
        try:
            layout=preview_layout(raw) if allow_partial else validate_layout(raw)
        except ValueError:
            folder=OUTPUT_DIR/'sort_runs'; folder.mkdir(exist_ok=True)
            cv2.imwrite(str(folder/'latest_overlay.jpg'),draw_layout(frame,raw,confirmed=False))
            raise
        sources=[]
        for i,current_source in enumerate(layout['sources']):
            updated=dict(config['sources'][i] if i<len(config['sources']) else {'rim_z_mm':None})
            updated.update(roi=current_source['roi'],polygon=current_source['polygon'])
            sources.append(updated)
    object_detector=config.get('_object_detector')
    if object_detector is not None and sources:
        instances=await asyncio.to_thread(object_detector.detect,frame,sources)
        items=analyze_instances(frame,sources,config,instances,depth,intrinsic)
        config['_object_diagnostics']=dict(backend='SAM2 mask hypotheses',masks=len(instances),items=len(items))
    else:
        items=[]
        config['_object_diagnostics']=dict(backend='disabled: container-only preview',masks=0,items=0)
    origin,basis = await camera_basis(machine)
    for item in items:
        if 'camera_mm' in item:
            item['world_mm'] = (origin+basis@np.array(item['camera_mm'])).tolist()
        else:
            item['rejected'].append('3D localization unavailable')
    end = await motion.get_pose('cam','world',timeout=15)
    if np.linalg.norm(np.array([getattr(end.pose,k)-getattr(cam_pose,k) for k in ('x','y','z')]))>1:
        raise ValueError('Camera moved while acquiring scan')
    folder = OUTPUT_DIR/'sort_runs'
    folder.mkdir(exist_ok=True)
    if layout:
        layout['destinations']=[container_contents(frame,d,config,depth,intrinsic,origin,basis)
                                for d in layout['destinations']]
        for item in items:
            radius=math.hypot(*item['size_mm'])/2+5 if 'size_mm' in item else None
            destination=choose_destination(item['color'],layout['destinations'],radius)
            item['destination_bin']=destination['id'] if destination else None
            if destination is None:
                item['rejected'].append('no matching-color or verified-empty destination')
    report = dict(time=datetime.now().isoformat(),camera_pose=pose_dict(cam_pose),items=items,
                  layout=layout,origin=origin.tolist(),basis=basis.tolist(),intrinsics=intrinsic,
                  has_aligned_depth=depth is not None,detection_debug=config.get('_detection_debug',{}),
                  stale_tracks=config.get('_tracking_ghosts',[]),tracking_rejections=config.get('_tracking_rejections',[]),
                  object_instances=config.get('_object_diagnostics',{}))
    annotated=overlay(frame,sources,items)
    debug=config.get('_detection_debug',{})
    for rejected in (debug.get('rejected',[]) if config.get('_show_rejected',False) else []):
        x,y,w,h=rejected['roi']
        cv2.rectangle(annotated,(x,y),(x+w,y+h),(130,130,130),1)
        cv2.putText(annotated,'ML rejected: size/shape',(x,max(20,y-7)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(90,90,90),1)
    if layout:
        annotated=draw_layout(annotated,layout['sources']+layout['destinations'],confirmed=layout['complete'],ghosts=config.get('_tracking_ghosts',[]))
    for path,img in ((folder/'latest.jpg',frame),(folder/'latest_overlay.jpg',annotated)):
        if not cv2.imwrite(str(path),img):
            raise IOError('Cannot save scan image')
    (folder/'latest.json').write_text(json.dumps(report,indent=2))
    if depth is not None:
        np.save(folder/'latest_depth.npy',depth)
    config['_current_layout']=layout
    config['_current_sources']=sources
    return items,origin,basis,intrinsic


async def teach(args, config, motion):
    pose = (await motion.get_pose('gripper','world',timeout=15)).pose
    name = args.name
    if name=='destination':
        config['destination_release_z_mm']=pose.z
        height=config.get('container_dimensions_mm',{}).get('destination',[0,0,0])[2]
        if height:
            config['destination_floor_z_mm']=pose.z-config['release_above_rim_mm']-height
    elif name=='destination-floor':
        config['destination_floor_z_mm']=pose.z
    elif name=='orientation':
        if pose.o_z > -0.97:
            raise ValueError('Gripper must be pointing vertically downward')
        config['pick_orientation'] = {k:getattr(pose,k) for k in ('o_x','o_y','o_z','theta')}
    elif name=='transit':
        config['transit_z_mm'] = pose.z
    else:
        index = int(name.split('-')[1])-1
        if len(config['sources'])!=2:
            report=json.loads((OUTPUT_DIR/'sort_runs/latest.json').read_text())
            if not report.get('layout') or len(report['layout']['sources'])!=2:
                raise ValueError('Run ML watch/plan before teaching rim heights')
            config['sources']=[dict(roi=d['roi'],rim_z_mm=None) for d in report['layout']['sources']]
        config['sources'][index]['rim_z_mm'] = pose.z
    save_config(config)
    print(f'Saved {name}: {pose_dict(pose)} (no movement)')


def clone_pose(pose, **changes):
    values=pose_dict(pose); values.update(changes)
    return Pose(**values)


def ray_plane(pixel,z,intrinsic,origin,basis):
    from sort_vision import camera_point
    ray=basis@camera_point(pixel,1,intrinsic)
    if abs(ray[2])<1e-8 or (z-origin[2])/ray[2]<=0:
        raise ValueError('Tray plane is not in front of camera')
    return origin+ray*((z-origin[2])/ray[2])


def dimension_filter(detections,config,intrinsic,origin,basis):
    """Use measured inside sizes as loose proposals checks, never motion calibration.

    Estimated rim height here assumes bins resting on the table; unknown outer
    rim/bottom thickness is allowed by a wide tolerance. Physical moves still
    require taught rim/floor heights.
    """
    dimensions=config.get('container_dimensions_mm')
    if not dimensions or config.get('_table_top_z_mm') is None:
        return detections
    kept=[]
    for detection in detections:
        if detection.get('clipped'):
            detection['rejection_reason']='partial / clipped'
            continue
        role=detection['role']; length,width,height=dimensions[role]
        x,y,w,h=detection['roi']
        polygon=detection.get('polygon') or [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
        try:
            corners=cv2.boxPoints(cv2.minAreaRect(np.array(polygon,np.float32)))
            points=[ray_plane(p,config['_table_top_z_mm']+height,intrinsic,origin,basis) for p in corners]
            sizes=sorted([float(np.linalg.norm(points[1]-points[0])),float(np.linalg.norm(points[2]-points[1]))])
        except ValueError:
            continue
        detection['estimated_span_mm']=sizes
        detection['dimension_error']=abs(math.log(max(sizes[0],1e-5)/width))+abs(math.log(max(sizes[1],1e-5)/length))
        if 0.55*width<=sizes[0]<=1.4*width and 0.55*length<=sizes[1]<=1.3*length:
            kept.append(detection)
    return kept


def tray_walls(config,origin,basis,intrinsic):
    geometries=[]
    for source in config.get('_current_sources',config['sources']):
        x,y,w,h=source['roi']; z=source['rim_z_mm']
        corners=[ray_plane(p,z,intrinsic,origin,basis) for p in
                 ((x,y),(x+w,y),(x+w,y+h),(x,y+h))]
        # Conservative 150 mm-tall walls descend from the taught rim.
        for a,b in zip(corners,corners[1:]+corners[:1]):
            center=(a+b)/2
            geometries.append(Geometry(center=Pose(x=center[0],y=center[1],z=z-75,
                o_z=1,theta=math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))),
                box=RectangularPrism(dims_mm=Vector3(x=float(np.linalg.norm(b-a)),y=6,z=150))))
    return GeometriesInFrame(reference_frame='world',geometries=geometries)


async def pick_one(machine,arm,gripper,motion,world,config,item,origin,basis,intrinsic):
    x,y,z=item['world_mm']
    transit=config['transit_z_mm']
    orientation=config['pick_orientation']
    pick=Pose(x=x,y=y,z=z+config['grasp_offset_mm'],**orientation)
    if pick.z < config['_table_top_z_mm']+5:
        raise ValueError('Pick is at/below known table top plus minimum margin')
    source=config['sources'][item['source']]
    approach=clone_pose(pick,z=max(z+config['approach_mm'],source['rim_z_mm']+50))
    if transit < approach.z+20:
        raise ValueError('Transit height too low for object approach')
    required_radius=math.hypot(*item['size_mm'])/2+5
    destination=choose_destination(item['color'],config['_current_layout']['destinations'],required_radius)
    if destination is None:
        raise RuntimeError('Destination no longer available')
    rim_z=config['destination_release_z_mm']-config['release_above_rim_mm']
    now=ray_plane(destination['free_center'],config['destination_floor_z_mm'],intrinsic,origin,basis)
    drop=Pose(x=float(now[0]),y=float(now[1]),z=config['destination_release_z_mm'],**orientation)
    local=WorldState(); local.CopyFrom(world)
    local.obstacles.append(tray_walls(config,origin,basis,intrinsic))

    async def move(pose):
        ok=await motion.move(component_name='gripper',
            destination=PoseInFrame(reference_frame='world',pose=pose),
            world_state=local,extra={'motion_profile':'linear'},timeout=60)
        if not ok:
            raise RuntimeError('Motion planner did not confirm success')

    current=(await motion.get_pose('gripper','world')).pose
    await move(clone_pose(current,z=transit))
    await move(clone_pose(approach,z=transit))
    await gripper.open(timeout=15)
    await move(approach)
    # Re-observe at approach height while the offset wrist camera can still see
    # the target. At the final grasp height the target can leave its field of view.
    current_origin,current_basis=await camera_basis(machine)
    point=current_basis.T@(np.array(item['world_mm'])-current_origin)
    if point[2]<=0:
        raise RuntimeError('Target is no longer in front of the camera')
    u=int(intrinsic[0]*point[0]/point[2]+intrinsic[2])
    v=int(intrinsic[1]*point[1]/point[2]+intrinsic[3])
    camera=Camera.from_robot(machine,'cam')
    frames,_=await camera.get_images(timeout=20)
    color_frame=cv2.imdecode(np.frombuffer(next(f for f in frames if f.name=='color').data,np.uint8),cv2.IMREAD_COLOR)
    depth_frame=decode_depth(next(f for f in frames if f.name=='depth').data)
    if color_frame is None or not 3<=u<color_frame.shape[1]-3 or not 3<=v<color_frame.shape[0]-3:
        raise RuntimeError('Cannot see target before grasp; no gripper close issued')
    from block_detector import COLORS
    hsv=cv2.cvtColor(color_frame[v-2:v+3,u-2:u+3],cv2.COLOR_BGR2HSV)
    mask=np.zeros(hsv.shape[:2],np.uint8)
    for lo,hi in COLORS[item['color']][0]:
        mask|=cv2.inRange(hsv,(lo,90,60),(hi,255,255))
    measured=depth_frame[v-2:v+3,u-2:u+3]
    good=measured[(measured>100)&(measured<2000)]
    if np.mean(mask>0)<0.7 or len(good)<15 or abs(float(np.median(good))-point[2])>5:
        raise RuntimeError('Target appearance/depth changed at approach; stopped without closing')
    await move(pick)
    if not await gripper.grab(timeout=20):
        raise RuntimeError('Gripper reported no object; stopped without transfer or release')
    # Conservative held-object envelope, attached to gripper during transfer.
    held_height=z-(config['_table_top_z_mm']+3)
    if not 5<held_height<=100:
        raise RuntimeError('Held-object height estimate outside commissioned range; stopped while holding')
    relative=await machine.transform_pose(PoseInFrame(reference_frame='world',
        pose=Pose(x=x,y=y,z=z-held_height/2,o_z=1)),'gripper')
    span=max(120.0,math.hypot(*item['size_mm'])+20)
    local.transforms.append(Transform(reference_frame='sort-held-object',
        pose_in_observer_frame=PoseInFrame(reference_frame='gripper',pose=relative.pose),
        physical_object=Geometry(center=Pose(o_z=1),box=RectangularPrism(
            dims_mm=Vector3(x=span,y=span,z=held_height)))))
    await move(approach)
    await move(clone_pose(approach,z=transit))
    holding=await gripper.is_holding_something(timeout=15)
    if not holding.is_holding_something:
        raise RuntimeError('Object holding check failed; stopped before destination')
    # Reacquire all containers while holding, so a pre-pick destination cannot
    # silently become the drop target after it has been moved.
    if not await motion.move(component_name='arm',destination=PoseInFrame(reference_frame='world',
            pose=Pose(**config['scan_pose'])),world_state=local,timeout=60):
        raise RuntimeError('Cannot return to scan view while holding')
    _,origin,basis,intrinsic=await snapshot(machine,camera,motion,config,config['_detector'])
    destination=choose_destination(item['color'],config['_current_layout']['destinations'],required_radius)
    if destination is None:
        raise RuntimeError('No verified matching/empty destination after rescan; stopped while holding')
    now=ray_plane(destination['free_center'],config['destination_floor_z_mm'],intrinsic,origin,basis)
    drop.x=float(now[0]); drop.y=float(now[1])
    # Keep the held object's bottom above the tallest modeled bin rim when releasing.
    drop.z=max(drop.z,rim_z+held_height+20)
    if drop.z+20>transit:
        raise RuntimeError('Insufficient transit height for release clearance')
    local.obstacles.pop()  # replace the old source-wall geometry after reacquisition
    local.obstacles.append(tray_walls(config,origin,basis,intrinsic))
    await move(clone_pose(drop,z=transit))
    await move(drop)
    await gripper.open(timeout=15)
    if (await gripper.is_holding_something(timeout=15)).is_holding_something:
        raise RuntimeError('Release not confirmed; stopped at destination')
    local.transforms.clear()
    await move(clone_pose(drop,z=transit))


async def run(args,config):
    if args.command=='run':
        check_ready(config)
        if not args.execute:
            raise ValueError('Use plan for observation, or run --execute for physical sorting')
    load_dotenv(ROOT/'.env')
    machine=await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])),30)
    attempted=False
    try:
        motion=MotionClient.from_robot(machine,'builtin')
        if args.command=='teach':
            # Teaching reads the configured gripper frame only. An unavailable
            # camera must not prevent recording a manually positioned tool.
            await teach(args,config,motion); return
        arm=Arm.from_robot(machine,'arm')
        gripper=Gripper.from_robot(machine,'gripper')
        camera=Camera.from_robot(machine,'cam')
        world,info=await inspect_setup(machine,motion)
        table_pose=(await motion.get_pose('table','world')).pose
        table_geometry=next(o for o in world.obstacles if o.reference_frame=='table').geometries[0]
        if abs(table_pose.o_z-1)>0.01 or not table_geometry.HasField('box'):
            raise ValueError('Expected horizontal box table geometry')
        config['_table_top_z_mm']=table_pose.z+table_geometry.center.z+table_geometry.box.dims_mm.z/2
        detector=ContainerDetector() if (MODELS_DIR/'container_profile.pt').exists() else None
        config['_detector']=detector
        with_instances=args.command in ('run','plan') or getattr(args,'instances',False)
        config['_object_detector']=ObjectInstanceDetector(MODELS_DIR/'sam2.1_t.pt') if with_instances else None
        config['_container_tracker']=ContainerTracker(hold_seconds=6 if with_instances else 3) if args.command=='watch' else None
        config['_show_rejected']=getattr(args,'debug',False)
        if args.command in ('watch','plan','run') and detector is None:
            raise ValueError('Run scan --move, then setup to create the ML container profile')
        # Supplied UI scan pose is arm-base relative. This installation has an
        # identity arm mounting frame, verified again before applying it in world.
        frames=await machine.get_frame_system_config()
        base=next(f.frame.pose_in_observer_frame for f in frames if f.frame.reference_frame=='arm')
        if base.reference_frame!='world' or any(abs(getattr(base.pose,k))>1e-5 for k in ('x','y','z','theta')) or abs(base.pose.o_z-1)>1e-5:
            raise ValueError('Arm base is no longer identity in world; transform supplied scan pose first')
        moving=args.command=='run' or getattr(args,'move',False)
        if moving:
            attempted=True
            await arm.do_command({'set_speed':5.0},timeout=15)
        iterations=args.max_objects if args.command=='run' else (1000000 if args.command=='watch' else 1)
        for index in range(iterations):
            if moving:
                ok=await motion.move(component_name='arm',destination=PoseInFrame(
                    reference_frame='world',pose=Pose(**config['scan_pose'])),world_state=world,timeout=60)
                if not ok:
                    raise RuntimeError('Cannot reach scan pose')
                await asyncio.sleep(1)
            try:
                items,origin,basis,intrinsic=await snapshot(machine,camera,motion,config,
                    None if args.command=='scan' else detector,
                    allow_partial=getattr(args,'preview',False))
            except ValueError as error:
                if args.command!='watch':
                    raise
                print(str(error),flush=True)
                image=cv2.imread(str(OUTPUT_DIR/'sort_runs/latest_overlay.jpg'))
                if image is not None:
                    cv2.imshow('live container tracking',image)
                if cv2.waitKey(1)&0xff==ord('q'):
                    return
                await asyncio.sleep(0.2)
                continue
            if args.command=='watch':
                cv2.imshow('live container tracking',cv2.imread(str(OUTPUT_DIR/'sort_runs/latest_overlay.jpg')))
                if cv2.waitKey(1)&0xff==ord('q'):
                    return
                await asyncio.sleep(0.1)
                continue
            print(json.dumps(dict(items=items,destination_order=ORDER),indent=2))
            if args.command!='run':
                print('Saved sort_runs/latest_overlay.jpg; no pick/place issued'); return
            eligible=[i for i in items if not i['rejected'] and 'world_mm' in i]
            if not eligible:
                print('No eligible isolated object. This does NOT mean the sources are empty.'); return
            item=max(eligible,key=lambda i:i['world_mm'][2])
            # Recheck calibration before every physical transfer.
            check_ready(config)
            print(f"Transfer {index+1}: source {item['source']+1}, {item['color']} destination")
            await pick_one(machine,arm,gripper,motion,world,config,item,origin,basis,intrinsic)
        print('Requested transfer limit reached; inspect results before continuing.')
    finally:
        try:
            if attempted:
                try:
                    await arm.stop(timeout=10)
                finally:
                    await gripper.stop(timeout=10)
        finally:
            cv2.destroyAllWindows()
            await machine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    scan=sub.add_parser('scan'); scan.add_argument('--move',action='store_true')
    sub.add_parser('setup'); sub.add_parser('plan')
    watch=sub.add_parser('watch'); watch.add_argument('--preview',action='store_true',help='View available bins; never enables sorting')
    watch.add_argument('--debug',action='store_true',help='Show rejected raw ML proposals')
    watch.add_argument('--instances',action='store_true',help='Segment individual piece hypotheses with SAM2; slower on CPU')
    ruler=sub.add_parser('ruler'); ruler.add_argument('--mm',type=float,required=True)
    teach_parser=sub.add_parser('teach')
    teach_parser.add_argument('name',choices=['destination','destination-floor','orientation','transit','source-1-rim','source-2-rim'])
    calibration=sub.add_parser('calibrate')
    calibration.add_argument('--aligned-depth',action='store_true')
    calibration.add_argument('--grasp-offset-mm',type=float)
    calibration.add_argument('--shared-bin-heights',action='store_true')
    run_parser=sub.add_parser('run')
    run_parser.add_argument('--execute',action='store_true')
    run_parser.add_argument('--max-objects',type=int,default=1)
    args=parser.parse_args()
    config=json.loads(CONFIG.read_text())
    if args.command=='ruler':
        if not math.isfinite(args.mm) or args.mm<=0:
            raise ValueError('Ruler reference distance must be positive and finite')
        ruler_check(args); return
    if args.command=='setup':
        setup(config); return
    if args.command=='calibrate':
        if args.aligned_depth:
            config['depth_aligned_verified']=True
        if args.shared_bin_heights:
            config['shared_bin_heights_verified']=True
        if args.grasp_offset_mm is not None:
            if not math.isfinite(args.grasp_offset_mm) or not -30<=args.grasp_offset_mm<=10:
                raise ValueError('Grasp offset must be finite, -30..10 mm')
            config['grasp_offset_mm']=args.grasp_offset_mm
        save_config(config); print('Calibration values saved; no motion'); return
    if args.command=='run' and not 1<=args.max_objects<=20:
        raise ValueError('Use a transfer limit of 1..20')
    asyncio.run(run(args,config))


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. No automatic return or release commanded.')
    except Exception as error:
        raise SystemExit(f'Sorter stopped: {error}') from None
