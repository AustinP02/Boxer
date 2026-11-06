"""Single-block pickup commissioning: approach, recheck, grasp and short lift.

No destination assignment and no transfer. Two stages allow inspecting the
approach image before the first descent. Never releases or returns on failure.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame, WorldState
import color_sorter as sorter
from block_detector import COLORS
from sort_vision import decode_depth
from sample_scan import inspect_setup, motion_world_state

FOLDER=sorter.OUTPUT_DIR/'pickup_demo'


def pickup_target(items, config, color):
    candidates=[]
    for item in items:
        ignored=('no matching-color or verified-empty destination',
                 'no verified destination with footprint clearance')
        reasons=[r for r in item['rejected'] if r not in ignored]
        if item['color']==color and not reasons and 'world_mm' in item:
            error=sorter.source_height_error(item,config['sources'][item['source']],config)
            if error is None:
                candidates.append(item)
    if len(candidates)!=1:
        raise ValueError('Need exactly one isolated, depth-valid '+color+' block; no motion issued')
    item=candidates[0]
    values=(*item['world_mm'],config['grasp_offset_mm'],config['transit_z_mm'],
            *config['pick_orientation'].values())
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite pickup calibration')
    if config['pick_orientation']['o_z']>-.97 or not -30<=config['grasp_offset_mm']<0:
        raise ValueError('Need a downward tool and a measured downward grasp offset')
    x,y,top=item['world_mm']
    pick=Pose(x=x,y=y,z=top+config['grasp_offset_mm'],**config['pick_orientation'])
    rim=max(s['rim_z_mm'] for s in config['sources'])
    if pick.z<config['_table_top_z_mm']+5:
        raise ValueError('Pickup target too close to modeled table')
    approach=sorter.clone_pose(pick,z=max(top+80,rim+60))
    lift=sorter.clone_pose(pick,z=max(top+60,rim+60))
    if config['transit_z_mm']<approach.z+20:
        raise ValueError('Insufficient transit clearance')
    return item,pick,approach,lift


async def observe_target(machine,camera,item):
    origin,basis=await sorter.camera_basis(machine)
    frames,_=await camera.get_images(timeout=20)
    frame=cv2.imdecode(np.frombuffer(next(f for f in frames if f.name=='color').data,np.uint8),cv2.IMREAD_COLOR)
    depth=decode_depth(next(f for f in frames if f.name=='depth').data)
    p=(await camera.get_properties(timeout=15)).intrinsic_parameters
    point=basis.T@(np.array(item['world_mm'])-origin)
    if frame is None or point[2]<=0:
        raise ValueError('Cannot observe pickup target')
    u=int(round(p.focal_x_px*point[0]/point[2]+p.center_x_px))
    v=int(round(p.focal_y_px*point[1]/point[2]+p.center_y_px))
    if not 6<=u<frame.shape[1]-6 or not 6<=v<frame.shape[0]-6:
        raise ValueError('Pickup target outside approach view; stopped without closing')
    hsv=cv2.cvtColor(frame[v-3:v+4,u-3:u+4],cv2.COLOR_BGR2HSV)
    mask=np.zeros(hsv.shape[:2],np.uint8)
    for lo,hi in COLORS[item['color']][0]:
        mask|=cv2.inRange(hsv,(lo,90,60),(hi,255,255))
    patch=depth[v-3:v+4,u-3:u+4]
    good=patch[np.isfinite(patch)&(patch>100)&(patch<2000)]
    cv2.drawMarker(frame,(u,v),(0,0,255),cv2.MARKER_CROSS,24,2)
    FOLDER.mkdir(exist_ok=True,parents=True)
    cv2.imwrite(str(FOLDER/'approach.jpg'),frame)
    if np.mean(mask>0)<.8 or len(good)<35 or abs(float(np.median(good))-point[2])>5:
        raise ValueError('Target color/depth disagrees at approach; stopped without descending/closing')
    return {'pixel':[u,v],'expected_depth_mm':float(point[2]),'observed_depth_mm':float(np.median(good))}


async def run(args, config):
    load_dotenv(sorter.ROOT/'.env')
    machine=await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])),30)
    attempted=False
    arm=None; gripper=None
    try:
        arm=Arm.from_robot(machine,'arm'); gripper=Gripper.from_robot(machine,'gripper')
        motion=MotionClient.from_robot(machine,'builtin'); camera=Camera.from_robot(machine,'cam')
        frames=await machine.get_frame_system_config()
        mount=next(f.frame.pose_in_observer_frame for f in frames if f.frame.reference_frame=='gripper')
        expected={'x':0,'y':0,'z':160,'o_x':0,'o_y':0,'o_z':1,'theta':180}
        if mount.reference_frame!='arm' or any(abs(getattr(mount.pose,k)-v)>1e-4 for k,v in expected.items()):
            raise ValueError('Gripper frame no longer matches measured 160 mm tool')
        world,_=await inspect_setup(machine,motion)
        table_pose=(await motion.get_pose('table','world',timeout=15)).pose
        table_geometry=next(o for o in world.obstacles if o.reference_frame=='table').geometries[0]
        config['_table_top_z_mm']=table_pose.z+table_geometry.center.z+table_geometry.box.dims_mm.z/2
        if args.stage=='approach':
            in_place=getattr(args,'in_place',False)
            if in_place:
                current=(await motion.get_pose('gripper','world',timeout=15)).pose
                if current.o_z>-.97:
                    raise ValueError('Current tool must be vertically downward for in-place pickup')
                config['pick_orientation']={k:getattr(current,k) for k in ('o_x','o_y','o_z','theta')}
                config['transit_z_mm']=max(current.z,max(s['rim_z_mm'] for s in config['sources'])+100)
            detector=sorter.ContainerDetector()
            config['_object_detector']=sorter.ObjectInstanceDetector(sorter.MODELS_DIR/'sam2.1_t.pt')
            items,origin,basis,intrinsic=await sorter.snapshot(machine,camera,motion,config,detector,allow_partial=in_place)
            item,pick,approach,lift=pickup_target(items,config,args.color)
            world.obstacles.append(sorter.tray_walls(config,origin,basis,intrinsic))
            state=dict(time=datetime.now(timezone.utc).isoformat(),item=item,
                pick=sorter.pose_dict(pick),approach=sorter.pose_dict(approach),lift=sorter.pose_dict(lift),
                world=world.SerializeToString().hex())
        else:
            state=json.loads((FOLDER/'target.json').read_text())
            if (datetime.now(timezone.utc)-datetime.fromisoformat(state['time'])).total_seconds()>180:
                raise ValueError('Pickup approach expired; reacquire from scan pose')
            item=state['item']; pick=Pose(**state['pick']); approach=Pose(**state['approach']); lift=Pose(**state['lift'])
            if item['color']!=args.color:
                raise ValueError('Requested color differs from saved approach')
            old=WorldState.FromString(bytes.fromhex(state['world']))
            world.obstacles.extend(o for o in old.obstacles if o.reference_frame=='world')
            current=(await motion.get_pose('gripper','world',timeout=15)).pose
            if math.dist([current.x,current.y,current.z],[approach.x,approach.y,approach.z])>2:
                raise ValueError('Arm moved away from recorded approach')
            if any(abs(getattr(current,k)-getattr(approach,k))>1 for k in ('o_x','o_y','o_z','theta')):
                raise ValueError('Tool orientation changed since approach')
        print('Pickup target:',item['color'],sorter.pose_dict(pick),flush=True)
        if not args.execute:
            print('Observation only; no motion or gripper commands issued')
            return
        await arm.do_command({'set_speed':2.0},timeout=15)
        async def move(target):
            nonlocal attempted
            attempted=True
            if not await motion.move(component_name='gripper',
                    destination=PoseInFrame(reference_frame='world',pose=target),
                    world_state=motion_world_state(world),extra={'motion_profile':'linear'},timeout=90):
                raise RuntimeError('Pickup motion was not confirmed')
        if args.stage=='approach':
            current=(await motion.get_pose('gripper','world',timeout=15)).pose
            await move(sorter.clone_pose(current,z=max(current.z,config['transit_z_mm'])))
            await move(sorter.clone_pose(approach,z=max(current.z,config['transit_z_mm'])))
            await gripper.open(timeout=20)
            await move(approach)
            print('Approach observation:',await observe_target(machine,camera,item),flush=True)
            state['time']=datetime.now(timezone.utc).isoformat()
            (FOLDER/'target.json').write_text(json.dumps(state,indent=2))
            print('Approach complete; gripper remains open above the block. No descent or close issued.',flush=True)
        else:
            print('Approach recheck:',await observe_target(machine,camera,item),flush=True)
            await move(pick)
            print('At grasp height; closing once.',flush=True)
            if not await gripper.grab(timeout=25):
                raise RuntimeError('Gripper did not report a grasp; stopped without lifting or retrying')
            # Lift vertically while keeping the same XY and orientation.
            await move(lift)
            holding=await gripper.is_holding_something(timeout=15)
            if not holding.is_holding_something:
                raise RuntimeError('Holding check failed after lift; no transfer or release issued')
            state['completed']=datetime.now(timezone.utc).isoformat()
            (FOLDER/'result.json').write_text(json.dumps(state,indent=2))
            (FOLDER/'target.json').unlink(missing_ok=True)
            print('PICKUP CONFIRMED: lifted and holding. Arm stays here; no release or transfer issued.',flush=True)
    finally:
        try:
            if attempted:
                try:
                    await arm.stop(timeout=10)
                finally:
                    await gripper.stop(timeout=10)
        finally:
            await machine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color',choices=list(COLORS),default='green')
    parser.add_argument('--stage',choices=['approach','finish'],default='approach')
    parser.add_argument('--in-place',action='store_true',help='Acquire a fresh view at the current downward tool pose')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    config=json.loads(sorter.CONFIG.read_text())
    asyncio.run(run(args,config))


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Stopped; no automatic release or return.')
    except Exception as error:
        raise SystemExit('Pickup stopped: '+str(error))
