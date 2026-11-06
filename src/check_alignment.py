"""Inspect received RGB/depth registration; never aligns frames or moves hardware."""
import asyncio
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from project_paths import PROJECT_ROOT
from sort_vision import decode_depth
from rim_fusion import cue_maps, draw_cues


async def main():
    load_dotenv(PROJECT_ROOT/'.env')
    machine=await asyncio.wait_for(RobotClient.at_address(os.environ['VIAM_ADDRESS'],
        RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                        api_key_id=os.environ['VIAM_API_KEY_ID'])),30)
    try:
        camera=Camera.from_robot(machine,'cam')
        frozen=False; edges_on=True; holes_on=False; rim_mode=False
        mouse=[None,None]
        cv2.namedWindow('RGB with received depth edges')
        def on_mouse(event,x,y,flags,param):
            mouse[:]=[x,y]
        cv2.setMouseCallback('RGB with received depth edges',on_mouse)
        print('No robot movement. Magenta = depth discontinuities. SPACE freeze; E overlay; Q exit.')
        print('These are the received streams; enable alignment in Viam camera attributes first.')
        while True:
            if not frozen:
                images,_=await camera.get_images(timeout=20)
                color=next(i for i in images if i.name=='color')
                depth_image=next(i for i in images if i.name=='depth')
                frame=cv2.imdecode(np.frombuffer(color.data,np.uint8),cv2.IMREAD_COLOR)
                depth=decode_depth(depth_image.data)
                if frame is None or depth.shape!=frame.shape[:2]:
                    raise ValueError('Color/depth sizes differ or color decoding failed')
            valid=np.isfinite(depth)&(depth>100)&(depth<2000)
            edge=np.zeros(valid.shape,np.uint8)
            dx=(np.abs(depth[:,1:]-depth[:,:-1])>15)&valid[:,1:]&valid[:,:-1]
            dy=(np.abs(depth[1:,:]-depth[:-1,:])>15)&valid[1:,:]&valid[:-1,:]
            edge[:,1:][dx]=255; edge[1:,:][dy]=255
            edge=cv2.dilate(edge,np.ones((3,3),np.uint8))
            overlay=draw_cues(frame,cue_maps(frame,depth)) if rim_mode else frame.copy()
            if holes_on:
                overlay[~valid]=(overlay[~valid].astype(np.float32)*0.4+np.array([0,0,255])*0.6).astype(np.uint8)
            if edges_on and not rim_mode:
                overlay[edge>0]=(255,0,255)
            cv2.putText(overlay,'SPACE freeze | E edges | I holes | F rim clues | Q quit',
                        (12,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
            if rim_mode:
                cv2.putText(overlay,'Yellow: dark RGB | Green: RGB edge near depth hole (diagnostic only)',
                            (12,52),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            values=depth[valid]
            if len(values):
                low,high=200,1000
                normalized=np.clip((depth-low)/max(high-low,1)*255,0,255).astype(np.uint8)
                colored=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
                colored[~valid]=0
                cv2.rectangle(colored,(0,0),(colored.shape[1],52),(0,0,0),-1)
                cv2.putText(colored,f'Fixed color scale: {low}-{high} mm | usable depth: {100*np.mean(valid):.1f}%',
                            (12,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
                cv2.putText(colored,'Black = missing / outside viewer range (100-2000 mm)',
                            (12,47),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                cv2.imshow('Depth (black = invalid)',cv2.resize(colored,(960,540)))
            if mouse[0] is not None:
                u=min(frame.shape[1]-1,max(0,int(mouse[0]*frame.shape[1]/960)))
                v=min(frame.shape[0]-1,max(0,int(mouse[1]*frame.shape[0]/540)))
                patch=valid[max(0,v-3):v+4,max(0,u-3):u+4]
                reading=f'{depth[v,u]:.0f} mm' if valid[v,u] else 'NO USABLE DEPTH'
                text=f'Pixel ({u},{v}): {reading} | nearby usable: {100*np.mean(patch):.0f}%'
                cv2.rectangle(overlay,(0,overlay.shape[0]-38),(overlay.shape[1],overlay.shape[0]),(0,0,0),-1)
                cv2.putText(overlay,text,(12,overlay.shape[0]-13),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
                cv2.drawMarker(overlay,(u,v),(255,255,255),cv2.MARKER_CROSS,12,1)
            cv2.imshow('RGB with received depth edges',cv2.resize(overlay,(960,540)))
            key=cv2.waitKey(1)&0xff
            if key in (ord('q'),27):
                break
            if key==ord('f'):
                rim_mode=not rim_mode
            if key==ord('e'):
                edges_on=not edges_on
            if key==ord('i'):
                holes_on=not holes_on
            if key==32:
                frozen=not frozen
            await asyncio.sleep(0.03)
    finally:
        cv2.destroyAllWindows()
        await machine.close()


if __name__=='__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
