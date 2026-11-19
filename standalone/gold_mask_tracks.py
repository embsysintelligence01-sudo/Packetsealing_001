"""Measure gold-mask centroids independently of permanent jewel identities."""
import cv2
import numpy as np
from .hand_keypoints import hand_continuity_mask
from .scene_alignment import reference_point


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


class GoldMaskTracks:
    def __init__(self):
        self.tracks={}
        self.next_id=1
        self.last_t=None

    def update(self, gold, hands, tracked, shape, t, scene=None):
        if self.last_t is not None and t<=self.last_t:
            return self.snapshots()
        previous={k:v for k,v in self.tracks.items() if t-v['time']<=.8}
        self.last_t=t
        detections=[]
        for detection in gold:
            x1,y1,x2,y2=map(int,detection.box)
            mask=detection.mask[y1:y2,x1:x2]
            moments=cv2.moments(mask,binaryImage=True)
            if moments['m00']<=0:
                continue
            cog=np.array([moments['m10']/moments['m00']+x1,moments['m01']/moments['m00']+y1])
            detections.append(dict(detection=detection,cog=cog,area=moments['m00']))
        proposals=[]
        for current in detections:
            ranked=[]
            box=current['detection'].box
            diagonal=max(1.,np.linalg.norm(np.subtract(box[2:],box[:2])))
            for key,old in previous.items():
                distance=np.linalg.norm(current['cog']-old['cog'])
                overlap=box_iou(box,old['detection'].box)
                if .25<=current['area']/old['area']<=4. and (overlap>.1 or distance<max(25.,diagonal*.6)):
                    ranked.append((overlap-.3*distance/diagonal,key))
            ranked.sort(reverse=True)
            proposals.append(ranked[0][1] if ranked and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>.15) else None)
        usage={key:proposals.count(key) for key in previous}
        fingers=[hand_continuity_mask(hand,shape) for hand in hands]
        current_tracks={}
        for current,proposal in zip(detections,proposals):
            old=previous.get(proposal) if usage.get(proposal)==1 else None
            key=proposal if old is not None else self.next_id
            if old is None:self.next_id+=1
            box=current['detection'].box
            x1,y1,x2,y2=map(int,box)
            support=current['detection'].mask[y1:y2,x1:x2]>0
            close=[i for i,finger in enumerate(fingers) if np.any((finger[y1:y2,x1:x2]>0)&support)]
            hand=hands[close[0]] if len(close)==1 else None
            point=(hand.points[4]+hand.points[8])*.5 if hand is not None and hasattr(hand,'points') else None
            cog_ref=reference_point(current['cog'],scene)
            hand_ref=reference_point(point,scene)
            coherent=False
            if old is not None and cog_ref is not None and hand_ref is not None and old.get('cog_ref') is not None and old.get('hand_ref') is not None:
                dg=cog_ref-old['cog_ref'];dh=hand_ref-old['hand_ref']
                ng,nh=np.linalg.norm(dg),np.linalg.norm(dh)
                coherent=(min(ng,nh)>=2. and max(ng,nh)<=max(45.,hand.scale) and
                          np.dot(dg,dh)/(ng*nh)>=.7 and np.linalg.norm(dg-dh)<=max(8.,hand.scale*.15) and
                          .75<=current['area']/old['area']<=1.25)
            hits=old['motion_hits']+1 if coherent else 0
            since=old['motion_since'] if coherent and old['motion_since'] is not None else t if coherent else None
            moving=bool(hits>=3 and t-since>=.45)
            merged=sum(box_iou(box,item['detection'].box)>.05 and
                       x1<=item['cog'][0]<=x2 and y1<=item['cog'][1]<=y2 for item in previous.values())>1
            ambiguous_mask=merged or (old is not None and old.get('ambiguous_mask',False))
            # Resolve identity only from a unique, fresh upstream SLIC/ID match.
            # Unknown masks still retain a COG and temporary track.
            matches=[jid for jid,tr in tracked.items() if tr.get('seen')==t and
                     box_iou(box,tr.get('detected_box',tr['box']))>.5]
            jid=matches[0] if len(matches)==1 else None
            conflicts=[jid2 for jid2,tr in tracked.items() if tr.get('seen')==t and
                       x1<=(tr['box'][0]+tr['box'][2])*.5<=x2 and y1<=(tr['box'][1]+tr['box'][3])*.5<=y2]
            if len(conflicts)>1 or ambiguous_mask:jid=None
            ambiguous=ambiguous_mask or len(conflicts)>1 or len(close)>1
            if ambiguous:
                hits,since=0,None
                cog_ref,hand_ref=None,None
            identity_hits=(old.get('identity_hits',0)+1 if old is not None and old.get('candidate_jewel_id')==jid else 1) if jid is not None else 0
            resolved=jid if identity_hits>=2 else None
            current.update(track_id=key,time=t,cog_ref=cog_ref,hand_ref=hand_ref,
                visible=True,
                motion_hits=hits,motion_since=since,candidate_jewel_id=jid,identity_hits=identity_hits,
                jewel_id=resolved,identity_status='upstream_id_confirmed' if resolved is not None else 'identity_pending',
                ambiguous_mask=ambiguous_mask,
                state='GOLD_MASK_AMBIGUOUS' if ambiguous else
                      'GOLD_MOVING_WITH_HAND' if moving else 'GOLD_NEAR_HAND' if hand is not None else 'GOLD_VISIBLE',
                hand_candidates=len(close))
            current_tracks[key]=current
        for key,old in previous.items():
            if key not in current_tracks:
                held=old.copy();held.update(visible=False,state='GOLD_OCCLUDED',hand_candidates=0)
                current_tracks[key]=held
        self.tracks=current_tracks
        return self.snapshots()

    def snapshots(self):
        return [dict(track=f'Gold-T{key}',box=list(map(int,item['detection'].box)),
                     cog=item['cog'].tolist() if item.get('visible',True) else None,
                     visible=item.get('visible',True),gold_pixels=int(item['area']),state=item['state'],
                     jewel_id=item['jewel_id'],identity_status=item['identity_status'],
                     motion_hits=item['motion_hits'],hand_candidates=item['hand_candidates'],
                     pickup_verified=False,physical_contact_verified=False,
                     motion_source='mask_centroid_and_nearby_hand_rule',workflow=item.get('workflow'))
                for key,item in sorted(self.tracks.items(),key=lambda pair:(not pair[1].get('visible',True),pair[0]))]

    def draw(self,frame,scale=1.):
        for item in self.snapshots():
            if item['cog'] is None:continue
            point=tuple(np.rint(item['cog']).astype(int))
            colour=(0,255,255)
            cv2.circle(frame,point,max(3,round(scale*3)),colour,-1)
            identity=f"J{item['jewel_id']}" if item['jewel_id'] is not None else 'ID pending'
            phase=(item.get('workflow')or{}).get('state',item['state'])
            label=f"{item['track']} | {identity} | {phase}"
            cv2.putText(frame,label,(item['box'][0],max(round(18*scale),item['box'][1]-round(6*scale))),
                        cv2.FONT_HERSHEY_SIMPLEX,.35*scale,colour,max(1,round(scale)))
