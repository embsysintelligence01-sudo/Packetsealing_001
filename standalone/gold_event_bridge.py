"""Connect anonymous mask continuity to candidates without inventing jewel IDs."""
from types import SimpleNamespace
import numpy as np
from .gold_mask_tracks import box_iou
from .hand_keypoints import hand_continuity_mask
from .home_rubbing import HomeRubbingFlow,TERMINAL


class GoldEventBridge:
    def __init__(self,emit):
        self.links={}
        self.proxies={}
        self.last_seen={}
        self.records=[]
        self.unassigned=[]
        def anonymous_event(data):
            data=dict(data)
            if data.get('jewel','').startswith('JGold-T'):data['jewel']=data['jewel'][1:]
            if data.get('state')=='RUBBING_PROVISIONAL':data['state']='UNASSIGNED_RUBBING_PROVISIONAL'
            if data.get('state')=='PICKUP_VERIFIED':data['state']='UNASSIGNED_MOVEMENT_OBSERVED'
            if data.get('state') in ('COMPLETED_PROVISIONAL','COMPLETED_INFERRED'):
                data['state']='UNASSIGNED_EVENT_REVIEW'
            if data.get('workflow'):
                data['workflow']=dict(data['workflow'],jewel_id=None,identity_verified=False,
                    state=data['state'],pickup_status='UNASSIGNED',pickup_verified_at=None,pickup_verified=False,departure_verified=False,identity_source='temporary_mask_continuity_identity_pending')
            data['identity_verified']=False
            data['reason']='Unassigned gold track: '+data.get('reason','')
            emit(data)
        self.flow=HomeRubbingFlow(anonymous_event)

    def prepare(self,engine,t):
        tracks={k:v for k,v in engine.gold_masks.tracks.items() if v.get('visible',True) and not v.get('ambiguous_mask') and v['state']!='GOLD_MASK_AMBIGUOUS'}
        engine.provisional_gold={}
        engine.provisional_cogs={}
        engine.provisional_quality={}
        assigned=set()
        for jewel in engine.jewels:
            detection=engine.live_gold.get(jewel.id)
            if detection is not None:
                matches=[key for key,item in tracks.items() if box_iou(detection.box,item['detection'].box)>.5]
                if len(matches)==1 and matches[0] not in assigned:self.links[jewel.id]=matches[0]
            elif jewel.id not in self.links:
                # A fresh upstream match may still be too partial for the named
                # observer. Keep this association provisional rather than forcing it.
                matches=[key for key,item in tracks.items() if item.get('candidate_jewel_id')==jewel.id]
                if len(matches)==1 and matches[0] not in assigned:self.links[jewel.id]=matches[0]
            key=self.links.get(jewel.id)
            if key not in tracks or key in assigned:continue
            event=engine.home_rubbing.events.get(jewel.id)
            if not ((event is not None and event.state not in TERMINAL) or (jewel.metrics or {}).get('hand_jewel_proximity')):
                continue
            assigned.add(key)
            item=tracks[key]
            if detection is None:
                engine.provisional_gold[jewel.id]=item['detection']
                engine.provisional_cogs[jewel.id]=item['cog']
                ratio=item['area']/max(1,np.count_nonzero(jewel.gold_mask))
                engine.provisional_quality[jewel.id]=dict(accepted=True,partial=True,area_ratio=ratio,
                    reason='temporary_mask_continuity_identity_pending',track=f'Gold-T{key}')
        collisions={key for key in self.links.values() if list(self.links.values()).count(key)>1}
        for jid,key in self.links.items():
            if key in collisions:
                engine.provisional_gold.pop(jid,None)
                engine.provisional_cogs.pop(jid,None)
                engine.provisional_quality.pop(jid,None)
        self.assigned=assigned-collisions

    def update_unassigned(self,engine,stone,roi,t):
        for label,event in list(self.flow.events.items()):
            if event.state in TERMINAL and t-self.last_seen.get(label,t)>10.:
                self.proxies.pop(label,None);self.flow.events.pop(label,None)
                self.flow.wait_exit.discard(label);self.last_seen.pop(label,None)
        live={}
        for key,item in engine.gold_masks.tracks.items():
            if key in self.assigned or not item.get('visible',True) or item['state']=='GOLD_MASK_AMBIGUOUS':continue
            label=f'Gold-T{key}'
            proxy=self.proxies.get(label)
            if proxy is None:
                proxy=SimpleNamespace(id=label,box=item['detection'].box,
                    home_cog=item['cog'].copy(),contact_at=t,state='UNCERTAIN',verified_at=None)
                self.proxies[label]=proxy
            self.last_seen[label]=t
            proxy.current_cog=item['cog'].copy()
            proxy.displacement=float(np.linalg.norm(proxy.current_cog-proxy.home_cog))
            x1,y1,x2,y2=item['detection'].box
            mask=item['detection'].mask[y1:y2,x1:x2]>0
            hands=[hid for hid,hand in engine.hand_tracks.items() if np.any((hand_continuity_mask(hand['detection'],roi.shape)[y1:y2,x1:x2]>0)&mask)]
            proxy.associated_hand=hands[0] if len(hands)==1 else None
            proxy.metrics=dict(hand_jewel_proximity=len(hands)==1,present=False,home_covered=False)
            proxy.gold_quality=dict(accepted=True,partial=False,reason='anonymous_gold_mask')
            live[label]=item['detection']
        eligible=[]
        for label,proxy in self.proxies.items():
            if label not in live:
                proxy.current_cog=None;proxy.associated_hand=None
                proxy.metrics=dict(hand_jewel_proximity=False,present=False,home_covered=False)
            eligible.append(proxy)
        anonymous=SimpleNamespace(jewels=eligible,live_gold=live,hand_tracks=engine.hand_tracks,
            global_uncertain=engine.global_uncertain,scene_status=getattr(engine,'scene_status',None),
            home_guard=getattr(engine,'home_guard',np.zeros_like(roi)))
        self.flow.update(anonymous,stone,roi,t)
        self.unassigned=[];self.records=[]
        for label,proxy in self.proxies.items():
            event=self.flow.events.get(label)
            if event is None:continue
            if label not in live and t-self.last_seen.get(label,t)>2. and event.state not in TERMINAL:
                self.flow.change(event,'REVIEW_REQUIRED',t,'Temporary gold track lost or handed to named candidate; buffered recording retained.')
            snapshot=event.snapshot()
            snapshot.update(jewel_id=None,gold_track=label,identity_verified=False,pickup_verified=False,pickup_status='UNASSIGNED',pickup_verified_at=None,
                departure_verified=False,identity_source='temporary_mask_continuity_identity_pending')
            if event.state=='RUBBING_PROVISIONAL':snapshot['state']='UNASSIGNED_RUBBING_PROVISIONAL'
            elif event.state=='AWAITING_RETURN':snapshot['state']='UNASSIGNED_AWAITING_RETURN'
            elif event.state=='PICKUP_VERIFIED':snapshot['state']='UNASSIGNED_MOVEMENT_OBSERVED'
            elif event.state in ('COMPLETED_PROVISIONAL','COMPLETED_INFERRED'):
                snapshot['state']='UNASSIGNED_EVENT_REVIEW'
            proxy.workflow=snapshot
            self.unassigned.append(snapshot)
            self.records.append(proxy)
            key=int(label.removeprefix('Gold-T'))
            if key in engine.gold_masks.tracks:engine.gold_masks.tracks[key]['workflow']=snapshot
