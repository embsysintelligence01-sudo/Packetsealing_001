"""Provisional hand-action rule and deferred resolution for Home mode.

This records evidence; it does not certify physical contact or replace the
trained motion classifier. Upstream jewel identity remains an explicit limit.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
from .scene_alignment import reference_point, scene_area_scale


TERMINAL = {'COMPLETED_PROVISIONAL', 'COMPLETED_INFERRED', 'RETURNED_RUBBING_UNVERIFIED',
            'CANCELLED', 'REVIEW_REQUIRED'}
WORKFLOW_STYLES = {
    'MOVEMENT_OBSERVED_ID_PENDING': ('', '#ffedd5', '#9a3412', 'Temporary mask moved with hand; permanent jewel identity remains pending.'),
    'UNASSIGNED_RUBBING_PROVISIONAL': ('', '#ede9fe', '#5b21b6', 'Hand-motion rubbing rule detected; jewel identity is unassigned.'),
    'UNASSIGNED_AWAITING_RETURN': ('', '#ffedd5', '#9a3412', 'Unassigned provisional action recorded; waiting for return or review.'),
    'UNASSIGNED_EVENT_REVIEW': ('', '#ffedd5', '#9a3412', 'Unassigned action footage retained; no named-jewel pickup confirmed.'),
    'GRIP_CANDIDATE': ('', '#fef3c7', '#92400e', 'Proximity candidate; recording has started.'),
    'PICKUP_PENDING': ('', '#ffedd5', '#9a3412', 'Candidate identity retained through occlusion.'),
    'PICKUP_VERIFIED': ('', '#dbeafe', '#1e40af', 'Departure evidence observed; recording continues until return.'),
    'RUBBING_PROVISIONAL': ('', '#ede9fe', '#5b21b6', 'Hand-motion rule suggests rubbing; physical contact is unverified.'),
    'AWAITING_RETURN': ('', '#dbeafe', '#1e40af', 'Provisional rubbing recorded; waiting for stable same-ID return.'),
    'COMPLETED_PROVISIONAL': ('', '#ede9fe', '#5b21b6', 'Departure and return observed; rubbing is provisional.'),
    'COMPLETED_INFERRED': ('', '#ffedd5', '#9a3412', 'Occluded departure inferred from sequence; review if required.'),
    'RETURNED_RUBBING_UNVERIFIED': ('', '#f3f4f6', '#374151', 'Returned without observed rubbing rule; footage retained.'),
    'CANCELLED': ('× ', '#fee2e2', '#991b1b', 'Jewel stayed at home without action evidence.'),
    'REVIEW_REQUIRED': ('', '#ffedd5', '#9a3412', 'Ambiguous or incomplete sequence; footage retained.'),
}


@dataclass
class RubEvent:
    jewel_id: int
    started: float
    state: str = 'GRIP_CANDIDATE'
    reason: str = 'Hand–jewel proximity; recording before departure confirmation.'
    hand_id: int | None = None
    last_hand: list | None = None
    hand_seen: float | None = None
    departure_verified: bool = False
    occluded: bool = False
    rubbing_at: float | None = None
    rubbing_evidence: dict | None = None
    returned_at: float | None = None
    updated_at: float | None = None
    remaining_at_home: list = field(default_factory=list)
    reversal_count: int = 0
    active_seconds: float = 0.
    near_seconds: float = 0.
    # Internal measurement state is omitted from the user-facing event.
    previous_t: float | None = None
    stone: list | None = None
    anchor: list | None = None
    direction: list | None = None
    away_since: float | None = None
    away_hits: int = 0
    last_area: int | None = None
    origin_cog: list | None = None
    stability: dict | None = None
    hand_status: str = 'associated'
    hand_rebindings: int = 0
    pending_hand: int | None = None
    pending_hand_hits: int = 0
    origin_hand: list | None = None
    motion_previous: tuple | None = None
    gold_measurement: dict | None = None
    rubbing_blocker: str | None = None
    last_gold_evidence: float | None = None
    last_rub_observation: float | None = None
    departure_observed: bool = False
    temporary_track: str | None = None
    named_identity_hits: int = 0
    event_number: int = 1
    previous_event: dict | None = None
    pickup_verified_at: float | None = None
    pickup_observed_at: float | None = None
    stone_observation: list | None = None
    origin_source: str = 'initial_fixed_home'
    return_since: float | None = None
    return_hits: int = 0
    return_cog: list | None = None
    clear_since: float | None = None
    clear_hits: int = 0

    def snapshot(self):
        keys = ('jewel_id', 'started', 'state', 'reason', 'hand_id', 'departure_verified',
                'occluded', 'rubbing_at', 'rubbing_evidence', 'returned_at', 'updated_at', 'remaining_at_home',
                'reversal_count', 'active_seconds', 'near_seconds', 'origin_cog', 'stability',
                'hand_status', 'hand_rebindings', 'gold_measurement', 'rubbing_blocker',
                'departure_observed','temporary_track','event_number','previous_event',
                'pickup_verified_at','pickup_observed_at','stone_observation','origin_source')
        data = asdict(self)
        pickup=('VERIFIED' if self.departure_verified else 'OBSERVED_ID_PENDING' if self.departure_observed else
                'CANCELLED_NO_DEPARTURE' if self.state=='CANCELLED' else 'UNVERIFIED' if self.state in TERMINAL else 'PENDING')
        return {key: data[key] for key in keys} | {'active': self.state not in TERMINAL,
            'pickup_status':pickup,'rubbing_status':'PROVISIONAL' if self.rubbing_at is not None else 'UNVERIFIED' if self.state in TERMINAL else 'PENDING',
            'return_status':'RETURNED' if self.returned_at is not None else 'NOT_REQUIRED' if self.state=='CANCELLED' else 'UNVERIFIED' if self.state in TERMINAL else 'PENDING',
            'rubbing_source': 'hand_back_and_forth_rule', 'physical_contact_verified': False,
            'identity_source': 'candidate_id_with_temporary_mask_continuity' if self.temporary_track else 'upstream_SLIC_tracker_and_temporal_continuity'}


class HomeRubbingFlow:
    def __init__(self, emit, min_seconds=1.2, min_reversals=3):
        self.emit = emit
        self.min_seconds = min_seconds
        self.min_reversals = min_reversals
        self.events = {}
        self.wait_exit = set()
        self.bed_positions={}
        self.history={}
        self.event_numbers={}

    def change(self, event, state, t, reason):
        changed = event.state != state
        event.state, event.reason = state, reason
        if changed:
            event.updated_at = t
            self.emit(dict(time=t, jewel=f'J{event.jewel_id}', state=state, reason=reason,
                           contact_at=event.started, workflow=event.snapshot()))

    @staticmethod
    def reset_motion(event):
        event.anchor = event.direction = None
        event.reversal_count = 0
        event.active_seconds = event.near_seconds = 0.

    def associate_carrier(self, event, jewel, engine, shape, t):
        from .hand_keypoints import hand_continuity_mask
        detection = engine.live_gold.get(jewel.id) or getattr(engine,'provisional_gold',{}).get(jewel.id)
        matches = []
        if detection is not None:
            x1,y1,x2,y2 = detection.box
            gold = detection.mask[y1:y2,x1:x2] > 0
            for key, hand in engine.hand_tracks.items():
                if any(e.state not in TERMINAL and e.hand_id==key and jid!=jewel.id for jid,e in self.events.items()):
                    continue
                fingers = hand_continuity_mask(hand['detection'],shape)[y1:y2,x1:x2] > 0
                if np.any(fingers & gold):
                    matches.append(key)
        if len(matches)>1:
            event.hand_status='ambiguous_hands_near_gold'
            event.pending_hand,event.pending_hand_hits=None,0
            return None
        if len(matches)==1 and matches[0]!=event.hand_id:
            key=matches[0]
            event.pending_hand_hits=event.pending_hand_hits+1 if event.pending_hand==key else 1
            event.pending_hand=key
            event.hand_status='confirming_new_hand_id'
            if event.pending_hand_hits<2:
                return None
            event.hand_id=key
            event.hand_rebindings+=1
            event.pending_hand,event.pending_hand_hits=None,0
            if event.hand_seen is None or t-event.hand_seen>.8:self.reset_motion(event)
        else:
            event.pending_hand,event.pending_hand_hits=None,0
        carrier=engine.hand_tracks.get(event.hand_id)
        if carrier is None and event.hand_seen is not None and t-event.hand_seen<=.8:
            nearby=[(key,hand) for key,hand in engine.hand_tracks.items()
                    if np.linalg.norm(hand['cog']-event.last_hand)<max(25.,hand['detection'].scale*.6)
                    and not any(e.state not in TERMINAL and e.hand_id==key and jid!=jewel.id for jid,e in self.events.items())]
            if len(nearby)==1:
                event.hand_id,carrier=nearby[0]
                event.hand_rebindings+=1
                event.motion_previous=None
        event.hand_status='associated' if carrier is not None else 'associated_hand_missing'
        return carrier

    def update(self, engine, stone, roi_mask, t):
        status = getattr(engine, 'scene_status', None)
        if status is None:
            # Compatibility with non-image callers and isolated state-machine tests.
            status = dict(state='INSUFFICIENT_EVIDENCE' if engine.global_uncertain else 'STABLE',
                          reason='legacy_scene_check', source='legacy', trusted=not engine.global_uncertain,
                          matrix=[[1.,0.,0.],[0.,1.,0.]])
        for jewel in engine.jewels:
            metrics = jewel.metrics or {}
            near = metrics.get('hand_jewel_proximity', False)
            event = self.events.get(jewel.id)
            if event is None or event.state in TERMINAL:
                if not near:
                    self.wait_exit.discard(jewel.id)
                if jewel.id in self.wait_exit or not near or jewel.associated_hand is None:
                    if event:
                        jewel.workflow = event.snapshot()
                    continue
                # Reserve hands to avoid two simultaneous candidate IDs sharing one hand.
                if any(e.state not in TERMINAL and e.hand_id == jewel.associated_hand
                       for jid, e in self.events.items() if jid != jewel.id):
                    continue
                previous=event.snapshot() if event is not None else None
                if previous is not None:self.history.setdefault(jewel.id,[]).append(previous)
                number=self.event_numbers.get(jewel.id,0)+1;self.event_numbers[jewel.id]=number
                summary=({key:previous.get(key) for key in ('event_number','started','pickup_status','rubbing_status','return_status','state')}
                         if previous is not None else None)
                event = RubEvent(jewel.id, t, hand_id=jewel.associated_hand, updated_at=t,event_number=number,previous_event=summary)
                home = np.asarray(getattr(jewel,'home_cog',(np.asarray(jewel.box[:2])+jewel.box[2:])*.5))
                origin=np.asarray(self.bed_positions.get(jewel.id,home))
                event.origin_source='last_verified_bed_position' if jewel.id in self.bed_positions else 'initial_fixed_home'
                event.origin_cog = origin.tolist() if origin is not None else None
                self.events[jewel.id] = event
                self.emit(dict(time=t, jewel=f'J{jewel.id}', state=event.state,
                               reason=event.reason, contact_at=t, workflow=event.snapshot()))
            event.remaining_at_home = [j.id for j in engine.jewels if j.id != jewel.id and
                                      (j.metrics or {}).get('present')]
            event.stone_observation=np.asarray(stone).tolist() if stone is not None else None
            dt = t-event.previous_t if event.previous_t is not None else 0.
            event.previous_t = t
            event.stability = {key:status.get(key) for key in ('state','reason','source','trusted','motion_px')}
            timing_ok = dt == 0 or 0 < dt <= .8

            carrier = self.associate_carrier(event,jewel,engine,roi_mask.shape,t)
            hand_point = None
            if carrier is not None:
                hand = carrier['detection']
                hand_point = (hand.points[4]+hand.points[8])*.5
                event.hand_seen, event.last_hand = t, carrier['cog'].tolist()

            if not timing_ok or not status.get('trusted'):
                self.reset_motion(event)
                event.return_since, event.return_hits = None, 0
                event.away_since, event.away_hits = None, 0
                event.clear_since, event.clear_hits = None, 0
                event.motion_previous=None
                event.reason = ('Frame gap exceeds 0.8 s; temporal votes restarted.' if not timing_ok else
                                f"Scene alignment unavailable: {status['reason']}; recording and hand tracking continue.")
                jewel.workflow = event.snapshot()
                continue

            identity_current=jewel.id in engine.live_gold
            event.named_identity_hits=event.named_identity_hits+1 if identity_current else 0
            detection = engine.live_gold.get(jewel.id) or getattr(engine,'provisional_gold',{}).get(jewel.id)
            measured_cog=jewel.current_cog if identity_current else getattr(engine,'provisional_cogs',{}).get(jewel.id)
            visible = detection is not None and measured_cog is not None
            event.gold_measurement=(getattr(jewel,'gold_quality',None) if identity_current else
                getattr(engine,'provisional_quality',{}).get(jewel.id,getattr(jewel,'gold_quality',None)))
            event.temporary_track=(event.gold_measurement or {}).get('track')
            if not identity_current and detection is not None and carrier is not None:near=True
            if not visible and carrier is not None:
                event.occluded = True  # Uncertainty, never departure evidence on its own.
                if not event.departure_verified and event.rubbing_at is None:
                    why=(event.gold_measurement or {}).get('reason','no_reliable_gold_measurement')
                    self.change(event, 'PICKUP_PENDING', t, f'Gold movement unavailable: {why}; candidate ID retained, recording continues.')
            diagonal = max(1., float(np.linalg.norm(np.subtract(jewel.box[2:], jewel.box[:2]))))
            aligned_cog = reference_point(measured_cog,status) if visible else None
            determinant = scene_area_scale(measured_cog,status) if visible else 1.
            area = np.count_nonzero(detection.mask)/max(.01,determinant) if visible else None
            home_pixels_applicable=(status['state'] in ('STABLE','NOT_CHECKED') or
                                    status['state']=='MARKER_ALIGNED' and status.get('motion_px',0)<2.5)
            home_present = home_pixels_applicable and metrics.get('present')
            away = (visible and event.origin_cog is not None and not home_present and
                    np.linalg.norm(aligned_cog-event.origin_cog) > max(20., diagonal*.35))
            area_stable = event.last_area is None or (area is not None and .85 <= area/max(1,event.last_area) <= 1.15)
            if visible:event.last_area = area
            partial=bool((event.gold_measurement or {}).get('partial'))
            if partial and not event.departure_verified and event.rubbing_at is None:
                event.reason=f"Partial gold {(event.gold_measurement or {}).get('area_ratio',0):.0%}; provisional tracking needs coherent hand/jewel movement."
            aligned_hand=reference_point(hand_point,status) if hand_point is not None else None
            if event.origin_hand is None and aligned_hand is not None:
                event.origin_hand=aligned_hand.tolist()
            coherent=False
            previous=event.motion_previous
            if aligned_hand is not None and aligned_cog is not None and previous is not None and 0<t-previous[0]<=.8:
                dg=aligned_cog-np.asarray(previous[1]);dh=aligned_hand-np.asarray(previous[2])
                ng,nh=np.linalg.norm(dg),np.linalg.norm(dh)
                scale=carrier['detection'].scale
                coherent=(min(ng,nh)>=2. and max(ng,nh)<=max(45.,scale) and
                          np.dot(dg,dh)/(ng*nh)>=.7 and np.linalg.norm(dg-dh)<=max(8.,scale*.15))
            if aligned_hand is not None and aligned_cog is not None:
                event.motion_previous=(t,aligned_cog.tolist(),aligned_hand.tolist())
            elif previous is not None and t-previous[0]>.8:event.motion_previous=None
            partial_supported=(not partial or (coherent and near and event.origin_hand is not None and
                np.linalg.norm(aligned_hand-event.origin_hand)>max(20.,diagonal*.35)))
            if away and area_stable and partial_supported:
                if event.away_since is None:
                    event.away_since = t
                event.away_hits += 1
                event.last_gold_evidence=t
                if event.away_hits >= 3 and t-event.away_since >= .45:
                    event.departure_observed=True
                    if event.pickup_observed_at is None:event.pickup_observed_at=t
                    if identity_current and event.named_identity_hits>=2:event.departure_verified = True
                    elif event.rubbing_at is None:
                        self.change(event,'MOVEMENT_OBSERVED_ID_PENDING',t,'Continuous temporary gold mask moved with the hand; permanent identity still pending.')
            elif visible or event.last_gold_evidence is None or t-event.last_gold_evidence>.8:
                event.away_since, event.away_hits = None, 0
            verified_at = getattr(jewel, 'verified_at', None)
            home = getattr(jewel, 'home_cog', (np.asarray(jewel.box[:2])+jewel.box[2:])*.5)
            if (jewel.state == 'PICKUP_CONFIRMED' and verified_at is not None and verified_at >= event.started
                    and event.origin_cog is not None and np.linalg.norm(np.asarray(event.origin_cog)-home) <= max(20.,diagonal*.35)):
                event.departure_verified = True
            if event.departure_verified:
                event.departure_observed=True
                if event.pickup_observed_at is None:event.pickup_observed_at=t
            if event.departure_verified and event.rubbing_at is None:
                if event.pickup_verified_at is None:event.pickup_verified_at=t
                self.change(event, 'PICKUP_VERIFIED', t, 'Same tracked jewel departed; recording continues toward stone and return.')
            elif event.departure_verified and event.pickup_verified_at is None:
                event.pickup_verified_at=t

            stone_box = np.asarray(stone, float) if stone is not None else None
            stone_stable = False
            if stone_box is not None:
                raw_size = max(1.,float(np.linalg.norm(stone_box[2:]-stone_box[:2])))
                stone_ref = np.concatenate((reference_point(stone_box[:2],status),reference_point(stone_box[2:],status)))
                size = max(1., float(np.linalg.norm(stone_ref[2:]-stone_ref[:2])))
                centre = (stone_ref[:2]+stone_ref[2:])*.5
                stone_stable = event.stone is None or np.linalg.norm(centre-(np.array(event.stone[:2])+event.stone[2:])*.5) <= size*.03
                event.stone = stone_ref.tolist()
            if hand_point is not None and stone_box is not None and stone_stable:
                margin = raw_size*.2
                near_stone = np.all(hand_point >= stone_box[:2]-margin) and np.all(hand_point <= stone_box[2:]+margin)
                if near_stone:
                    event.last_rub_observation=t
                    event.rubbing_blocker='insufficient_reversals_or_active_time'
                    point = (reference_point(hand_point,status)-centre)/size
                    event.near_seconds += max(0., dt)
                    if event.anchor is not None:
                        delta = point-event.anchor
                        distance = float(np.linalg.norm(delta))
                        if .025 <= distance <= .5:
                            direction = delta/distance
                            if event.direction is not None and np.dot(direction,event.direction) < -.3:
                                event.reversal_count += 1
                            event.direction = direction.tolist()
                            event.anchor = point.tolist()
                            event.active_seconds += max(0.,dt)
                        elif distance > .5:
                            self.reset_motion(event)
                    else:
                        event.anchor = point.tolist()
                    if (event.rubbing_at is None and event.reversal_count >= self.min_reversals and
                            event.active_seconds >= self.min_seconds and event.near_seconds >= self.min_seconds):
                        # Seeing the original jewel still at home contradicts attributing this hand action to it.
                        if not (visible and home_present and not event.departure_verified):
                            event.rubbing_at = t
                            event.rubbing_blocker=None
                            event.rubbing_evidence = dict(reversals=event.reversal_count,
                                active_seconds=event.active_seconds, near_seconds=event.near_seconds,
                                stone_box=event.stone)
                            self.change(event,'RUBBING_PROVISIONAL',t,
                                'Sustained back-and-forth hand motion near stone; rule-based rubbing, physical contact unverified.')
                elif event.rubbing_at is None:
                    if event.last_rub_observation is None or t-event.last_rub_observation>.8:self.reset_motion(event)
                    event.rubbing_blocker='associated_hand_outside_stone_region'
                elif event.state == 'RUBBING_PROVISIONAL':
                    self.change(event,'AWAITING_RETURN',t,'Provisional rubbing recorded; following jewel return to bench.')
            else:
                if event.last_rub_observation is None or t-event.last_rub_observation>.8:self.reset_motion(event)
                if event.rubbing_at is None:
                    event.rubbing_blocker=(event.hand_status if hand_point is None else
                        'stone_not_detected' if stone_box is None else 'stone_alignment_changed')
                    event.reason=f'Recording; rubbing rule paused: {event.rubbing_blocker}.'
                elif event.state == 'RUBBING_PROVISIONAL':
                    self.change(event,'AWAITING_RETURN',t,'Provisional rubbing recorded; waiting for stable same-ID return.')

            # Stable visible same-ID jewel, released from hand and outside stone,
            # can resolve a sequence even when it returns to a different bench spot.
            released = identity_current and visible and not partial and not near
            if released:
                x1,y1,x2,y2 = detection.box
                support = detection.mask[y1:y2,x1:x2] > 0
                guard = getattr(engine,'home_guard',np.zeros_like(roi_mask))[y1:y2,x1:x2]
                released = bool(np.any(support) and np.mean(guard[support]>0) <= .05)
                cog = jewel.current_cog
                x,y = np.rint(aligned_cog).astype(int)
                released = released and 0<=y<roi_mask.shape[0] and 0<=x<roi_mask.shape[1] and roi_mask[y,x]>0
                released = released and stone_box is not None and stone_stable and not np.all((cog>=stone_box[:2]-raw_size*.1)&(cog<=stone_box[2:]+raw_size*.1))
                if released:
                    low=np.maximum(np.floor(stone_box[:2]-raw_size*.1).astype(int),0)
                    high=np.minimum(np.ceil(stone_box[2:]+raw_size*.1).astype(int),(roi_mask.shape[1],roi_mask.shape[0]))
                    released=np.count_nonzero(detection.mask[low[1]:high[1],low[0]:high[0]])/max(1,area)<=.05
            if released and (event.return_cog is None or np.linalg.norm(aligned_cog-event.return_cog)<=max(5.,diagonal*.05)):
                if event.return_since is None:
                    event.return_since=t
                event.return_hits+=1
                if event.return_cog is None:
                    event.return_cog=aligned_cog.tolist()
            else:
                event.return_since,event.return_cog,event.return_hits=None,None,0
            returned = event.return_hits>=3 and t-event.return_since>=.6
            at_start=(visible and identity_current and not partial and event.origin_cog is not None and
                      np.linalg.norm(aligned_cog-event.origin_cog)<=max(8.,diagonal*.08))
            if returned and identity_current and not partial:
                self.bed_positions[jewel.id]=aligned_cog.tolist()
            if returned and (event.departure_verified or event.rubbing_at is not None):
                event.returned_at=t
                if event.rubbing_at is not None:
                    outcome='COMPLETED_PROVISIONAL' if event.departure_verified else 'COMPLETED_INFERRED' if event.occluded else 'REVIEW_REQUIRED'
                    self.change(event,outcome,t,'Same upstream jewel ID returned stably to bench; rubbing is inferred by hand-motion rule, not verified contact.')
                else:
                    self.change(event,'RETURNED_RUBBING_UNVERIFIED',t,'Rubbing rule not detected before return; action unverified, recording retained.')
            elif returned:
                cancel=at_start and not event.occluded and not event.departure_observed
                self.change(event,'CANCELLED' if cancel else 'REVIEW_REQUIRED',t,
                    'Same jewel remained at its confirmed starting position; passing-hand candidate cancelled.' if cancel else
                    'Jewel seen at a different or uncertain position; departure unresolved, recording retained for review.')
            elif at_start and not event.occluded and not event.departure_observed and not event.departure_verified and event.rubbing_at is None and visible and home_present and not metrics.get('home_covered') and not near:
                if event.clear_since is None:
                    event.clear_since=t
                event.clear_hits+=1
                if event.clear_hits>=3 and t-event.clear_since>=.45:
                    self.change(event,'CANCELLED',t,'Jewel remained at home without departure or rubbing evidence.')
            else:
                event.clear_since,event.clear_hits=None,0
            if t-event.started>120 and event.state not in TERMINAL:
                self.change(event,'REVIEW_REQUIRED',t,'Unresolved sequence after 120 seconds; recording retained for review.')
            if event.state in TERMINAL:
                self.wait_exit.add(jewel.id)
            jewel.workflow=event.snapshot()
