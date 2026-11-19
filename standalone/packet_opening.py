"""Advisory packet-box classification; never changes placement or sealing rules."""
from pathlib import Path

import cv2

from packet_mouth_lab.mouth_core import TemporalConsensus, crop_roi, to_working_size
from .flow import INSIDE


class PacketOpening:
    def __init__(self, path):
        self.path=Path(path)
        self.classifier=None
        self.loaded=False
        self.temporal=TemporalConsensus()
        self.model_info={'path':str(self.path), 'status':'not_loaded', 'input_region':'packet_bbox'}
        self.current={'status':'unavailable', 'label':None, 'consistent':False}

    def reset(self):
        self.temporal.reset()
        self.current={'status':'unavailable', 'label':None, 'consistent':False}

    def update(self, frame, bag, timestamp):
        self.current={'timestamp':timestamp, 'status':'packet_unavailable', 'label':None, 'consistent':False}
        if bag is None:
            self.temporal.reset()
            return self.current
        self.current['box_xyxy']=list(map(int,bag.box))
        try:
            if not self.loaded:
                self.loaded=True
                if not self.path.is_file():
                    self.model_info['status']='model_missing'
                else:
                    from packet_mouth_lab.mouth_training import MouthClassifier
                    self.classifier=MouthClassifier(self.path)
                    if self.classifier.payload.get('input_region')!='packet_bbox':
                        self.classifier=None
                        raise ValueError('Classifier must be trained on packet bounding boxes.')
                    self.model_info.update(status='ready', validation=self.classifier.payload['validation'],
                                           trained_at=self.classifier.payload.get('created_at'),
                                           min_probability=self.temporal.min_prob,
                                           min_samples=self.temporal.min_samples,
                                           min_span=self.temporal.min_span,
                                           max_gap=self.temporal.max_gap)
            if self.classifier is None:
                self.current['status']=self.model_info['status']
                self.temporal.reset()
                return self.current
            # Match the lab's max-960 working frame before cropping and 64px features.
            working=to_working_size(frame)
            sx,sy=working.shape[1]/frame.shape[1],working.shape[0]/frame.shape[0]
            x1,y1,x2,y2=bag.box
            roi=(round(x1*sx),round(y1*sy),round(x2*sx)-round(x1*sx),round(y2*sy)-round(y1*sy))
            crop=crop_roi(working,roi)
            if crop is None:
                self.temporal.reset()
                return self.current
            label,probability,probabilities=self.classifier.predict(crop)
            self.model_info['status']='ready'
            consensus=self.temporal.add(label,probability,timestamp)
            self.current.update(status='classified',label=label,probability=float(probability),
                                probabilities=probabilities, consistent=consensus==f'{label}_CONSISTENT',
                                box_xyxy=list(map(int,bag.box)))
        except Exception as exc:
            self.temporal.reset()
            self.model_info.update(status='classifier_error',error=str(exc))
            self.current.update(status='classifier_error',error=str(exc))
        return self.current

    def record(self, flow, timestamp):
        flow.packet_opening_model=dict(self.model_info)
        observation=self.current
        for jewel in flow.jewels.values():
            if jewel.started is None or timestamp<jewel.started or (jewel.ended is not None and timestamp>jewel.ended):
                continue
            if jewel.ended==timestamp and jewel.packet_opening_at_placement is None:
                jewel.packet_opening_at_placement=(dict(observation) if observation.get('timestamp')==timestamp else
                                                  dict(timestamp=timestamp,status='observation_unavailable',label=None,consistent=False))
                jewel.packet_opening_at_placement['basis']='single_placement_frame'
            if observation.get('timestamp')!=timestamp or not observation.get('consistent'):
                continue
            events=jewel.packet_opening_observations
            if events and events[-1]['label']==observation['label'] and timestamp-events[-1]['last_seen']<=1.1:
                events[-1].update(last_seen=timestamp,samples=events[-1]['samples']+1)
            else:
                events.append(dict(label=observation['label'],first_seen=timestamp,last_seen=timestamp,
                                   samples=1,probability=observation['probability']))
                del events[:-12]

    def save_placement_frame(self, frame, jewel, timestamp, work):
        """Save this exact clean frame, before any evidence overlays are drawn."""
        from .evidence import image_save
        observation=jewel.packet_opening_at_placement
        if observation is None or observation.get('timestamp')!=timestamp or jewel.ended!=timestamp:
            return
        name=f'J{jewel.id}_placement.jpg'
        working=to_working_size(frame)
        image_save(work/name,working)
        observation.update(frame=name, source_size_wh=[frame.shape[1],frame.shape[0]],
                           saved_size_wh=[working.shape[1],working.shape[0]], box_coordinate_space='source_frame')
        if observation.get('box_xyxy'):
            sx,sy=working.shape[1]/frame.shape[1],working.shape[0]/frame.shape[0]
            x1,y1,x2,y2=observation['box_xyxy']
            crop=crop_roi(working,(round(x1*sx),round(y1*sy),round(x2*sx)-round(x1*sx),round(y2*sy)-round(y1*sy)))
            if crop is not None:
                name=f'J{jewel.id}_placement_packet.jpg'
                image_save(work/name,crop)
                observation['packet_crop']=name

    def draw(self, frame):
        observation=self.current
        if not observation.get('label') or not observation.get('box_xyxy'):
            return
        x1,y1,x2,y2=observation['box_xyxy']
        text=f"{observation['label']} {observation['probability']:.0%} | "
        text+='consistent' if observation['consistent'] else 'unconfirmed'
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,40),2)
        cv2.putText(frame,text,(max(0,x1),max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,180,40),2)


def opening_description(record):
    """PDF wording separates classifier observations from transfer-based inference."""
    events=record.get('packet_opening_observations',[])
    labels={event['label'] for event in events}
    descriptions=[]
    placement=record.get('packet_opening_at_placement')
    if placement is not None:
        label=placement.get('label')
        if placement.get('timestamp')!=record.get('end'):
            label=None
        views={'OPEN_CAMERA':'a camera-facing packet opening',
               'OPEN_SIDE':'the packet viewed from the side with an apparent opening',
               'CLOSED_FOLDED':'a closed or folded packet view',
               'UNCLEAR':'an unclear packet view'}
        if label in views and placement.get('status')=='classified':
            descriptions.append(f"In J{record['id']}'s placement frame, the classifier indicated {views[label]} "
                                f"({label}; score: {placement['probability']:.1%}). Single-frame prediction; "
                                +('temporally consistent.' if placement.get('consistent') else 'temporal consistency not established.'))
        else:
            reason=('no accepted packet detection' if placement.get('status')=='packet_unavailable'
                    else 'classifier or matching frame unavailable')
            descriptions.append(f'Placement-frame classification unavailable: {reason}.')
        if record.get('status')==INSIDE:
            descriptions.append(f"Jewel J{record['id']} was inferred placed inside from transfer evidence.")
        return ' '.join(descriptions)
    if 'OPEN_CAMERA' in labels:
        descriptions.append('Classifier indicated a camera-facing packet opening during transfer.')
    if 'OPEN_SIDE' in labels:
        descriptions.append('Classifier indicated the packet was viewed from the side with an apparent opening.')
    if 'CLOSED_FOLDED' in labels:
        descriptions.append('Classifier indicated a closed or folded packet view during transfer.')
    if not descriptions:
        descriptions.append('Packet opening state could not be confirmed during transfer.')
    if record.get('status')==INSIDE:
        descriptions.append(f"Jewel J{record['id']} was inferred placed inside from transfer evidence.")
    return ' '.join(descriptions)
