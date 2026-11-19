"""Small on-device classifier, with session-grouped held-out evaluation.

Trains only when all labels can be evaluated on entirely held-out packet/session IDs.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

if __package__:
    from .mouth_core import mouth_features, LABELS
else:
    from mouth_core import mouth_features, LABELS


class MouthDataset:
    def __init__(self, root, input_region='mouth_roi'):
        self.root = Path(root)
        self.input_region = input_region
        self.root.mkdir(exist_ok=True, parents=True)

    def add(self, crop, label, session_id, context=None):
        if label not in LABELS:
            raise ValueError('Unsupported class label')
        if not session_id.strip():
            raise ValueError('Enter a packet/session ID (different for independent recordings/packets).')
        if crop is None or crop.size == 0:
            raise ValueError('Select a valid mouth ROI first.')
        stamp = f'{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}'
        path = self.root / f'{stamp}.jpg'
        if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise IOError('Could not save labelled example')
        meta = {'image':path.name,'label':label,'session_id':session_id.strip(),'created_at':time.time()}
        meta['input_region'] = self.input_region
        if context is not None:
            meta['packet_context'] = context
        (self.root / f'{stamp}.json').write_text(json.dumps(meta,indent=2))
        return path

    def add_packet(self, frame, detection, session_id, source, mouth_roi=None, label=None, label_region='mouth_roi'):
        """Keep full-frame context and coordinates for future mouth localisation."""
        if not session_id.strip():
            raise ValueError('Enter a packet/recording group ID first.')
        if detection is None:
            raise ValueError('Exactly one current packet detection is required.')
        x1,y1,x2,y2 = map(int,detection.box)
        packet = frame[y1:y2,x1:x2]
        if packet.size == 0:
            raise ValueError('Detected packet crop is empty.')
        stamp = f'{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}'
        folder = self.root/'packet_frames'/stamp
        folder.mkdir(parents=True)
        for name,image in [('frame.jpg',frame),('packet.jpg',packet),('packet_mask.png',detection.mask)]:
            if not cv2.imwrite(str(folder/name),image):
                raise IOError(f'Could not save {name}')
        metadata = {'session_id':session_id.strip(), 'source':source,
                    'packet_box_xyxy':[x1,y1,x2,y2],
                    'packet_confidence':float(detection.confidence),
                    'packet_model':'bagnewmodel.onnx', 'bag_filter':'fp_filter_resnet18_bag.onnx',
                    'mouth_roi_xywh':list(mouth_roi) if mouth_roi is not None else None,
                    'mouth_label':label if label_region=='mouth_roi' else None,
                    'packet_label':label if label_region=='packet_bbox' else None,
                    'label_region':label_region, 'coordinate_space':'working_frame',
                    'frame_size_wh':[frame.shape[1],frame.shape[0]], 'created_at':time.time()}
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
        return str(folder.relative_to(self.root))

    def examples(self):
        for p in self.root.glob('*.json'):
            try:
                d=json.loads(p.read_text())
                if d.get('label') in LABELS and d.get('session_id'):
                    img=cv2.imread(str(self.root / d['image']))
                    if img is not None:
                        yield mouth_features(img),d['label'],str(d['session_id']),str(p)
            except (ValueError,KeyError,TypeError,OSError):
                continue

    def counts(self):
        return dict(Counter(l for _,l,_,_ in self.examples()))


def _pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ('scale',StandardScaler()),
        ('model',LogisticRegression(max_iter=1500,class_weight='balanced',solver='lbfgs')),
    ])


def train_validated(dataset: MouthDataset, output_path, min_per_class=12):
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
    examples=list(dataset.examples())
    by_label=Counter(e[1] for e in examples)
    active=sorted(by_label)
    if len(active)<3:
        raise ValueError('Collect at least three classes, including a closed/folded and an open class.')
    if 'CLOSED_FOLDED' not in active or not any(x.startswith('OPEN_') for x in active):
        raise ValueError('Need CLOSED_FOLDED and at least one OPEN class.')
    if any(by_label[label] < min_per_class for label in active):
        raise ValueError(f'Collect >= {min_per_class} diverse crops in EACH recorded class. Counts: {dict(by_label)}')
    class_groups = {lab:set(e[2] for e in examples if e[1]==lab) for lab in active}
    if any(len(group)<2 for group in class_groups.values()):
        raise ValueError('Each class needs at least two distinct packet/session IDs for held-out testing.')
    X=np.vstack([x[0] for x in examples])
    y=np.asarray([e[1] for e in examples])
    groups=np.asarray([e[2] for e in examples])
    splitter=GroupShuffleSplit(n_splits=150,test_size=.25,random_state=41)
    split=None
    for tr,te in splitter.split(X,y,groups=groups):
        if set(y[tr])==set(active) and set(y[te])==set(active) and not set(groups[tr])&set(groups[te]):
            split=(tr,te)
            break
    if split is None:
        raise ValueError('Cannot make held-out train/test split containing each class. Capture more INDEPENDENT packet/session IDs.')
    tr,te=split
    validator=_pipeline()
    validator.fit(X[tr],y[tr])
    pred=validator.predict(X[te])
    labels=sorted(active)
    report=classification_report(y[te],pred,labels=labels,zero_division=0,output_dict=True)
    report.update({
        'labels':labels,
        'confusion_matrix':confusion_matrix(y[te],pred,labels=labels).tolist(),
        'train_samples':len(tr),'test_samples':len(te),
        'train_sessions':sorted(set(groups[tr])),'heldout_sessions':sorted(set(groups[te])),
        'note':'Held-out accuracy is indicative only; collect multiple physical packets and hard negatives.'
    })
    # Final model is trained on all labelled examples only AFTER valid held-out evaluation.
    final=_pipeline()
    final.fit(X,y)
    path=Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'pipeline':final,'labels':labels,'validation':report,'created_at':time.time(),
                 'input_region':dataset.input_region},str(path))
    path.with_suffix('.validation.json').write_text(json.dumps(report,indent=2))
    return report


class MouthClassifier:
    def __init__(self, path):
        self.path=Path(path)
        self.payload=None
        if self.path.exists():
            self.reload()

    @property
    def ready(self):
        return self.payload is not None

    def reload(self):
        import joblib
        # Trust only a model trained by this application. joblib files are executable pickle formats.
        self.payload=joblib.load(self.path)

    def predict(self, crop):
        if not self.ready:
            return None,0.,{}
        x=mouth_features(crop).reshape(1,-1)
        p=self.payload['pipeline'].predict_proba(x)[0]
        classes=self.payload['pipeline'].classes_
        result={str(k):float(v) for k,v in zip(classes,p)}
        best=max(result,key=result.get)
        return best,result[best],result
