"""Ordered rubbing sequences, human labels, and ResNet-50 + GRU classification.

No rule-based cue is presented as a trained OK/NOT_OK prediction.
"""
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np

CUES = ('visible', 'distance', 'overlap', 'relative_x', 'relative_y',
        'velocity_x', 'velocity_y', 'reversal', 'near_seconds', 'dt', 'hand_distance')
LABELS = ('NOT_OK', 'OK')


def model_factory(pretrained=False):
    import torch
    from torch import nn
    from torchvision.models import resnet50, ResNet50_Weights

    class MotionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.encoder.fc = nn.Identity()
            self.encoder.requires_grad_(False)
            self.temporal = nn.GRU(4096+len(CUES), 128, batch_first=True)
            self.head = nn.Linear(128, 2)

        def encode(self, frames):
            # [time, two views, height, width, RGB]
            x = torch.as_tensor(frames).permute(0, 1, 4, 2, 3).float()/255.
            x = (x-torch.tensor([.485, .456, .406])[None, None, :, None, None])/torch.tensor([.229, .224, .225])[None, None, :, None, None]
            self.encoder.eval()
            with torch.no_grad():
                features = [self.encoder(batch) for batch in x.flatten(0, 1).split(8)]
            return torch.cat(features).reshape(len(frames), 4096)

        def forward(self, features, cues):
            sequence, _ = self.temporal(torch.cat((features, cues), dim=-1))
            return self.head(sequence[:, -1])
    return MotionModel()


class MotionClassifier:
    def __init__(self, checkpoint):
        import torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['cues'] != list(CUES) or saved['labels'] != list(LABELS):
            raise ValueError('Incompatible motion checkpoint schema')
        self.model = model_factory()
        self.model.load_state_dict(saved['state'])
        self.model.eval()
        self.window = saved['window']

    def predict(self, frames, cues):
        import torch
        with torch.no_grad():
            features = self.model.encode(frames)
            probabilities = self.model(features[None], torch.as_tensor(cues)[None]).softmax(-1)[0]
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        return dict(label=LABELS[index] if confidence >= .8 else 'UNCERTAIN', confidence=confidence,
                    ok_probability=float(probabilities[1]))


def crop_rgb(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((224, 224, 3), np.uint8)
    return cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (224, 224)), cv2.COLOR_BGR2RGB)


class MotionSequences:
    def __init__(self, folder, window=16, checkpoint=None, session=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.classifier = MotionClassifier(checkpoint) if checkpoint else None
        self.window = self.classifier.window if self.classifier else window
        if self.window < 4:
            raise ValueError('Motion window must contain at least four frames')
        self.tracks = {}
        self.count = 0
        self.session = session or str(self.folder.resolve())
        self.status = 'UNTRAINED: collecting unlabeled motion windows'

    def update(self, frame, gold, tracked, hands, stone, roi, t, stable=True):
        results = {}
        if stone is None or not stable:
            self.tracks.clear()
            self.status = 'Waiting for one stable detected stone' if stone is None else 'Camera/stone unstable; sequence reset'
            return results
        stone = np.asarray(stone, float)
        centre = (stone[:2]+stone[2:])/2
        diagonal = max(1., float(np.linalg.norm(stone[2:]-stone[:2])))
        for missing in set(self.tracks)-set(tracked):
            del self.tracks[missing]
        for jid, observation in tracked.items():
            state = self.tracks.setdefault(jid, dict(samples=deque(maxlen=self.window), previous=None,
                velocity=None, near=0., last=None, stone=None))
            dt = t-state['last'] if state['last'] is not None else 0.
            old_stone = state['stone']
            if dt <= 0 or dt > .8 or (old_stone is not None and
                    np.linalg.norm(centre-(old_stone[:2]+old_stone[2:])/2) > diagonal*.03):
                state.update(samples=deque(maxlen=self.window), previous=None, velocity=None, near=0.)
                dt = 0.
            state['last'], state['stone'] = t, stone.copy()
            box = observation.get('detected_box', observation['box'])
            matches = []
            if observation.get('seen') == t and observation.get('visibility') != 'UNDER_HAND':
                for det in gold:
                    a, b = np.asarray(box), np.asarray(det.box)
                    intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))))
                    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
                    if union > 0 and intersection/union > .5:
                        matches.append(det)
            cue = np.zeros(len(CUES), np.float32)
            cue[9], cue[10] = dt, -1.
            target = np.zeros((224, 224, 3), np.uint8)
            visible = len(matches) == 1
            if visible:
                det = matches[0]
                visible = not any(other != jid and row.get('seen') == t and
                    det.box[0] <= (row['box'][0]+row['box'][2])/2 <= det.box[2] and
                    det.box[1] <= (row['box'][1]+row['box'][3])/2 <= det.box[3]
                    for other, row in tracked.items())
            if visible:
                yy, xx = np.nonzero(det.mask)
                visible = len(xx) > 0
            if visible:
                cog = np.array([xx.mean(), yy.mean()])
                relative = (cog-centre)/diagonal
                distance = np.linalg.norm(np.maximum(np.maximum(stone[:2]-cog, cog-stone[2:]), 0))/diagonal
                x1, y1, x2, y2 = np.rint(stone).astype(int)
                overlap = np.count_nonzero(det.mask[max(0,y1):max(0,y2), max(0,x1):max(0,x2)])/len(xx)
                velocity = (relative-state['previous'])/dt if state['previous'] is not None and dt else np.zeros(2)
                previous_velocity = state['velocity']
                reversal = (previous_velocity is not None and min(np.linalg.norm(velocity), np.linalg.norm(previous_velocity)) > .03 and
                            np.dot(velocity, previous_velocity) < -.3*np.linalg.norm(velocity)*np.linalg.norm(previous_velocity))
                state['near'] = state['near']+dt if distance < .15 else 0.
                hand_distance = min((np.linalg.norm(h.points[[4,8,12,16,20]]-cog, axis=1).min()/diagonal
                                     for h in hands if hasattr(h, 'points')), default=-1.)
                cue[:] = [1., distance, overlap, *relative, *velocity, float(reversal), state['near'], dt, hand_distance]
                state['previous'], state['velocity'] = relative, velocity
                target = crop_rgb(frame, det.box)
            else:
                state['previous'], state['velocity'], state['near'] = None, None, 0.
            # Shared context preserves the hand, stone and bench; a second view
            # identifies the target jewel without learning arbitrary ID labels.
            context = crop_rgb(frame, roi)
            state['samples'].append((np.stack((context, target)), cue, t))
            if len(state['samples']) == self.window:
                frames, cues, times = map(np.stack, zip(*state['samples']))
                self.count += 1
                name = f'J{jid}_{self.count:06d}'
                np.savez_compressed(self.folder/f'{name}.npz', frames=frames, cues=cues, times=times)
                metadata = dict(label=None, session=self.session, jewel_id=int(jid),
                    cue_names=list(CUES), start=float(times[0]), end=float(times[-1]), stone_box=stone.tolist())
                (self.folder/f'{name}.json').write_text(json.dumps(metadata, indent=2))
                result = self.classifier.predict(frames, cues) if self.classifier else dict(label='UNTRAINED', confidence=None)
                # Missing targets cannot produce a conclusive learned decision.
                if self.classifier and cues[:, 0].mean() < .75:
                    result['label'] = 'INSUFFICIENT_VISIBILITY'
                results[jid] = result
                self.status = f'J{jid}: {result["label"]} | saved {self.count} windows'
                state['samples'].clear()  # non-overlapping training samples
        return results


def review(folder):
    """Human labels apply to complete windows; never auto-label with motion rules."""
    for path in sorted(Path(folder).rglob('*.npz')):
        metadata_path = path.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('label') in LABELS:
            continue
        with np.load(path) as data:
            frames = data['frames'].copy()
        index = 0
        while True:
            context, target = frames[index % len(frames)]
            display = cv2.cvtColor(np.concatenate((context, target), axis=1), cv2.COLOR_RGB2BGR)
            display = cv2.resize(display, (1008, 448))
            cv2.putText(display, 'O: OK | N: NOT OK | S: skip | Q: quit', (10, 25), 0, .6, (0,255,255), 2)
            cv2.imshow(path.name, display)
            key = cv2.waitKey(180) & 255
            index += 1
            if key in (ord('o'), ord('n')):
                metadata['label'] = 'OK' if key == ord('o') else 'NOT_OK'
                metadata_path.write_text(json.dumps(metadata, indent=2))
                break
            if key == ord('s'):
                break
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
        cv2.destroyAllWindows()


def train(folder, checkpoint, epochs=15):
    import torch
    torch.manual_seed(7)
    records = []
    for path in sorted(Path(folder).rglob('*.npz')):
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata.get('label') in LABELS:
            records.append((path, metadata))
    sessions = sorted({m['session'] for _, m in records})
    if len(sessions) < 2:
        raise ValueError('Label separate recording sessions for training and validation (both classes in each).')
    # A session never contributes frames to both train and validation.
    validation_session = sessions[-1]
    train_records = [(p,m) for p,m in records if m['session'] != validation_session]
    val_records = [(p,m) for p,m in records if m['session'] == validation_session]
    for subset in (train_records, val_records):
        if {m['label'] for _,m in subset} != set(LABELS):
            raise ValueError('Both OK and NOT_OK labels are required in training and held-out validation sessions.')
    model = model_factory(pretrained=True)
    def prepare(subset):
        prepared = []
        for path, meta in subset:
            with np.load(path) as data:
                if meta['cue_names'] != list(CUES):
                    raise ValueError('Incompatible cue schema')
                features = model.encode(data['frames'])
                prepared.append((features, torch.tensor(data['cues']), LABELS.index(meta['label'])))
        return prepared
    training, validation = prepare(train_records), prepare(val_records)
    window = training[0][0].shape[0]
    if any(x.shape[0] != window for x,_,_ in training+validation):
        raise ValueError('All windows must have the same length')
    optimizer = torch.optim.Adam(list(model.temporal.parameters())+list(model.head.parameters()), lr=1e-3)
    best = float('inf')
    for epoch in range(epochs):
        model.temporal.train(); model.head.train()
        for index in torch.randperm(len(training)):
            features, cues, label = training[index]
            loss = torch.nn.functional.cross_entropy(model(features[None], cues[None]), torch.tensor([label]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            losses, correct = [], 0
            for features, cues, label in validation:
                logits = model(features[None], cues[None])
                losses.append(float(torch.nn.functional.cross_entropy(logits, torch.tensor([label]))))
                correct += int(logits.argmax() == label)
        value = float(np.mean(losses))
        print(f'Epoch {epoch+1}: validation loss={value:.4f}, accuracy={correct/len(validation):.3f}')
        if value < best:
            best = value
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state=model.state_dict(), cues=list(CUES), labels=list(LABELS), window=window,
                validation_session=validation_session, validation_accuracy=correct/len(validation)), checkpoint)
