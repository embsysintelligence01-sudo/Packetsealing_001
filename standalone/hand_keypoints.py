"""Full-frame MediaPipe landmarks for the standalone fingertip pickup experiment."""
from types import SimpleNamespace

import cv2
import numpy as np


TIP_IDS = (4, 8, 12, 16, 20)
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
               (9, 10), (10, 11), (11, 12), (9, 13),
               (13, 14), (14, 15), (15, 16), (13, 17),
               (0, 17), (17, 18), (18, 19), (19, 20))


def home_guard_mask(hand, shape):
    """Conservative exclusion area, not an estimate of exposed bench pixels."""
    if not hasattr(hand, 'points'):
        return hand.mask
    points = np.vstack((hand.points, hand.points[0]+(hand.points[0]-hand.points[9])))
    margin = max(8, round(hand.scale*.25))
    low = np.floor(points.min(axis=0)-margin).astype(int)
    high = np.ceil(points.max(axis=0)+margin).astype(int)
    mask = np.zeros(shape[:2], np.uint8)
    x1, y1 = np.maximum(low, 0)
    x2, y2 = np.minimum(high+1, (shape[1], shape[0]))
    mask[y1:y2, x1:x2] = 255
    return mask


def home_contact_mask(hand, shape):
    """Finger-side and tip proximity; never a physical-contact assertion."""
    if not hasattr(hand, 'points'):
        return hand.mask
    mask = np.zeros(shape[:2], np.uint8)
    radius = max(8, round(hand.scale*.18))
    for index in (4, 8):
        cv2.circle(mask, tuple(np.rint(hand.points[index]).astype(int)), radius, 255, -1)
    for a, b in ((2, 3), (3, 4), (6, 7), (7, 8)):
        cv2.line(mask, tuple(np.rint(hand.points[a]).astype(int)),
                 tuple(np.rint(hand.points[b]).astype(int)), 255, radius*2+1)
    return mask


def hand_continuity_mask(hand,shape):
    """Palm/wrist support for an existing or anonymous candidate, not touch proof."""
    mask=home_contact_mask(hand,shape)
    if hasattr(hand,'points'):
        points=np.rint(hand.points).astype(np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(points[[0,1,2,5,9,13,17]]),255)
        cv2.circle(mask,tuple(points[0]),max(8,round(hand.scale*.2)),255,-1)
    return mask


class HandKeypoints:
    def __init__(self, model_path):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=4, min_hand_detection_confidence=.5,
            min_hand_presence_confidence=.5)
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.model.detect(image)
        height, width = frame.shape[:2]
        hands = []
        for landmarks in result.hand_landmarks:
            points = np.array([(p.x*width, p.y*height) for p in landmarks], np.float32)
            if not np.all(np.isfinite(points)):
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width-1)
            points[:, 1] = np.clip(points[:, 1], 0, height-1)
            # Hull is only for ROI filtering / temporal association, never an
            # assertion that pixels outside it are exposed empty bench.
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(points).astype(np.int32)), 255)
            x, y, w, h = cv2.boundingRect(mask)
            scale = max(1., float(np.linalg.norm(points[0]-points[9])))
            hands.append(SimpleNamespace(points=points, mask=mask,
                         box=(x, y, x+w, y+h), scale=scale))
        return hands

    def close(self):
        self.model.close()


def draw_hand(frame, hand, thickness=1):
    points = np.rint(hand.points).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(points[a]), tuple(points[b]), (220, 80, 220), thickness)
    for index, point in enumerate(points):
        cv2.circle(frame, tuple(point), max(2, thickness*2),
                   (0, 240, 240) if index in TIP_IDS else (220, 80, 220), -1)
