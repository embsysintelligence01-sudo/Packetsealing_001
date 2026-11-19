"""Fixed-reference alignment with explicit uncertainty and camera compensation."""
import cv2
import numpy as np


def reference_point(point, status):
    """Map a current-frame point into the initial fixed scene coordinates."""
    if point is None or not status or not status.get('trusted'):
        return None
    matrix=np.asarray(status['matrix'],np.float64)
    if matrix.shape==(3,3):
        point=cv2.perspectiveTransform(np.asarray(point,np.float64).reshape(1,1,2),np.linalg.inv(matrix))[0,0]
        return point if np.all(np.isfinite(point)) else None
    inverse = cv2.invertAffineTransform(matrix)
    return inverse[:, :2] @ np.asarray(point, float) + inverse[:, 2]


def scene_area_scale(point,status):
    matrix=np.asarray(status['matrix'],float)
    if matrix.shape==(2,3):return abs(float(np.linalg.det(matrix[:,:2])))
    base=reference_point(point,status)
    dx=reference_point(np.asarray(point)+[1.,0.],status)-base
    dy=reference_point(np.asarray(point)+[0.,1.],status)-base
    return 1./max(.0001,abs(float(np.linalg.det(np.column_stack((dx,dy))))))


class SceneAlignment:
    def __init__(self, gray, background_mask, max_width=960):
        self.shape = gray.shape
        self.ratio = min(1., max_width/gray.shape[1])
        size = (round(gray.shape[1]*self.ratio), round(gray.shape[0]*self.ratio))
        self.reference = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        self.mask = cv2.resize(background_mask, size, interpolation=cv2.INTER_NEAREST)
        self.points = []
        # Distribute reference features so hand/stone motion cannot dominate one cluster.
        h, w = self.reference.shape
        for row in range(3):
            for column in range(4):
                tile = np.zeros_like(self.mask)
                tile[row*h//3:(row+1)*h//3, column*w//4:(column+1)*w//4] = 255
                points = cv2.goodFeaturesToTrack(self.reference, 25, .01, 6, mask=tile & self.mask)
                if points is not None:
                    self.points.extend(points)
        self.points = np.asarray(self.points, np.float32).reshape(-1, 1, 2)
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.reference, self.mask)

    @staticmethod
    def unknown(reason, source='background_flow', **details):
        return dict(state='INSUFFICIENT_EVIDENCE', reason=reason, source=source,
                    trusted=False, matrix=None, **details)

    def fit(self, before, after, source):
        before = np.asarray(before, np.float32).reshape(-1, 2)
        after = np.asarray(after, np.float32).reshape(-1, 2)
        if len(before) < 8:
            return self.unknown('too_few_background_points', source, matches=len(before))
        matrix, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
        if matrix is None or inliers is None:
            return self.unknown('poor_tracking_matches', source, matches=len(before))
        selected = inliers.ravel() > 0
        if selected.sum() < 8 or selected.mean() < .6:
            return self.unknown('poor_tracking_matches', source, matches=len(before), inliers=int(selected.sum()))
        fixed = before[selected]
        h, w = self.reference.shape
        cells = {(min(3,int(x*4/w)), min(2,int(y*3/h))) for x,y in fixed}
        hull_area = cv2.contourArea(cv2.convexHull(fixed))
        if len(cells) < 3 or hull_area < w*h*.025:
            return self.unknown('background_points_not_distributed', source,
                                matches=len(before), inliers=int(selected.sum()))
        # Convert the small-image affine translation back to capture pixels.
        matrix[:, 2] /= self.ratio
        scale = float(np.hypot(matrix[0,0], matrix[0,1]))
        angle = float(abs(np.degrees(np.arctan2(matrix[1,0], matrix[0,0]))))
        motion = float(np.median(np.linalg.norm(after[selected]-fixed, axis=1))/self.ratio)
        stable = motion < 2.5 and .99 < scale < 1.01 and angle < 1.
        trusted = .8 <= scale <= 1.25 and angle <= 15. and motion < min(self.shape)*.25
        return dict(state='STABLE' if stable else 'CAMERA_MOVED',
                    reason='fixed_background_aligned' if stable else 'measured_camera_shift' if trusted else 'camera_change_too_large',
                    source=source, trusted=bool(trusted), matrix=matrix.tolist(),
                    motion_px=round(motion,3), scale=round(scale,5), angle_deg=round(angle,3),
                    matches=len(before), inliers=int(selected.sum()))

    def measure(self, gray, excluded_mask):
        current = cv2.resize(gray, (self.reference.shape[1],self.reference.shape[0]), interpolation=cv2.INTER_AREA)
        blocked = cv2.resize(excluded_mask, (current.shape[1],current.shape[0]), interpolation=cv2.INTER_NEAREST)
        # LK uses a neighbourhood; reject features close to a moving boundary too.
        blocked = cv2.dilate(blocked, np.ones((11,11),np.uint8))
        clear = blocked == 0
        outcome = self.unknown('too_few_background_points')
        if len(self.points) >= 8:
            after, ok, _ = cv2.calcOpticalFlowPyrLK(self.reference,current,self.points,None,maxLevel=3)
            if after is not None:
                back, reverse, _ = cv2.calcOpticalFlowPyrLK(current,self.reference,after,None,maxLevel=3)
                if back is not None:
                    valid = (ok.ravel()>0)&(reverse.ravel()>0)
                    valid &= np.linalg.norm(back-self.points,axis=2).ravel()<1.5
                    xy = np.rint(after[:,0]).astype(int)
                    inside = (xy[:,0]>=0)&(xy[:,0]<current.shape[1])&(xy[:,1]>=0)&(xy[:,1]<current.shape[0])
                    indices = np.flatnonzero(valid & inside)
                    indices = indices[clear[xy[indices,1],xy[indices,0]]]
                    outcome = self.fit(self.points[indices],after[indices],'distributed_background_flow')
                    if outcome['trusted']:
                        return outcome
        # Independent descriptor matching on the fixed bed/stone/background can
        # recover alignment when long-baseline optical flow loses its features.
        if self.descriptors is not None:
            keypoints, descriptors = self.orb.detectAndCompute(current, (clear.astype(np.uint8)*255)&self.mask)
            if descriptors is not None and len(descriptors)>=2:
                pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.descriptors,descriptors,k=2)
                matches = [pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < .7*pair[1].distance and pair[0].distance<64]
                fallback = self.fit([self.keypoints[m.queryIdx].pt for m in matches],
                                    [keypoints[m.trainIdx].pt for m in matches], 'fixed_background_descriptors')
                fallback['flow_failure'] = outcome['reason']
                if fallback['trusted'] or fallback['state'] == 'CAMERA_MOVED':
                    return fallback
                outcome['descriptor_failure'] = fallback['reason']
        return outcome
