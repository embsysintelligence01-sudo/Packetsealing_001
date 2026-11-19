"""Single fixed-bed marker reference for the separate ArUco trigger mode."""
import cv2
import numpy as np


class ArucoReference:
    def __init__(self, roi=None):
        if not hasattr(cv2, 'aruco'):
            raise RuntimeError('ArUco mode requires opencv-contrib-python in the tracker environment.')
        self.roi=roi
        self.search_box=None
        self.detectors=[]
        for name in ('DICT_4X4_1000','DICT_5X5_1000','DICT_6X6_1000','DICT_7X7_1000','DICT_ARUCO_ORIGINAL','DICT_APRILTAG_16h5','DICT_APRILTAG_25h9','DICT_APRILTAG_36h10','DICT_APRILTAG_36h11'):
            parameters=cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
            self.detectors.append((name,cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,name)),parameters)))
        self.identity=None
        self.reference=None
        self.current=None
        self.probe=None
        self.hits=0
        self.status=dict(state='WAITING_MARKER',reason='Show a sharp marker at the top-left of the fixed bed.')

    def detect(self,frame):
        h,w=frame.shape[:2]
        selected=self.roi if self.roi is not None else (0.,0.,.55,.55)
        x,y,rw,rh=selected
        padding=max(8,round(max(rw*w,rh*h)*.1))
        x1=max(0,round(x*w)-padding);y1=max(0,round(y*h)-padding)
        x2=min(w,round((x+rw)*w)+padding);y2=min(h,round((y+rh)*h)+padding)
        self.search_box=(x1,y1,x2,y2)
        self.current=None
        if x2-x1<12 or y2-y1<12:
            self.hits=0
            self.status=dict(state='MARKER_UNAVAILABLE',reason='marker_roi_too_small')
            return None
        gray=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
        factor=min(3.,max(1.,640/max(gray.shape)))
        if factor>1.:
            gray=cv2.resize(gray,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
        candidates=[];reasons={};decoded=[];rejected_quads=0
        for name,detector in self.detectors:
            if self.identity is not None and name!=self.identity[0]:continue
            corners,ids,rejected=detector.detectMarkers(gray)
            rejected_quads+=len(rejected)
            if ids is None:continue
            for polygon,identifier in zip(corners,ids.ravel()):
                decoded.append(dict(dictionary=name,marker_id=int(identifier)))
                if self.identity is not None and (name,int(identifier))!=self.identity:
                    reasons['wrong_marker_id']=reasons.get('wrong_marker_id',0)+1
                    continue
                points=polygon.reshape(4,2)/factor+np.array([x1,y1],np.float32)
                sides=np.linalg.norm(points-np.roll(points,1,axis=0),axis=1)
                reason=('marker_too_small' if min(sides)<24 else
                        'marker_excessively_skewed' if max(sides)/min(sides)>1.8 else
                        'marker_clipped_at_frame_edge' if np.any(points<5) or np.any(points[:,0]>w-5) or np.any(points[:,1]>h-5) else None)
                if reason:
                    reasons[reason]=reasons.get(reason,0)+1
                    continue
                candidates.append((name,int(identifier),points))
        details=dict(search_roi=list(self.search_box),roi_normalized=list(selected),upscale=factor,
                     decoded=decoded,rejections=reasons,rejected_quadrilaterals=rejected_quads)
        if len(candidates)!=1:
            reason='multiple_marker_candidates' if len(candidates)>1 else next(iter(reasons),'no_supported_marker_decoded')
            self.status=dict(state='MARKER_UNAVAILABLE',reason=reason,detected_candidates=len(candidates),**details)
            self.hits=0
            return None
        name,identifier,points=candidates[0]
        identity=(name,identifier)
        # Confirm dictionary/ID, without requiring stationary marker corners.
        consistent=self.probe is not None and self.probe[0]==identity
        self.hits=self.hits+1 if consistent else 1
        self.probe=(identity,points.copy())
        self.current=points
        if self.identity is None and self.hits>=3:self.identity=identity
        self.status=dict(state='MARKER_VISIBLE' if self.identity is not None else 'ACQUIRING_MARKER',
                         dictionary=name,marker_id=identifier,corners=points.tolist(),identity_observations=self.hits,**details)
        return points

    def freeze(self):
        if self.identity is None or self.current is None or self.hits<3:
            return False
        self.reference=self.current.copy()
        self.status['reference_corners']=self.reference.tolist()
        return True

    def alignment(self,shape):
        if self.reference is None or self.current is None:
            return None
        homography=cv2.getPerspectiveTransform(self.reference.astype(np.float32),self.current.astype(np.float32))
        h,w=shape[:2]
        probes=np.array([[0,0],[w,0],[w,h],[0,h],[w/2,h/2]],np.float32)
        denominator=probes@homography[2,:2]+homography[2,2]
        if np.any(denominator<=.2) or not np.all(np.isfinite(homography)):
            self.status['reason']='unreliable_marker_projection'
            return None
        projected=cv2.perspectiveTransform(probes[None],homography)[0]
        scale=cv2.contourArea(projected[:4])/max(1.,w*h)
        displacement=float(np.max(np.linalg.norm(projected-probes,axis=1)))
        if not .64<scale<1.56 or displacement>min(w,h)*.25:
            self.status['reason']='marker_projection_change_too_large'
            return None
        return dict(state='STABLE' if displacement<2.5 else 'CAMERA_MOVED',
                    reason='aruco_bed_reference',source='aruco_single_marker',trusted=True,
                    matrix=homography.tolist(),motion_px=round(displacement,3),
                    marker_id=self.identity[1],dictionary=self.identity[0],
                    limitation='single_corner_marker_planar_alignment')

    def draw(self,frame,scale=1.):
        if self.search_box is not None:
            x1,y1,x2,y2=self.search_box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,0),max(1,round(scale)))
        if self.current is not None:
            cv2.polylines(frame,[np.rint(self.current).astype(np.int32)],True,(0,255,0),max(1,round(scale)))
        label='ArUco: '+self.status['state']+' '+str(self.status.get('marker_id',''))+' '+self.status.get('reason','')
        cv2.putText(frame,label,(round(12*scale),round(24*scale)),cv2.FONT_HERSHEY_SIMPLEX,.45*scale,(0,220,255),max(1,round(scale)))


class ArucoHomeMixin:
    """Adds marker alignment without changing HomePositionEngine itself."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.aruco=ArucoReference()
        self.marker_t=None
        self.stone_reference=None
        self.stone_seen=None

    def prepare_marker(self,frame,t):
        if self.marker_t!=t:
            self.aruco.detect(frame)
            if self.aruco.reference is not None:
                self.aruco.status['reference_corners']=self.aruco.reference.tolist()
            self.marker_t=t

    def adopt_tracks(self,frame,tracks,bags,roi_mask,t,hands=()):
        self.prepare_marker(frame,t)
        if not self.jewels and (self.aruco.identity is None or self.aruco.current is None or self.aruco.hits<3):
            self.assignment_probe=None
            self.assignment_since=None
            self.message='ArUco mode: waiting for 3 matching marker IDs, then clear-home capture.'
            return
        super().adopt_tracks(frame,tracks,bags,roi_mask,t,hands)

    def set_references(self,frame,indexed_gold,bags,roi_mask,t):
        self.prepare_marker(frame,t)
        if not self.aruco.freeze():return
        super().set_references(frame,indexed_gold,bags,roi_mask,t)

    def update(self,frame,gold,hands,bags,roi_mask,t,tracked=None,legacy_ids=()):
        self.prepare_marker(frame,t)
        if not self.jewels:return
        super().update(frame,gold,hands,bags,roi_mask,t,tracked,legacy_ids)

    def assess_scene(self,frame,excluded_mask):
        marker=self.aruco.alignment(frame.shape)
        if marker is not None:
            marker.update(state='MARKER_ALIGNED',stability_checked=False,compensation_available=True)
            self.scene_status=marker
            return True
        stable=super().assess_scene(frame,excluded_mask)
        self.scene_status['aruco_status']=self.aruco.status.copy()
        return stable

    def resolve_stone(self,detected,frame,t):
        from .scene_alignment import reference_point
        status=getattr(self,'scene_status',None)
        if detected is not None:
            self.aruco.status['stone_source']='current_detection'
            if status and status.get('trusted'):
                x1,y1,x2,y2=detected
                self.stone_reference=np.array([reference_point(p,status) for p in ((x1,y1),(x2,y1),(x2,y2),(x1,y2))])
                self.stone_seen=t
            return detected
        # A brief occlusion may reuse the last stone location only while the
        # actual marker is visible and its transform is trustworthy.
        if (self.stone_reference is None or self.stone_seen is None or t-self.stone_seen>2. or
                not status or not status.get('trusted') or status.get('source')!='aruco_single_marker'):
            self.aruco.status['stone_source']='unavailable'
            return None
        matrix=np.asarray(status['matrix'],float)
        corners=cv2.perspectiveTransform(self.stone_reference.reshape(1,4,2),matrix)[0]
        low=np.maximum(np.floor(corners.min(0)).astype(int),0)
        high=np.minimum(np.ceil(corners.max(0)).astype(int),(frame.shape[1],frame.shape[0]))
        self.aruco.status['stone_source']='marker_aligned_recent_detection'
        self.aruco.status['stone_age_s']=round(t-self.stone_seen,3)
        return tuple(map(int,(*low,*high))) if np.all(high>low) else None
