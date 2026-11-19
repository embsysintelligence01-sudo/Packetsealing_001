"""Packet bounding-box dataset collection and optional CPU state classification."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

if __package__:
    from .mouth_core import (TemporalConsensus, LABELS, crop_roi,
                             to_working_size, diagnostic_metrics)
    from .mouth_training import MouthDataset, MouthClassifier, train_validated
else:
    from mouth_core import (TemporalConsensus, LABELS, crop_roi,
                            to_working_size, diagnostic_metrics)
    from mouth_training import MouthDataset, MouthClassifier, train_validated

ROOT = Path(__file__).resolve().parent


def saved_camera_resolution():
    """Read the same USB resolution as the sealing app without loading its tracker."""
    tracker = Path(os.environ.get('JEWEL_TRACKER_DIR', ROOT.parent.parent/'jewel_tracker'/'jewel_tracker'))
    try:
        saved = json.loads((tracker/'camera.json').read_text())
        width,height = map(int,str(saved['usb']['resolution']).lower().split('x'))
        if width > 0 and height > 0:
            return width,height
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 2560,1440


class FrameCanvas(QtWidgets.QWidget):
    selected = QtCore.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.frame = None
        self.image = None
        self.roi = None
        self.origin = None
        self.current = None
        self.setMinimumSize(600, 365)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CrossCursor)

    def set_content(self, frame, roi):
        self.frame = frame
        self.roi = roi
        if frame is not None:
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2]
            self.image=QtGui.QImage(rgb.data,w,h,3*w,QtGui.QImage.Format_RGB888).copy()
        else:
            self.image=None
        self.update()

    def _scale(self):
        if self.image is None:
            return None
        iw,ih=self.image.width(),self.image.height()
        scale=min(self.width()/iw,self.height()/ih)
        rw,rh=iw*scale,ih*scale
        return scale,(self.width()-rw)/2,(self.height()-rh)/2

    def _map_to_image(self,point):
        info=self._scale()
        if info is None:
            return None
        scale,px,py=info
        x=(point.x()-px)/scale
        y=(point.y()-py)/scale
        if not 0<=x<self.image.width() or not 0<=y<self.image.height():
            return None
        return int(x),int(y)

    def paintEvent(self,event):
        painter=QtGui.QPainter(self)
        painter.fillRect(self.rect(),QtGui.QColor('#17212b'))
        info=self._scale()
        if self.image is None or info is None:
            painter.setPen(QtGui.QColor('#d7dce0'))
            painter.drawText(self.rect(),QtCore.Qt.AlignCenter,'Open an image/video or a camera')
            return
        scale,ox,oy=info
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform,True)
        target=QtCore.QRectF(ox,oy,self.image.width()*scale,self.image.height()*scale)
        painter.drawImage(target,self.image)
        if self.roi is not None:
            x,y,w,h=self.roi
            painter.setPen(QtGui.QPen(QtGui.QColor('#20e58c'),2,QtCore.Qt.SolidLine))
            painter.drawRect(QtCore.QRectF(ox+x*scale,oy+y*scale,w*scale,h*scale))
        if self.origin is not None and self.current is not None:
            x1,y1=self.origin;x2,y2=self.current
            painter.setPen(QtGui.QPen(QtGui.QColor('#f9c74f'),2,QtCore.Qt.DashLine))
            painter.drawRect(QtCore.QRectF(ox+min(x1,x2)*scale,oy+min(y1,y2)*scale,
                                           abs(x2-x1)*scale,abs(y2-y1)*scale))



class TrainWorker(QtCore.QThread):
    trained=QtCore.pyqtSignal(dict)
    failed=QtCore.pyqtSignal(str)
    def __init__(self, dataset, output):
        super().__init__()
        self.dataset=dataset
        self.output=output

    def run(self):
        try:
            self.trained.emit(train_validated(self.dataset,self.output))
        except Exception as exc:
            self.failed.emit(str(exc))


class MouthWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Packet Mouth Lab — independent CPU proof of concept')
        self.resize(1210,790)
        self.dataset=MouthDataset(ROOT/'dataset'/'packet_bbox', input_region='packet_bbox')
        self.model_path=ROOT/'models'/'packet_bbox_classifier.joblib'
        self.classifier=None
        self.temporal=TemporalConsensus()
        self.cap=None
        self.source_kind=None
        self.source_name=None
        self.playing=False
        self.frame=None
        self.source_time=None
        self.last_prediction=None
        self.confirmed='UNCONFIRMED'
        self.packet_detector=None
        self.packet_detections=[]
        self.packet_error=None
        self.capture_requested=None
        self.capture_size=None

        central=QtWidgets.QWidget();self.setCentralWidget(central)
        layout=QtWidgets.QVBoxLayout(central)
        controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        for label,callback in [('Open image',self.open_image),('Open video',self.open_video),
                               ('Open camera',self.open_camera),('Pause / resume',self.pause_resume)]:
            b=QtWidgets.QPushButton(label);b.clicked.connect(callback);controls.addWidget(b)
        controls.addStretch()
        self.capture_status=QtWidgets.QLabel('Camera capture uses the saved resolution (default 2560×1440).')
        layout.addWidget(self.capture_status)
        content=QtWidgets.QHBoxLayout();layout.addLayout(content,1)
        self.canvas=FrameCanvas();self.canvas.setCursor(QtCore.Qt.ArrowCursor)
        content.addWidget(self.canvas,3)
        side=QtWidgets.QVBoxLayout(); content.addLayout(side,1)
        side.addWidget(QtWidgets.QLabel('<b>PACKET — existing ONNX model</b>'))
        self.use_packet=QtWidgets.QCheckBox('Detect packet (ONNX + false-positive filter)')
        self.use_packet.setChecked(True)
        self.use_packet.toggled.connect(self.packet_mode_changed)
        side.addWidget(self.use_packet)
        self.packet_status=QtWidgets.QLabel('Packet detector ready to load on first frame.')
        self.packet_status.setWordWrap(True);side.addWidget(self.packet_status)
        b=QtWidgets.QPushButton('Save detected packet frame (unlabelled)')
        b.clicked.connect(self.save_packet_sample);side.addWidget(b)
        side.addWidget(QtWidgets.QLabel('<b>CLASSIFICATION INPUT — packet bbox</b>'))
        self.stage_a=QtWidgets.QLabel('The detected packet box is used automatically.')
        self.stage_a.setWordWrap(True);side.addWidget(self.stage_a)
        side.addSpacing(12)
        side.addWidget(QtWidgets.QLabel('<b>STAGE B — open/side/closed</b>'))
        self.use_classifier=QtWidgets.QCheckBox('Enable trained packet classification')
        self.use_classifier.setToolTip('Tick to use the trained packet classifier. If none exists, collect labels and train first.')
        self.use_classifier.toggled.connect(self.change_mode)
        side.addWidget(self.use_classifier)
        self.stage_b=QtWidgets.QLabel('Classification off. Select a class and save the detected packet crop.')
        self.stage_b.setWordWrap(True);side.addWidget(self.stage_b)
        self.metrics=QtWidgets.QLabel('Frame-only geometry metrics are NOT proof of an opening.')
        self.metrics.setWordWrap(True);side.addWidget(self.metrics)
        side.addSpacing(12)
        side.addWidget(QtWidgets.QLabel('<b>Label the detected packet crop</b>'))
        self.label_box=QtWidgets.QComboBox();self.label_box.addItems(LABELS);side.addWidget(self.label_box)
        side.addWidget(QtWidgets.QLabel('Physical packet / recording group ID:'))
        self.session_input=QtWidgets.QLineEdit('packet_001');side.addWidget(self.session_input)
        b=QtWidgets.QPushButton('Save manually labelled example');b.clicked.connect(self.label_sample);side.addWidget(b)
        self.counts_label=QtWidgets.QLabel('');side.addWidget(self.counts_label)
        side.addSpacing(12)
        b=QtWidgets.QPushButton('Train + held-out evaluation (CPU)');b.clicked.connect(self.start_train);side.addWidget(b)
        self.train_button=b
        self.validation=QtWidgets.QLabel('Held-out evaluation required before model activation.')
        self.validation.setWordWrap(True);side.addWidget(self.validation)
        b=QtWidgets.QPushButton('Export current evidence frame');b.clicked.connect(self.save_evidence);side.addWidget(b)
        side.addStretch()
        layout.addWidget(QtWidgets.QLabel('Packet bbox is the classifier and dataset input. Untick classification to collect labels; tick after training to display predictions.'))
        self.statusBar().showMessage('Ready. No changes are made to your main packet-sealing project.')
        self.timer=QtCore.QTimer(self);self.timer.setInterval(120);self.timer.timeout.connect(self.next_frame)
        self._refresh_counts()
        if self.classifier is not None and self.classifier.ready:
            self.validation.setText(self._validation_text(self.classifier.payload['validation']))

    def packet_mode_changed(self, enabled):
        self._detect_packet()
        self._analyze(fresh=False)

    def _detect_packet(self):
        # Results always belong to self.frame; never reuse a previous detection.
        self.packet_detections=[];self.packet_error=None
        if not self.use_packet.isChecked():
            self.packet_status.setText('Packet detector disabled.')
            return
        if self.frame is None:
            self.packet_status.setText('Open a source to detect packets.')
            return
        try:
            if self.packet_detector is None:
                if str(ROOT.parent) not in sys.path:
                    sys.path.insert(0,str(ROOT.parent))
                from standalone.models import PacketBagDetector
                self.packet_detector=PacketBagDetector(ROOT.parent/'model onnx')
            self.packet_detections=self.packet_detector.detect(self.frame)
            count=len(self.packet_detections)
            if count==1:
                d=self.packet_detections[0]
                self.packet_status.setText(f'PACKET DETECTED | score={d.confidence:.1%}\nBox={d.box}\nUsing this entire box for classification and collection.')
            else:
                self.packet_status.setText(f'{count} packets detected. Collection requires exactly one visible packet.')
        except Exception as exc:
            self.packet_error=str(exc)
            self.packet_status.setText('Packet detection failed: '+str(exc))

    def _packet_context(self, label=None):
        if len(self.packet_detections)!=1:
            raise ValueError('Show exactly one detected packet before collecting. No stale packet box will be saved.')
        return self.dataset.add_packet(self.frame,self.packet_detections[0],self.session_input.text(),
                                       {'name':self.source_name,'kind':self.source_kind,'time':self.source_time},
                                       label=label, label_region='packet_bbox')

    def save_packet_sample(self):
        try:
            path=self._packet_context()
            self._refresh_counts()
            self.statusBar().showMessage('Saved packet collection: '+path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self,'Packet collection unavailable',str(exc))

    def change_mode(self, enabled):
        self.temporal.reset();self.confirmed='UNCONFIRMED';self.last_prediction=None
        if enabled:
            try:
                self.classifier=MouthClassifier(self.model_path)
                if self.classifier.ready and self.classifier.payload.get('input_region')!='packet_bbox':
                    raise ValueError('Train a classifier on full packet bounding boxes first.')
                if not self.classifier.ready:
                    message='No trained packet-box classifier yet. Save labelled packet crops, then click Train + held-out evaluation.'
                    self.stage_b.setText(message)
                    self.validation.setText(message)
                    self.statusBar().showMessage(message)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self,'Classifier unavailable',str(exc))
                self.use_classifier.setChecked(False)
                return
        self._analyze(fresh=False)

    def _validation_text(self,r):
        return f"Model held-out accuracy: {r['accuracy']:.1%}; test images: {r['test_samples']}; held-out groups: {len(r['heldout_sessions'])}. Must validate on site."

    def _refresh_counts(self):
        packets=sum(1 for _ in (self.dataset.root/'packet_frames').glob('*/metadata.json'))
        self.counts_label.setText('Packet labels: '+str(self.dataset.counts())+f'\nPacket frames: {packets}')

    def _stop(self):
        self.timer.stop()
        self.playing=False
        if self.cap is not None:
            self.cap.release();self.cap=None

    def _new_source(self,kind,name):
        self._stop();self.source_kind=kind;self.source_name=name
        self.frame=None;self.source_time=None;self.last_prediction=None
        self.capture_requested=None;self.capture_size=None
        self.capture_status.setText('Waiting for source frame.')
        self.packet_detections=[];self.packet_error=None
        self.packet_status.setText('Waiting for current packet detection.')
        self.temporal.reset();self.confirmed='UNCONFIRMED'
        self.stage_b.setText('Waiting for the packet bounding box.')

    def open_image(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open image',str(ROOT/'sample_images'),
                    'Images (*.jpg *.jpeg *.png *.bmp)')
        if not path:return
        frame=cv2.imread(path)
        if frame is None:
            QtWidgets.QMessageBox.warning(self,'Read failure','Could not open image.');return
        self._new_source('image',path)
        self.frame=to_working_size(frame)
        self._detect_packet();self._analyze(fresh=False)
        self.statusBar().showMessage('Image loaded. Select the packet class and save, or enable trained classification.')

    def _open_capture(self,source,kind):
        cap=cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            QtWidgets.QMessageBox.warning(self,'Source failure',f'Cannot open {source}')
            return
        requested=None
        if kind=='camera':
            requested=saved_camera_resolution()
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,requested[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,requested[1])
        self._new_source(kind,str(source))
        self.capture_requested=requested
        self.cap=cap;self.playing=True;self.timer.start();self.next_frame()

    def open_video(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open video',str(ROOT),
                          'Videos (*.mp4 *.avi *.mov *.mkv)')
        if path:self._open_capture(path,'video')

    def open_camera(self):
        index,ok=QtWidgets.QInputDialog.getInt(self,'Camera index','Camera number:',0,0,12)
        if ok:self._open_capture(index,'camera')

    def pause_resume(self):
        if self.source_kind=='image':return
        if self.cap is not None:
            self.playing=not self.playing
            if self.playing:self.timer.start()
            else:self.timer.stop()
            self.statusBar().showMessage('Playing' if self.playing else 'Paused. Label the detected packet or train now.')

    def packet_roi(self):
        if self.frame is None or len(self.packet_detections)!=1:
            return None
        x1,y1,x2,y2=self.packet_detections[0].box
        return (x1,y1,x2-x1,y2-y1)

    def next_frame(self):
        if self.cap is None or not self.playing:return
        ok,frame=self.cap.read()
        if not ok:
            self.timer.stop();self.playing=False
            self.statusBar().showMessage('Video finished. Evidence can still be exported.')
            return
        self.frame=to_working_size(frame)
        self.capture_size=(frame.shape[1],frame.shape[0])
        actual=f'{self.capture_size[0]}×{self.capture_size[1]}'
        working=f'{self.frame.shape[1]}×{self.frame.shape[0]}'
        requested=(f'Requested {self.capture_requested[0]}×{self.capture_requested[1]} | '
                   if self.capture_requested else '')
        self.capture_status.setText(f'{requested}Actual capture {actual} | Processing {working}')
        self.source_time=(self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if self.source_kind=='video'
                          else time.monotonic())
        self._detect_packet()
        self._analyze(fresh=True)

    def _analyze(self,fresh):
        if self.frame is None:return
        roi=self.packet_roi()
        self.stage_a.setText(f'Packet bbox (x,y,w,h): {roi}' if roi else 'Exactly one detected packet is required.')
        crop=crop_roi(self.frame,roi)
        metrics=diagnostic_metrics(crop)
        self.metrics.setText('Uncalibrated diagnostics: '+str(metrics))
        if crop is None:
            self.stage_b.setText('Unconfirmed: exactly one detected packet is required.')
            self.last_prediction=None;self.confirmed='UNCONFIRMED'
            self.temporal.reset()
        elif not self.use_classifier.isChecked():
            self.last_prediction=None
            self.temporal.reset();self.confirmed='UNCONFIRMED'
            self.stage_b.setText('Classification off — dataset collection.\nSelect OPEN_CAMERA, OPEN_SIDE, CLOSED_FOLDED or UNCLEAR and save the packet crop.')
        elif self.classifier is None or not self.classifier.ready:
            self.stage_b.setText('No packet classifier yet. Collect labelled packet crops and train first.')
            self.last_prediction=None;self.confirmed='UNCONFIRMED'
            self.temporal.reset()
        else:
            label,p,probs=self.classifier.predict(crop)
            self.last_prediction={'label':label,'probability':p,'probabilities':probs}
            if fresh and self.source_time is not None and self.packet_detections[0].confidence>=.65:
                self.confirmed=self.temporal.add(label,p,self.source_time)
            else:
                self.temporal.reset()
                self.confirmed='FRAME_ONLY_NOT_TEMPORAL' if self.source_kind=='image' else 'UNCONFIRMED'
            self.stage_b.setText(f'Frame classification: {label} ({p:.1%})\n'
                                  f'Temporal evidence: {self.confirmed}\n'
                                  'Class probabilities are not guarantees. No actual insertion is proven.')
        display=self.packet_overlay(self.frame)
        self.canvas.set_content(display,None)

    def packet_overlay(self, frame):
        display=frame.copy()
        for detection in self.packet_detections:
            selected=detection.mask>0
            display[selected]=(display[selected]*.78+np.array([255,150,0])*.22).astype(np.uint8)
            x1,y1,x2,y2=detection.box
            cv2.rectangle(display,(x1,y1),(x2,y2),(255,150,0),2)
            mode='collection'
            if self.use_classifier.isChecked() and (self.classifier is None or not self.classifier.ready):
                mode='classifier not trained'
            text=f'Packet {detection.confidence:.0%} | {mode}'
            if len(self.packet_detections)==1 and self.use_classifier.isChecked() and self.last_prediction:
                text=f"{self.last_prediction['label']} {self.last_prediction['probability']:.0%}"
            cv2.putText(display,text,(x1,max(18,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(255,150,0),2)
        return display

    def label_sample(self):
        if self.frame is None:
            QtWidgets.QMessageBox.warning(self,'No frame','Open image/video/camera first.');return
        crop=crop_roi(self.frame,self.packet_roi())
        if crop is None:
            QtWidgets.QMessageBox.warning(self,'No packet','Show exactly one detected packet first.');return
        try:
            context=self._packet_context(self.label_box.currentText())
            dest=self.dataset.add(crop,self.label_box.currentText(),self.session_input.text(),context=context)
            self._refresh_counts()
            self.statusBar().showMessage(f'Saved labelled packet bbox: {dest.name}')
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self,'Save failed',str(exc))

    def start_train(self):
        if self.playing:
            QtWidgets.QMessageBox.information(self,'Pause first','Pause video/camera before training to avoid inference interference.')
            return
        if getattr(self,'trainer',None) is not None and self.trainer.isRunning():return
        self.train_button.setEnabled(False)
        self.validation.setText('Training + evaluating on held-out packet/session IDs ...')
        self.trainer=TrainWorker(self.dataset,self.model_path)
        self.trainer.trained.connect(self.train_done)
        self.trainer.failed.connect(self.train_failed)
        self.trainer.finished.connect(lambda:self.train_button.setEnabled(True))
        self.trainer.start()

    def train_done(self,report):
        self.classifier=MouthClassifier(self.model_path)
        self.use_classifier.setEnabled(True)
        self.use_classifier.setChecked(True)
        self.temporal.reset();self.confirmed='UNCONFIRMED'
        self.validation.setText(self._validation_text(report))
        self.statusBar().showMessage('New local classifier activated after grouped evaluation. Review confusion matrix file.')
        self._analyze(fresh=False)

    def train_failed(self,error):
        self.validation.setText('NOT TRAINED: '+error)
        self.statusBar().showMessage('Validation blocked training. Collect more independent examples.')

    def save_evidence(self):
        roi=self.packet_roi()
        if roi is None:
            QtWidgets.QMessageBox.warning(self,'Missing packet','Show exactly one detected packet first.');return
        output=ROOT/'evidence'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True,exist_ok=True)
        annotated=self.packet_overlay(self.frame)
        x,y,w,h=roi
        cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,200,80),2)
        cv2.imwrite(str(output/'frame.jpg'),annotated)
        crop=crop_roi(self.frame,roi)
        if crop is not None:cv2.imwrite(str(output/'packet_crop.jpg'),crop)
        info={
            'source':self.source_name,'source_kind':self.source_kind,'classification_roi_xywh':roi,'input_region':'packet_bbox',
            'frame_prediction':self.last_prediction,'temporal_evidence':self.confirmed,
            'mode':'trained_classifier' if self.use_classifier.isChecked() else 'collection',
            'source_time':self.source_time,
            'packet_detections':[{'box_xyxy':list(d.box),'confidence':d.confidence} for d in self.packet_detections],
            'packet_detection_error':self.packet_error,
            'warning':'A tracked/categorised mouth is not proof that jewellery entered or remained inside.'}
        (output/'evidence.json').write_text(json.dumps(info,indent=2))
        self.statusBar().showMessage('Saved evidence: '+str(output))

    def closeEvent(self,event):
        self._stop()
        if getattr(self,'trainer',None) is not None and self.trainer.isRunning():
            self.trainer.wait()
        super().closeEvent(event)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-ui',action='store_true',help='Construct UI and exit without opening a camera')
    parser.add_argument('--source',type=Path,help='Open a local image or video')
    args=parser.parse_args(argv)
    # OpenCV wheels inject their own incompatible Qt plugin/font directories.
    # This window uses PyQt5 for display, not OpenCV's HighGUI.
    for name in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):
        if '/cv2/' in os.environ.get(name,'').replace('\\','/'):
            os.environ.pop(name,None)
    os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH',
                          QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))
    app=QtWidgets.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    window=MouthWindow()
    if args.source:
        if not args.source.is_file():
            parser.error(f'Source does not exist: {args.source}')
        frame=cv2.imread(str(args.source)) if args.source.suffix.lower() in ('.jpg','.jpeg','.png','.bmp') else None
        if frame is not None:
            window._new_source('image',str(args.source))
            window.frame=to_working_size(frame)
            window._detect_packet();window._analyze(fresh=False)
        else:
            window._open_capture(str(args.source),'video')
    window.show()
    if args.smoke_ui:
        app.processEvents()
        window.close()
        print('PASS: Packet Mouth Lab UI constructed; mouth classifier disabled, packet detector '+('enabled.' if window.use_packet.isChecked() else 'disabled.'))
        return 0
    return app.exec_()


if __name__=='__main__':
    sys.exit(main())
