"""Normalized marker-only ROI configuration and a Qt rubber-band selector."""
import json
import math


def load_marker_roi(path):
    try:
        values=tuple(float(v) for v in json.loads(path.read_text())['roi'])
        if len(values)!=4 or not all(math.isfinite(v) for v in values):return None
        x,y,w,h=values
        return values if min(x,y)>=0 and min(w,h)>0 and x+w<=1.000001 and y+h<=1.000001 else None
    except (OSError,ValueError,KeyError,TypeError):
        return None


def save_marker_roi(path,roi):
    path.write_text(json.dumps(dict(roi=list(roi) if roi is not None else None,coordinates='normalized_full_frame'),indent=2))


def choose_marker_roi(parent,image,QtCore,QtGui,QtWidgets):
    dialog=QtWidgets.QDialog(parent)
    dialog.setWindowTitle('Select marker ROI — include the full white margin')
    layout=QtWidgets.QVBoxLayout(dialog)
    layout.addWidget(QtWidgets.QLabel('Drag a rectangle around the marker and its white margin.'))

    class Selection(QtWidgets.QLabel):
        def __init__(self):
            super().__init__()
            self.setPixmap(QtGui.QPixmap.fromImage(image))
            self.setFixedSize(image.size())
            self.band=QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle,self)
            self.origin=None
            self.selection=None

        def mousePressEvent(self,event):
            if event.button()==QtCore.Qt.LeftButton:
                self.origin=event.pos()
                self.band.setGeometry(QtCore.QRect(self.origin,QtCore.QSize()))
                self.band.show()

        def mouseMoveEvent(self,event):
            if self.origin is not None:
                self.band.setGeometry(QtCore.QRect(self.origin,event.pos()).normalized().intersected(self.rect()))

        def mouseReleaseEvent(self,event):
            if self.origin is not None:
                self.selection=QtCore.QRect(self.origin,event.pos()).normalized().intersected(self.rect())
                self.band.setGeometry(self.selection)
                self.origin=None

    selector=Selection();layout.addWidget(selector)
    buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
    layout.addWidget(buttons)
    buttons.rejected.connect(dialog.reject)
    def accept():
        rect=selector.selection
        if rect is not None and min(rect.width(),rect.height())>=12:dialog.accept()
    buttons.accepted.connect(accept)
    if dialog.exec_()!=QtWidgets.QDialog.Accepted:return None
    rect=selector.selection
    return rect.x()/image.width(),rect.y()/image.height(),rect.width()/image.width(),rect.height()/image.height()
