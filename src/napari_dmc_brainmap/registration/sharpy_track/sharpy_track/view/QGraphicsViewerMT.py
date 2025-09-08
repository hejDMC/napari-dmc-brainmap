from PyQt5.QtWidgets import QGraphicsView
from PyQt5 import QtCore

class QGraphicsViewMT(QGraphicsView):
    mouseMoved = QtCore.pyqtSignal()
    mouseEntered = QtCore.pyqtSignal()
    mouseLeft = QtCore.pyqtSignal()
    mouseClicked = QtCore.pyqtSignal()
    """
    Custom QGraphicsView with mouse tracking and enter/leave signals
    """
    def __init__(self,scene) -> None:
        super().__init__(scene)

    def mouseMoveEvent(self,event):
        self.cursorPos = [event.pos().x(),event.pos().y()]
        self.mouseMoved.emit()
        super(QGraphicsViewMT, self).mouseMoveEvent(event)
    
    def mousePressEvent(self, event):
        # store click position and emit click signal
        self.clickPos = [event.pos().x(), event.pos().y()]
        self.mouseClicked.emit()
        super(QGraphicsViewMT, self).mousePressEvent(event)
    
    def enterEvent(self, event):
        """Override enterEvent to emit custom signal"""
        self.mouseEntered.emit()
        super(QGraphicsViewMT, self).enterEvent(event)
    
    def leaveEvent(self, event):
        """Override leaveEvent to emit custom signal"""
        self.mouseLeft.emit()
        super(QGraphicsViewMT, self).leaveEvent(event)
    

