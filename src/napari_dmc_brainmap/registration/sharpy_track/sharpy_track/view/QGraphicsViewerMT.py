from PyQt5.QtWidgets import QGraphicsView
from PyQt5 import QtCore

class QGraphicsViewMT(QGraphicsView):
    mouseMoved = QtCore.pyqtSignal()
    """
    """
    def __init__(self,scene) -> None:
        super().__init__(scene)


    def mouseMoveEvent(self,event):
        self.cursorPos = [event.pos().x(),event.pos().y()]
        self.mouseMoved.emit()
        super(QGraphicsViewMT, self).mouseMoveEvent(event)
    

