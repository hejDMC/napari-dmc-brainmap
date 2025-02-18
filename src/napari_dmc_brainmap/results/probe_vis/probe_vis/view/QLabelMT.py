from PyQt5.QtWidgets import QLabel
from PyQt5 import QtCore

class QLabelMT(QLabel):
    mouseMoved = QtCore.pyqtSignal()
    """
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)


    def mouseMoveEvent(self,event):
        self.cursorPos = [event.pos().x(),event.pos().y()]
        self.mouseMoved.emit()
        super(QLabelMT, self).mouseMoveEvent(event)
        
    

