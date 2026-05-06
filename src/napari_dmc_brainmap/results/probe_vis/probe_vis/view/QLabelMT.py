from qtpy.QtCore import Signal
from qtpy.QtWidgets import QLabel

class QLabelMT(QLabel):
    mouseMoved = Signal()
    """
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)


    def mouseMoveEvent(self,event):
        self.cursorPos = [event.pos().x(),event.pos().y()]
        self.mouseMoved.emit()
        super(QLabelMT, self).mouseMoveEvent(event)
        
    

