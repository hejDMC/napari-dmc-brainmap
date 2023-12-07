from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsEllipseItem

class DotObject(QGraphicsEllipseItem):
    def __init__(self, x, y, r):
        super().__init__(0, 0, r, r)
        # self.app = app
        self.setPos(x-int(r/2), y-int(r/2))
        self.setBrush(Qt.blue)
        self.setAcceptHoverEvents(True)
    
    def linkPairedDot(self,pairDot):
        self.pairDot = pairDot

    # mouse hover event
    def hoverEnterEvent(self, event):
        # self.app.instance().setOverrideCursor(Qt.CrossCursor)
        self.setBrush(Qt.red)
        self.pairDot.setBrush(Qt.red)

    def hoverLeaveEvent(self, event):
        # self.app.instance().restoreOverrideCursor()
        self.setBrush(Qt.blue)
        self.pairDot.setBrush(Qt.blue)

    # mouse click event
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            print('Right Click outside the dot(s) to remove the most recently added dot')

    def mouseMoveEvent(self, event):
        orig_cursor_position = event.lastScenePos()
        updated_cursor_position = event.scenePos()
        orig_position = self.scenePos()
        updated_cursor_x = updated_cursor_position.x() - orig_cursor_position.x() + orig_position.x()
        updated_cursor_y = updated_cursor_position.y() - orig_cursor_position.y() + orig_position.y()
        self.setPos(QPointF(updated_cursor_x, updated_cursor_y))
