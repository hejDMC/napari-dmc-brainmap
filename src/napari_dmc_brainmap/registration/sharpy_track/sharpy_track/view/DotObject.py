from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsEllipseItem

class DotObject(QGraphicsEllipseItem):
    def __init__(self, x, y, r):
        super().__init__(0, 0, r, r)
        self.size = r
        # self.app = app
        self.setPos(x-int(r/2), y-int(r/2))
        self.setBrush(Qt.blue)
        self.setAcceptHoverEvents(True)
    
    def linkPairedDot(self,pairDot):
        self.pairDot = pairDot

    def get_xybound(self):
        # get xy moving boundary
        dim_x,dim_y = self.scene().width(), self.scene().height()
        self.bound_x,self.bound_y = int(dim_x)-1, int(dim_y)-1

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
        # DotObject stays inside viewer
            # check if bound exist
        if hasattr(self,'bound_x'):
            pass
        else:
            self.get_xybound()

        if updated_cursor_x+int(self.size/2) < 0:
            updated_cursor_x = -int(self.size/2)
        elif updated_cursor_x+int(self.size/2) > self.bound_x:
            updated_cursor_x = self.bound_x - int(self.size/2)
        else:
            pass

        if updated_cursor_y+int(self.size/2) < 0:
            updated_cursor_y = -int(self.size/2)
        elif updated_cursor_y+int(self.size/2) > self.bound_y:
            updated_cursor_y = self.bound_y - int(self.size/2)
        else:
            pass
        # update dot position
        self.setPos(QPointF(updated_cursor_x, updated_cursor_y))
