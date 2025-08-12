from PyQt5.QtWidgets import QLabel,QGraphicsScene,QFrame, QGraphicsItemGroup, QGraphicsPixmapItem
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.QGraphicsViewerMT import QGraphicsViewMT
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import mapPointTransform
import numpy as np

class ViewerGeneral():
    def __init__(self,regViewer) -> None:
        self.regViewer = regViewer
        self.labelImg = QLabel()
        self.labelImg.setFixedSize(regViewer.singleWindowSize[0],regViewer.singleWindowSize[1])
        self.scene = QGraphicsScene(0,0,regViewer.singleWindowSize[0],regViewer.singleWindowSize[1],parent=regViewer)
        self.scene.addWidget(self.labelImg)
        self.itemGroup = [] # create itemGroup, store DotObjects
        self.view = QGraphicsViewMT(self.scene) # QGraphicsView with mousetracking
        self.view.setFixedSize(regViewer.singleWindowSize[0],regViewer.singleWindowSize[1])
        self.view.setSceneRect(0,0,regViewer.singleWindowSize[0],regViewer.singleWindowSize[1])
        self.view.setFrameShape(QFrame.NoFrame)
        # TRE variables
        # project source position to target position using current transformation matrix
        self.tform = None
        self.targetPointHover = QGraphicsItemGroup()  # container for dynamically projected point

    def leaveLabel(self):
        """Slot for mouse leave signal"""
        self.regViewer.status.cursor = 0
    
    # to be connected with mouseMoved signal on the left viewer
    def getCursorPos(self):
        if self.regViewer.status.contour == 1: # only when contour active, update in status
            # cursor position within boundary check
            if (self.view.cursorPos[0] in self.regViewer.res_x_range) and (
                self.view.cursorPos[1] in self.regViewer.res_y_range):
                self.regViewer.status.hoverX = self.regViewer.res_up[self.view.cursorPos[0]] # save hover position to status, if single window size different from 1140*800 scale coordinates
                self.regViewer.status.hoverY = self.regViewer.res_up[self.view.cursorPos[1]]
                self.regViewer.atlasModel.treeFindArea()
                
            else:
                pass
        else:
            pass
    
    # to be connected with mouseMoved signal on the right viewer
    def projectSourcePos(self):
        x_target, y_target = mapPointTransform(self.view.cursorPos[0], self.view.cursorPos[1], self.tform)
        # round
        x_target, y_target = np.round(x_target), np.round(y_target)
        # within boundary check
        if any([
                x_target < 0,
                x_target >= self.regViewer.singleWindowSize[0],
                y_target < 0,
                y_target >= self.regViewer.singleWindowSize[1]
                ]):
            # target point out of boundary, indicate with red cursor on the right viewer
            self.regViewer.widget.viewerRight.view.setCursor(self.regViewer.measurementPage.cursor_r_64)
            # clear target point marker
            if self.targetPointHover.childItems():
                self.clearTargetPointHover()
            # PREVENT THE CREATION OF SOURCE DOT
        else:
            # switch cursor to yellow pointer
            self.regViewer.widget.viewerRight.view.setCursor(self.regViewer.measurementPage.cursor_y_64)
            # check if there is already a pixmap item in the list
            if self.targetPointHover.childItems():
                # update the position of the existing item
                self.targetPointHover.childItems()[0].setPos(x_target - 16, y_target - 16)  # update position of existing items
            else:
                # overlay small yellow pointer(32pix by 32pix) with [16,16] as center
                item = QGraphicsPixmapItem(self.regViewer.measurementPage.pixmap_y_32)
                self.targetPointHover.addToGroup(item)
                self.targetPointHover.childItems()[0].setPos(x_target - 16, y_target - 16)


    def clearTargetPointHover(self):
        self.regViewer.widget.viewerLeft.scene.removeItem(self.targetPointHover.childItems()[0])

