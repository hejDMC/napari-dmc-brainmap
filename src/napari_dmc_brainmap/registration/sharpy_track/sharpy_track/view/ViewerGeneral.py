from PyQt5.QtWidgets import QLabel,QGraphicsScene,QFrame
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.QGraphicsViewerMT import QGraphicsViewMT

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

    def leaveLabel(self,event):
        self.regViewer.status.cursor = 0
    
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



        