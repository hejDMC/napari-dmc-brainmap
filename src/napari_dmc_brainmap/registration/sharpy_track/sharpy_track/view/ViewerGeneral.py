from PyQt5.QtWidgets import QLabel,QGraphicsScene,QFrame
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.QGraphicsViewerMT import QGraphicsViewMT

class ViewerGeneral():
    def __init__(self,regViewer) -> None:
        self.labelImg = QLabel()
        self.labelImg.setFixedSize(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1])
        self.scene = QGraphicsScene(0,0,regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1],parent=regViewer)
        self.scene.changed.connect(lambda: regViewer.atlasModel.updateDotPosition(regViewer)) # here update
        self.scene.addWidget(self.labelImg)
        self.itemGroup = [] # create itemGroup, store DotObjects
        self.view = QGraphicsViewMT(self.scene) # QGraphicsView with mousetracking
        self.view.leaveEvent = lambda event: self.leaveLabel(regViewer)
        self.view.setFixedSize(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1])
        self.view.setSceneRect(0,0,regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1])
        self.view.setFrameShape(QFrame.NoFrame)

    def leaveLabel(self,regViewer):
        regViewer.status.cursor = 0
    
    def getCursorPos(self,regViewer):
        if regViewer.status.contour == 1: # only when contour active, update in status
            regViewer.status.hoverX = int(self.view.cursorPos[0] / regViewer.status.scaleFactor) # save hover position to status, if single window size different from 1140*800 scale coordinates
            regViewer.status.hoverY = int(self.view.cursorPos[1] / regViewer.status.scaleFactor)
            regViewer.atlasModel.treeFindArea(regViewer)
        else:
            pass



        