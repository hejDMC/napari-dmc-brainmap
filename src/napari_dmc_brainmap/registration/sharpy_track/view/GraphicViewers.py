from PyQt5.QtGui import QPixmap,QImage
from sharpy_track.view.ViewerGeneral import ViewerGeneral
from PyQt5.QtWidgets import QLabel

class ViewerLeft(ViewerGeneral):
    def __init__(self,regViewer) -> None:
        super().__init__(regViewer)
        self.labelContour = QLabel()
        self.labelContour.setVisible(False)
        self.labelContour.setStyleSheet("background:transparent")
        self.scene.addWidget(self.labelContour)
        self.view.enterEvent = lambda event: self.hoverLeft(regViewer)
        

    def hoverLeft(self,regViewer):
        regViewer.status.cursor = -1 # when mouse cursor is on of left viewer
    
    def loadSlice(self,regViewer):
        regViewer.atlasModel.getSlice(regViewer)
        self.labelImg.setPixmap(QPixmap.fromImage(regViewer.atlasModel.sliceQimg))
        if regViewer.status.contour == 1: # if show contour active
            regViewer.atlasModel.displayContour(regViewer) # display contour
        else:
            pass
    
    def showContourLabel(self,regViewer):
        # render transparent contour QImage
        contourImg = regViewer.atlasModel.outline
        contourQimg = QImage(contourImg.data, contourImg.shape[1],contourImg.shape[0],contourImg.strides[0],QImage.Format_RGBA8888)
        self.labelContour.setPixmap(QPixmap.fromImage(contourQimg))
        # show contour
        self.labelContour.setVisible(True)
    
    def hideContourLabel(self):
        self.labelContour.setVisible(False)





class ViewerRight(ViewerGeneral):
    def __init__(self,regViewer) -> None:
        super().__init__(regViewer)
        self.view.enterEvent = lambda event: self.hoverRight(regViewer)
    
    def hoverRight(self,regViewer):
        regViewer.status.cursor = 1 # when mouse cursor is on right viewer
    
    def loadSample(self,regViewer):
        regViewer.atlasModel.getSample(regViewer)
        self.labelImg.setPixmap(QPixmap.fromImage(regViewer.atlasModel.sampleQimg))








