from PyQt5.QtGui import QPixmap,QImage
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.ViewerGeneral import ViewerGeneral
from PyQt5.QtWidgets import QLabel
import cv2

class ViewerLeft(ViewerGeneral):
    def __init__(self,regViewer) -> None:
        super().__init__(regViewer)
        self.labelContour = QLabel()
        self.labelContour.setFixedSize(regViewer.singleWindowSize[0], regViewer.singleWindowSize[1])
        self.labelContour.setVisible(False)
        self.labelContour.setStyleSheet("background:transparent")
        self.scene.addWidget(self.labelContour)
        

    def hoverLeft(self):
        """Slot for mouse enter signal"""
        self.regViewer.status.cursor = -1 # when mouse cursor is on of left viewer
    
    def loadSlice(self):
        self.regViewer.atlasModel.getSlice()
        self.labelImg.setPixmap(QPixmap.fromImage(self.regViewer.atlasModel.sliceQimg))
        if self.regViewer.status.contour == 1: # if show contour active
            self.regViewer.atlasModel.displayContour() # display contour
        else:
            pass
    
    def showContourLabel(self):
        # render transparent contour QImage
        contourImg = self.regViewer.atlasModel.outline
        contourImg = cv2.resize(contourImg,(self.regViewer.singleWindowSize[0],self.regViewer.singleWindowSize[1]))
        contourQimg = QImage(contourImg.data, contourImg.shape[1],contourImg.shape[0],contourImg.strides[0],QImage.Format_RGBA8888)
        self.labelContour.setPixmap(QPixmap.fromImage(contourQimg))
        # show contour
        self.labelContour.setVisible(True)
    
    def hideContourLabel(self):
        self.labelContour.setVisible(False)
    
    def highlightArea(self,listCoordMM,activeArea,structureName):
        contourHighlight = self.regViewer.atlasModel.outline.copy()
        contourHighlight[activeArea[0],activeArea[1],:] = [255,0,0,50] # change active area to 50% red
        # add mm coordinates, structureName
        offset = int(self.regViewer.atlasModel.fontscale * 10)

        text_w, text_h = cv2.getTextSize("AP:"+str(listCoordMM[0])+" mm", cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.regViewer.atlasModel.fontscale, self.regViewer.atlasModel.fontthickness)[0]
        ap_text_location = [contourHighlight.shape[1]-offset-text_w,contourHighlight.shape[0]-offset-text_h-offset-text_h-offset]

        text_w, text_h = cv2.getTextSize("ML:"+str(listCoordMM[2])+" mm", cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.regViewer.atlasModel.fontscale, self.regViewer.atlasModel.fontthickness)[0]
        ml_text_location = [contourHighlight.shape[1]-offset-text_w,contourHighlight.shape[0]-offset-text_h-offset]

        text_w, text_h = cv2.getTextSize("DV:"+str(listCoordMM[1])+" mm", cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.regViewer.atlasModel.fontscale, self.regViewer.atlasModel.fontthickness)[0]
        dv_text_location = [contourHighlight.shape[1]-offset-text_w,contourHighlight.shape[0]-offset]

        cv2.putText(contourHighlight, "AP:"+str(listCoordMM[0])+" mm", (ap_text_location[0],ap_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.regViewer.atlasModel.fontscale, (255,0,0,255), self.regViewer.atlasModel.fontthickness, cv2.LINE_AA)
        cv2.putText(contourHighlight, "ML:"+str(listCoordMM[2])+" mm", (ml_text_location[0],ml_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.regViewer.atlasModel.fontscale, (255,0,0,255), self.regViewer.atlasModel.fontthickness, cv2.LINE_AA)
        cv2.putText(contourHighlight, "DV:"+str(listCoordMM[1])+" mm", (dv_text_location[0],dv_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.regViewer.atlasModel.fontscale, (255,0,0,255), self.regViewer.atlasModel.fontthickness, cv2.LINE_AA)

        structure_text_location = [offset,contourHighlight.shape[0]-offset]
        cv2.putText(contourHighlight, structureName, (structure_text_location[0],structure_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8*self.regViewer.atlasModel.fontscale, (255,0,0,255), self.regViewer.atlasModel.fontthickness, cv2.LINE_AA)
        
        contourHighlight = cv2.resize(contourHighlight,(self.regViewer.singleWindowSize[0],self.regViewer.singleWindowSize[1]))
        contourHighlight = QImage(contourHighlight.data, contourHighlight.shape[1],contourHighlight.shape[0],contourHighlight.strides[0],QImage.Format_RGBA8888)
        self.labelContour.setPixmap(QPixmap.fromImage(contourHighlight)) # update contour label








class ViewerRight(ViewerGeneral):
    def __init__(self,regViewer) -> None:
        super().__init__(regViewer)
    
    def hoverRight(self):
        """Slot for mouse enter signal"""
        self.regViewer.status.cursor = 1 # when mouse cursor is on right viewer
    
    def loadSample(self):
        self.regViewer.atlasModel.getSample()
        self.labelImg.setPixmap(QPixmap.fromImage(self.regViewer.atlasModel.sampleQimg))








