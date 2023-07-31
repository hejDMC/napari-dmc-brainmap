from PyQt5.QtWidgets import QMainWindow, QMenu, QAction, QFileDialog
from napari_dmc_brainmap.probe_visualizer.probe_vis.probe_vis.view.MainWidget import MainWidget
from napari_dmc_brainmap.preprocessing.preprocessing_tools import adjust_contrast, do_8bit
from napari_dmc_brainmap.utils import get_bregma

import numpy as np

import cv2
import json
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QImage
from pathlib import Path
from pkg_resources import resource_filename
from bg_atlasapi import BrainGlobeAtlas

# todo merge this with atlas model

class ProbeVisualizer(QMainWindow):
    def __init__(self, app, params_dict):
        super().__init__()
        self.app = app
        self.params_dict = params_dict
        self.setWindowTitle("Probe Visualizer")
        # todo change window size matching screen size as in sharpy track
        self.setFixedSize(1900,1200) # int(self.annot.shape[0]*1.5), int(self.annot.shape[1]*1.05))
        # do not create status object, handle DV information by self
        self.createActions()
        self.createMenus()
        print("loading reference atlas...")
        self.atlas = BrainGlobeAtlas(self.params_dict['atlas_info']['atlas'])
        self.ori_idx = [self.atlas.space.axes_description.index(o) for o in
                        ['ap', 'si', 'rl']]  # get orientation indices for [ap, dv, ml] order
        self.loadTemplate()
        self.loadAnnot()
        self.loadStructureTree()
        self.calculateImageGrid()
        self.currentAP = int(self.annot.shape[0]/2)  # half AP
        self.currentDV = int(self.annot.shape[1]/2) # half depth DV
        self.currentML = int(self.annot.shape[2]/2) # midline ML
        self.viewerID = 1 # axial view by default
        shape = [self.atlas.shape[i] for i in self.ori_idx]
        resolution = [self.atlas.resolution[i]/1000 for i in self.ori_idx]
        self.widget = MainWidget(self, shape, resolution)
        self.setCentralWidget(self.widget.widget)

    def loadTemplate(self):
        print('loading template volume...')
        self.template = self.atlas.reference
        self.template = adjust_contrast(self.template, (0, self.template.max()))
        self.template = do_8bit(self.template)

    def loadAnnot(self):
        self.annot = self.atlas.annotation.transpose(self.ori_idx)  # change axis of atlas to match [ap, dv, ml] order

    def loadStructureTree(self):
        self.sTree = self.atlas.structures
        self.bregma = get_bregma(self.params_dict['atlas_info']['atlas'])
        self.bregma = [self.bregma[o] for o in self.ori_idx]

    def calculateImageGrid(self):
        dv = np.arange(self.annot.shape[1])
        ml = np.arange(self.annot.shape[2])
        grid_x,grid_y = np.meshgrid(ml,dv)
        self.r_grid_x = grid_x.ravel()
        self.r_grid_y = grid_y.ravel()
        self.grid = np.stack([grid_y,grid_x],axis=2)
    
    def getContourIndex(self):

        # todo here sth with thte idx of annot
        if self.viewerID == 1: # axial view
            self.sliceAnnot = self.annot[:,self.currentDV,:].copy().astype(np.int32).T # rotate image by 90 degrees
            empty = np.zeros((self.annot.shape[2],self.annot.shape[0]),dtype=np.uint8)
        elif self.viewerID == 0: # coronal view
            self.sliceAnnot = self.annot[self.currentAP,:,:].copy().astype(np.int32)
            empty = np.zeros((self.annot.shape[1],self.annot.shape[2]),dtype=np.uint8)
        else: # sagital view
            self.sliceAnnot = self.annot[:,:,self.currentML].copy().astype(np.int32).T # rotate image by 90 degrees
            empty = np.zeros((self.annot.shape[1],self.annot.shape[0]),dtype=np.uint8)
        # get contours
        contours,_ = cv2.findContours(self.sliceAnnot, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # draw contours on canvas
        self.outline = cv2.drawContours(empty,contours,-1,color=255) # grayscale, 8bit
        self.outline= cv2.cvtColor(self.outline, cv2.COLOR_GRAY2RGBA) # convert to RGBA
        self.outline[:,:,3][np.where(self.outline[:,:,0]==0)] = 0 # set black background transparent

    def treeFindArea(self):
        # get coordinates in mm
        # from cursor position get annotation index
        if self.viewerID == 0: # coronal
            structure_id = self.annot[self.currentAP,self.widget.labelContour.cursorPos[1],self.widget.labelContour.cursorPos[0]]
            coord_mm = self.getCoordMM([self.currentAP,self.widget.labelContour.cursorPos[1],self.widget.labelContour.cursorPos[0]])
        elif self.viewerID == 1: # axial
            structure_id = self.annot[self.widget.labelContour.cursorPos[0],self.currentDV,self.widget.labelContour.cursorPos[1]]
            coord_mm = self.getCoordMM([self.widget.labelContour.cursorPos[0],self.currentDV,self.widget.labelContour.cursorPos[1]])
        else: # sagital
            structure_id = self.annot[self.widget.labelContour.cursorPos[0],self.widget.labelContour.cursorPos[1],self.currentML]
            coord_mm = self.getCoordMM([self.widget.labelContour.cursorPos[0],self.widget.labelContour.cursorPos[1],self.currentML])
        if structure_id > 0:
            # get highlight area index
            activeArea = np.where(self.sliceAnnot == structure_id)
            # find name in sTree
            structureName = self.sTree.data[structure_id]['name']
            self.highlightArea(coord_mm,activeArea,structureName)

    def highlightArea(self,listCoordMM,activeArea,structureName):
        contourHighlight = self.outline.copy()
        contourHighlight[activeArea[0],activeArea[1],:] = [255,0,0,50] # change active area to 50% red
        # add mm coordinates, structureName
        
        if self.viewerID == 0:
            text_x1 = 1150 - 180
            text_y1 = 1050 - 340
            text_y2 = 1080 - 340
            text_y3 = 1110 - 340
            
        elif self.viewerID == 1:
            text_x1 = 1150
            text_y1 = 1050
            text_y2 = 1080
            text_y3 = 1110
        else:
            text_x1 = 1150
            text_y1 = 1050 - 340
            text_y2 = 1080 - 340
            text_y3 = 1110 - 340

        cv2.putText(contourHighlight, "AP:"+str(listCoordMM[0])+" mm", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)
        cv2.putText(contourHighlight, "ML:"+str(listCoordMM[2])+" mm", (text_x1,text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)
        cv2.putText(contourHighlight, "DV:"+str(listCoordMM[1])+" mm", (text_x1,text_y3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)
        cv2.putText(contourHighlight, structureName, (10,text_y3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0,255), 2, cv2.LINE_AA)

        contourHighlight = QImage(contourHighlight.data, contourHighlight.shape[1],contourHighlight.shape[0],contourHighlight.strides[0],QImage.Format_RGBA8888)
        self.widget.labelContour.setPixmap(QPixmap.fromImage(contourHighlight)) # update contour label

    def gSliderChanged(self):
        if self.viewerID == 1:
            self.currentDV = self.widget.gSlider.value()
        elif self.viewerID == 0:
            self.currentAP = self.widget.gSlider.value()
        else:
            self.currentML = self.widget.gSlider.value()

        self.widget.loadSlice(self)

        if self.widget.outlineBool is True:
            self.getContourIndex()
            contourQimg = QImage(self.outline.data, self.outline.shape[1],self.outline.shape[0],self.outline.strides[0],QImage.Format_RGBA8888)
            self.widget.labelContour.setPixmap(QPixmap.fromImage(contourQimg))
        else:
            pass
    
    def switchViewer(self,new_viewerID):
        self.viewerID = new_viewerID
        windowWidth = self.size().width()

        if self.viewerID == 0: # switch QLabel size
            self.move(self.pos().x()+(windowWidth-int(self.annot.shape[2] * 1.5)),self.pos().y()) # pin to top-right of window
            self.setFixedSize(int(self.annot.shape[2] * 1.5),int(self.annot.shape[1]*1.05))
            self.widget.viewer.setFixedSize(self.annot.shape[2],self.annot.shape[1])
            self.widget.labelContour.setFixedSize(self.annot.shape[2],self.annot.shape[1])
        elif self.viewerID == 1:
            self.move(self.pos().x()+(windowWidth-int(self.annot.shape[0]*1.5)),self.pos().y())
            self.setFixedSize(int(self.annot.shape[0]*1.5), int(self.annot.shape[2]*1.05))
            self.widget.viewer.setFixedSize(self.annot.shape[0],self.annot.shape[2])
            self.widget.labelContour.setFixedSize(self.annot.shape[0],self.annot.shape[2])
        else:
            self.move(self.pos().x()+(windowWidth-int(self.annot.shape[0]*1.5)),self.pos().y())
            self.setFixedSize(int(self.annot.shape[0]*1.5), int(self.annot.shape[1]*1.05))
            self.widget.viewer.setFixedSize(self.annot.shape[0],self.annot.shape[1])
            self.widget.labelContour.setFixedSize(self.annot.shape[0],self.annot.shape[1])

        self.widget.updateSlider(self)
        self.widget.loadSlice(self)


    
    def organizeProbe(self):
        # get probes into numpy array
        self.probe_list = [] # get voxel coordinates
        self.probe_axis = [] # get primary probe axis
        for p in self.probeDict.keys():
            self.probe_list.append(np.array(self.probeDict[p]['Voxel'],dtype=np.uint16))
            if self.probeDict[p]['axis'] == "AP":
                self.probe_axis.append(0)
            elif self.probeDict[p]['axis'] == "DV":
                self.probe_axis.append(1)
            else:
                self.probe_axis.append(2)
        self.widget.loadSlice(self)
            
    
    def keyPressEvent(self, event):   
        if event.key() == Qt.Key_A: # A for showing brain region outline
            if self.widget.outlineBool is False:
                self.widget.outlineBool = True
                self.getContourIndex()
                contourQimg = QImage(self.outline.data, self.outline.shape[1],self.outline.shape[0],self.outline.strides[0],QImage.Format_RGBA8888)
                self.widget.labelContour.setPixmap(QPixmap.fromImage(contourQimg))
                 # show contour
                self.widget.labelContour.setVisible(True)
            else:
                self.widget.outlineBool = False
                self.widget.labelContour.setVisible(False)
    
    def getCoordMM(self,vox_index):
        vox_ap,vox_dv,vox_ml = vox_index
        ap_mm = np.round((self.bregma[0] - vox_ap) *
                         (self.atlas.resolution[self.atlas.space.axes_description.index('ap')]/1000),2)
        dv_mm = np.round((self.bregma[1] - vox_dv) *
                         (self.atlas.resolution[self.atlas.space.axes_description.index('si')] / 1000), 2)
        ml_mm = np.round((self.bregma[2] - vox_ml) *
                         (self.atlas.resolution[self.atlas.space.axes_description.index('rl')] / 1000), 2)
        return [ap_mm,dv_mm,ml_mm]

    def createMenus(self):
        self.fileMenu = QMenu("&File",self)
        self.fileMenu.addAction(self.loadAct)

        self.menuBar().addMenu(self.fileMenu)
    
    def createActions(self):
        self.loadAct = QAction("L&oad Probe JSON",self,shortcut='Ctrl+L',triggered=self.loadJSON)
    
    def loadJSON(self):
        self.probeJsonPath = QFileDialog.getOpenFileName(self,"Choose Probe JSON","","JSON File (*.json)")[0]
        if self.probeJsonPath == "":
            pass
        else:
            with open (self.probeJsonPath,'r') as probeData:
                self.probeDict = json.load(probeData)
            self.organizeProbe()
            self.widget.createProbeSelector(self)
            self.widget.loadSlice(self)


