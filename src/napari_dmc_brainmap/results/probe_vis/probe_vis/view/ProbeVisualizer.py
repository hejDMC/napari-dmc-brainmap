from PyQt5.QtWidgets import QMainWindow, QMenu, QAction, QFileDialog, QApplication
from napari_dmc_brainmap.results.probe_vis.probe_vis.view.MainWidget import MainWidget
# from napari_dmc_brainmap.preprocessing.preprocessing_tools import adjust_contrast, do_8bit
from napari_dmc_brainmap.utils.atlas_utils import get_bregma

import numpy as np
import cv2
import json
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QImage
from pathlib import Path

from napari.utils.notifications import show_info
from bg_atlasapi import BrainGlobeAtlas
import os


# todo merge this with atlas model

class ProbeVisualizer(QMainWindow):
    def __init__(self, app, params_dict):
        super().__init__()
        self.app = app
        self.params_dict = params_dict
        self.calculate_z_decimal()
        self.setWindowTitle("Probe Visualizer")
        # get primary screen size
        QAppInstance = QApplication.instance()
        self.screenSize = [QAppInstance.primaryScreen().size().width(),QAppInstance.primaryScreen().size().height()]

        # do not create status object, handle DV information by self
        self.createActions()
        self.createMenus()
        show_info("loading reference atlas...")
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
        self.applySizePolicy(shape)
        self.widget = MainWidget(self, shape, resolution)
        self.setCentralWidget(self.widget.widget)

    
    def calculate_z_decimal(self):
        step_float = self.params_dict['atlas_info']['xyz_dict']['z'][2] / 1000
        # by default keep 2 decimals
        decimal = 2
        if np.round(step_float,decimal) == 0: # extend decimal
            while np.abs(np.round(step_float,decimal)-step_float) >= 0.01 * step_float:
                decimal += 1
        else:
            pass
        self.decimal = decimal

        
    def loadTemplate(self):
        brainglobe_dir = Path.home() / ".brainglobe"
        atlas_name_general  = f"{self.params_dict['atlas_info']['atlas']}_v*"
        atlas_names_local = list(brainglobe_dir.glob(atlas_name_general))[0] # glob returns generator object, need to exhaust it in list, then take out

        # for any atlas else, in this case test with zebrafish atlas
        show_info('checking template volume...')
        if os.path.isfile(os.path.join(brainglobe_dir,atlas_names_local,'reference_8bit.npy')): # when directory has 8-bit template volume, load it
            show_info('loading template volume...')
            self.template = np.load(os.path.join(brainglobe_dir,atlas_names_local,'reference_8bit.npy'))

        else: # when saved template not found
            # check if template volume from brainglobe is already 8-bit
            self.template = self.atlas.reference
            if np.issubdtype(self.template.dtype,np.uint16): # check if template is 16-bit
                show_info('creating 8-bit template volume...')
                # rescale intensity
                lim_16_min = self.template.min()
                lim_16_max = self.template.max()
                self.template = self.template - lim_16_min # adjust brightness and downsample to 8-bit
                self.template = self.template / (lim_16_max-lim_16_min) * 255
                self.template = self.template.astype(np.uint8)
                # save to 8-bit npy file
                np.save(os.path.join(brainglobe_dir,atlas_names_local,'reference_8bit.npy'), self.template) # save volume for next time loading
            
            elif np.issubdtype(self.template.dtype,np.uint8): # if 8-bit, no need for downsample
                pass
            else: # other nparray.dtype
                show_info("Data type for reference volume: {}".format(self.template.dtype))
                show_info("at : {}".format(os.path.join(brainglobe_dir,atlas_names_local,'reference.tiff')))
                show_info("8-bit / 16-bit grayscale volume is required.")
                show_info("Reference volume cannot be correctly loaded to ProbeVisualizer!")


        

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
        # scale outline RGBA layer according to scale factor
        self.outline = cv2.resize(self.outline,(int(empty.shape[1]*self.scaleFactor),int(empty.shape[0]*self.scaleFactor)))

    def treeFindArea(self):
        # get coordinates in mm
        # from cursor position get annotation index
        
        # scale cursorPos
        self.widget.labelContour.cursorPos = [int(self.widget.labelContour.cursorPos[0]/self.scaleFactor),int(self.widget.labelContour.cursorPos[1]/self.scaleFactor)]
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
        # scale activeArea
        contourHighlight[(activeArea[0]*self.scaleFactor).astype(int),(activeArea[1]*self.scaleFactor).astype(int),:] = [255,0,0,50] # change active area to 50% red
        # add mm coordinates, structureName
        text_w, text_h = cv2.getTextSize("AP:"+str(listCoordMM[0])+" mm", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = contourHighlight.shape[1] - text_w - 10
        text_y = contourHighlight.shape[0] - 2*text_h - 20
        cv2.putText(contourHighlight, "AP:"+str(listCoordMM[0])+" mm", (text_x,text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)

        text_w, text_h = cv2.getTextSize("ML:"+str(listCoordMM[2])+" mm", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = contourHighlight.shape[1] - text_w - 10
        text_y = contourHighlight.shape[0] - text_h - 15
        cv2.putText(contourHighlight, "ML:"+str(listCoordMM[2])+" mm", (text_x,text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)

        text_w, text_h = cv2.getTextSize("DV:"+str(listCoordMM[1])+" mm", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = contourHighlight.shape[1] - text_w - 10
        text_y = contourHighlight.shape[0] - 10
        cv2.putText(contourHighlight, "DV:"+str(listCoordMM[1])+" mm", (text_x,text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)

        cv2.putText(contourHighlight, structureName, (10,text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0,255), 2, cv2.LINE_AA)

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
    
    def switchViewer(self,new_viewerID): # adapt this function when clicking on view radio button
        self.viewerID = new_viewerID
        windowWidth = self.size().width()

        if self.viewerID == 0: # switch QLabel size
            self.move(self.pos().x()+(windowWidth-int(self.annot.shape[0]*1.5*self.scaleFactor)),self.pos().y()) # pin to top-right of window
            self.setFixedSize(int(self.annot.shape[0]*1.5*self.scaleFactor),int(self.annot.shape[1]*1.1*self.scaleFactor))
            self.widget.viewer.setFixedSize(int(self.annot.shape[2]*self.scaleFactor),int(self.annot.shape[1]*self.scaleFactor))
            self.widget.labelContour.setFixedSize(int(self.annot.shape[2]*self.scaleFactor),int(self.annot.shape[1]*self.scaleFactor))
        elif self.viewerID == 1:
            self.move(self.pos().x()+(windowWidth-int(self.annot.shape[0]*1.5 * self.scaleFactor)),self.pos().y())
            self.setFixedSize(int(self.annot.shape[0]*1.5*self.scaleFactor), int(self.annot.shape[2]*1.1*self.scaleFactor))
            self.widget.viewer.setFixedSize(int(self.annot.shape[0]*self.scaleFactor),int(self.annot.shape[2]*self.scaleFactor))
            self.widget.labelContour.setFixedSize(int(self.annot.shape[0]*self.scaleFactor),int(self.annot.shape[2]*self.scaleFactor))
        else:
            self.move(self.pos().x()+(windowWidth-int(self.annot.shape[0]*1.5 * self.scaleFactor)),self.pos().y())
            self.setFixedSize(int(self.annot.shape[0]*1.5*self.scaleFactor), int(self.annot.shape[1]*1.1*self.scaleFactor))
            self.widget.viewer.setFixedSize(int(self.annot.shape[0]*self.scaleFactor),int(self.annot.shape[1]*self.scaleFactor))
            self.widget.labelContour.setFixedSize(int(self.annot.shape[0]*self.scaleFactor),int(self.annot.shape[1]*self.scaleFactor))

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
                         (self.atlas.resolution[self.atlas.space.axes_description.index('ap')]/1000),self.decimal)
        dv_mm = np.round((self.bregma[1] - vox_dv) *
                         (self.atlas.resolution[self.atlas.space.axes_description.index('si')] / 1000), self.decimal)
        ml_mm = np.round((self.bregma[2] - vox_ml) *
                         (self.atlas.resolution[self.atlas.space.axes_description.index('rl')] / 1000), self.decimal)
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
    

    def applySizePolicy(self,template_shape):
        if np.min(self.screenSize) < np.max(template_shape): # if the biggest axis of volume exceeds the smallest length of screen, downscale
            self.scaleFactor = np.round((np.min(self.screenSize) / np.max(template_shape)) * 0.8, 2)
        else:
            self.scaleFactor = 1 

        self.setFixedSize(int(self.annot.shape[0]*1.5 * self.scaleFactor) # viewerID=1 at initiation
                            ,int(self.annot.shape[2]*1.1 * self.scaleFactor))

        



        # if self.screenSize[0] > round(atlas_resolution[0]*2.2) and self.screenSize[1] > round(atlas_resolution[1] * 1.1):  # 2x width (plus margin) and 1x height (plus margin)
        #     self.scaleFactor = 1
        #     self.fullWindowSizeNarrow = [2350,940]  # todo delete this
        #     self.fullWindowSizeWide = [int(round(atlas_resolution[0]*2.2)), int(round(atlas_resolution[1]*1.1))]
        #     self.singleWindowSize = atlas_resolution

        # else: # [1920,1080] resolution
        #     self.scaleFactor = round(self.screenSize[0]/(atlas_resolution[0] * 2.5), 2)
        #     self.fullWindowSizeNarrow = [1762,705]
        #     self.fullWindowSizeWide = [int(round((atlas_resolution[0]*2.2) * self.scaleFactor)),
        #                                int(round((atlas_resolution[1]*1.25) * self.scaleFactor))]
        #     self.singleWindowSize = [int(i*self.scaleFactor) for i in atlas_resolution]


