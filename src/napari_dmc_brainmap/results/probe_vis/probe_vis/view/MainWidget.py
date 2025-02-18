import cv2
import numpy as np
import functools
import distinctipy as dc

from PyQt5.QtWidgets import QSlider,QWidget,QGridLayout,QHBoxLayout,QVBoxLayout,QLabel,QCheckBox,QRadioButton,QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QImage
from napari.utils.notifications import show_info

from napari_dmc_brainmap.results.probe_vis.probe_vis.view.QLabelMT import QLabelMT

class MainWidget():
    def __init__(self,probeV, shape, resolution):
        self.widget = QWidget()
        self.shape = shape
        self.resolution = resolution
        self.mainLayout = QHBoxLayout()
        self.widget.setLayout(self.mainLayout)
        
        self.labelBox = QGridLayout()
        self.labelBox.setAlignment(Qt.AlignTop)

        self.viewer = QLabel()
        self.viewer.setFixedSize(int(self.shape[0]*probeV.scaleFactor),int(self.shape[2]*probeV.scaleFactor))
        self.labelBox.addWidget(self.viewer,0,0,Qt.AlignLeft,Qt.AlignTop)

        self.labelContour = QLabelMT()
        self.labelContour.setFixedSize(int(self.shape[0]*probeV.scaleFactor),int(self.shape[2]*probeV.scaleFactor))
        self.labelBox.addWidget(self.labelContour,0,0,Qt.AlignLeft,Qt.AlignTop)

        self.mainLayout.addLayout(self.labelBox)
        
        self.createSliderGeneral(probeV)
        self.createCtlPanel(probeV)
        # create overlaying labelContour
        self.outlineBool = False
        self.labelContour.mouseMoved.connect(lambda: probeV.treeFindArea())
        self.labelContour.setVisible(False) # hide contour by default
        self.labelContour.setStyleSheet("background:transparent")

        self.loadSlice(probeV)
    
    def loadSlice(self,probeV):
        # get text height
        _, text_h = cv2.getTextSize("AP", cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
        if probeV.viewerID == 0: # coronal
            self.slice = probeV.template[probeV.currentAP, :, :].copy()
            cv2.putText(self.slice, "AP: " + str(probeV.currentAP), (10, text_h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA) # voxel coordinate
            cv2.putText(self.slice, "(" + str(np.round((probeV.bregma[0] - probeV.currentAP) * self.resolution[0], probeV.decimal)) + " mm)", 
                        (10, text_h*2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA) # mm coordinate

        elif probeV.viewerID == 1: # axial
            # rotate AP axis to screen width axis
            self.slice = probeV.template[:, probeV.currentDV, :].T.copy()
            cv2.putText(self.slice, "DV: "+str(probeV.currentDV), (10,text_h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
            cv2.putText(self.slice, "("+str(np.round((probeV.bregma[1] - probeV.currentDV)*self.resolution[1],probeV.decimal))+" mm)" , 
                        (10,text_h*2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA) # mm coordinate
        else: # sagittal
            # rotate AP axis to screen width axis
            self.slice = probeV.template[:, :, probeV.currentML].T.copy()
            cv2.putText(self.slice, "ML: "+str(probeV.currentML), (10,text_h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
            cv2.putText(self.slice, "("+str(np.round((probeV.bregma[2] -probeV.currentML) * self.resolution[2],probeV.decimal))+" mm)" , 
                        (10,text_h*2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA) # mm coordinate
        
        # add active probe to slice
        if hasattr(self,'probeDisplayList'):
            self.slice = cv2.cvtColor(self.slice, cv2.COLOR_GRAY2BGR) # prepare BGR image for colorful probes
            if np.sum(self.probeDisplayList)>0:  # has active probe(s)
                for p in np.where(np.array(self.probeDisplayList)==1)[0]:
                    vox_this_probe = probeV.probe_list[p]
                    if probeV.viewerID == 1: # axial plane
                        if probeV.currentDV in vox_this_probe[:,1]: # this probe goes through current axial plane
                            probe_coord_2d = vox_this_probe[np.where(vox_this_probe[:,1]==probeV.currentDV)[0]][:,[0,2]] # Probably needs to transpose
                            # draw dot
                            for coord in probe_coord_2d:
                                cv2.circle(self.slice, (coord[0],coord[1]), 3, self.probeColor[p], -1)
                        else:
                            pass # probe not going through this axial plane

                    elif probeV.viewerID == 0: # coronal plane
                        if probeV.currentAP in vox_this_probe[:, 0]:
                            probe_coord_2d = vox_this_probe[np.where(vox_this_probe[:,0] == probeV.currentAP)[0]][:, [1, 2]]
                            # draw dot
                            for coord in probe_coord_2d:
                                cv2.circle(self.slice, (coord[1],coord[0]), 3, self.probeColor[p], -1) # transpost XY here
                        else:
                            pass

                    else: # sagital plane
                        if probeV.currentML in vox_this_probe[:,2]:
                            probe_coord_2d = vox_this_probe[np.where(vox_this_probe[:,2]==probeV.currentML)[0]][:,[0,1]]
                            # draw dot
                            for coord in probe_coord_2d:
                                cv2.circle(self.slice, (coord[0],coord[1]), 3, self.probeColor[p], -1)
                        else:
                            pass

            else:
                pass # no active probe
            
            # scaling image according to scaleFactor
            self.slice = cv2.resize(self.slice,(int(self.slice.shape[1]*probeV.scaleFactor),int(self.slice.shape[0]*probeV.scaleFactor)))
            self.sliceQ = QImage(self.slice.data, self.slice.shape[1],self.slice.shape[0],self.slice.strides[0],QImage.Format_BGR888)
        else:
            self.slice = cv2.resize(self.slice,(int(self.slice.shape[1]*probeV.scaleFactor),int(self.slice.shape[0]*probeV.scaleFactor)))
            self.sliceQ = QImage(self.slice.data, self.slice.shape[1],self.slice.shape[0],self.slice.strides[0],QImage.Format_Grayscale8)
        self.viewer.setPixmap(QPixmap.fromImage(self.sliceQ))

    
    def createSliderGeneral(self,probeV):
        self.gSlider = QSlider(Qt.Vertical)
        self.gSlider.setMinimum(0)
        self.gSlider.setMaximum(self.shape[1]-1) # change maximum value according to viewerID
        self.gSlider.setSingleStep(1)
        self.gSlider.setInvertedAppearance(True)
        self.gSlider.setSliderPosition(probeV.currentDV) # set slider position to current AP/DV/ML
        self.gSlider.valueChanged.connect(lambda: probeV.gSliderChanged())
        self.mainLayout.addWidget(self.gSlider)
    
    def updateSlider(self,probeV):
        if probeV.viewerID == 0:
            currentAP = probeV.currentAP # save current_z copy
            self.gSlider.setMaximum(self.shape[0]-1) # current_z reset
            self.gSlider.setSliderPosition(currentAP) # restore current_z
        elif probeV.viewerID == 1:
            currentDV = probeV.currentDV
            self.gSlider.setMaximum(self.shape[1]-1)
            self.gSlider.setSliderPosition(currentDV)
        else:
            currentML = probeV.currentML
            self.gSlider.setMaximum(self.shape[2]-1)
            self.gSlider.setSliderPosition(currentML)


    def createCtlPanel(self,probeV):
        # create control panel layout
        self.ctlPanel = QVBoxLayout()
        # add VBox to layoutgrid
        self.mainLayout.addLayout(self.ctlPanel)
        # create title
        self.ctlTitle = QLabel()
        self.ctlTitle.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding))
        self.ctlTitle.setAlignment(Qt.AlignTop)
        self.ctlTitle.setFixedHeight(50)
        # set text and font
        self.ctlTitle.setText("Control Panel")
        font = self.ctlTitle.font()
        font.setPointSize(20)
        self.ctlTitle.setFont(font)
        # add title to VBox
        self.ctlPanel.addWidget(self.ctlTitle)

        # create switch viewing radio button
        self.viewSwitchLayout = QHBoxLayout() # create horizontal layout for radio buttons
        # self.viewSwitchLayout.addStretch()
        self.viewSwitchLayout.setAlignment(Qt.AlignTop)

        self.radioAP = QRadioButton("Coronal") # viewer id 0 # todo: make font bigger
        self.radioDV = QRadioButton("Horizontal") # viewer id 1
        self.radioML = QRadioButton("Sagittal") # viewer id 2
        # set fixed height
        self.radioAP.setFixedHeight(100)
        self.radioDV.setFixedHeight(100)
        self.radioML.setFixedHeight(100)
        # style radio button
        self.radioAP.setStyleSheet('QRadioButton{font: 15pt Helvetica MS;} QRadioButton::indicator { width: 20px; height: 20px;};')
        self.radioDV.setStyleSheet('QRadioButton{font: 15pt Helvetica MS;} QRadioButton::indicator { width: 20px; height: 20px;};')
        self.radioML.setStyleSheet('QRadioButton{font: 15pt Helvetica MS;} QRadioButton::indicator { width: 20px; height: 20px;};')
        # button clicked behavior
        self.radioAP.clicked.connect(lambda: probeV.switchViewer(0))
        self.radioDV.clicked.connect(lambda: probeV.switchViewer(1))
        self.radioML.clicked.connect(lambda: probeV.switchViewer(2))
        # check Axial section by default
        self.radioDV.setChecked(True) 
        # add radio buttons to horizontal box layout
        self.viewSwitchLayout.addWidget(self.radioAP,alignment=Qt.AlignTop)
        self.viewSwitchLayout.addWidget(self.radioDV,alignment=Qt.AlignTop)
        self.viewSwitchLayout.addWidget(self.radioML,alignment=Qt.AlignTop)
        # add HBox layout to grid
        self.ctlPanel.addLayout(self.viewSwitchLayout)

    def createProbeSelector(self,probeV):
        # create probe selector, get number of probes in CSV file, create check box accordingly
        number_of_probes = len(probeV.probe_axis)
        show_info(f'{number_of_probes} probe(s) added.')
        self.probeSelectLayout = QVBoxLayout()
        self.probeSelectLayout.setAlignment(Qt.AlignTop)
        self.probeCheckBoxList = []
        self.probeDisplayList = []
        
        self.probeColor = np.rint(np.array(dc.get_colors(number_of_probes)) * 255) # distinctipy
        
        for p in range(len(probeV.probe_axis)):
            probeCB = QCheckBox("neuropixels_probe"+str(p))
            # size policy, minimum expanding
            probeCB.setSizePolicy(QSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)) # shrinking
            # style probe check button
            bgr_string = [str(self.probeColor[p][0]),str(self.probeColor[p][1]),str(self.probeColor[p][2])]
            probeCB.setStyleSheet('QCheckBox{font: 15pt Helvetica MS; color: '+'rgb({0},{1},{2})'.format(bgr_string[2],bgr_string[1],bgr_string[0])+'; border-width: 5px; border-style: solid; border-color: '+'rgb({0},{1},{2})'.format(bgr_string[2],bgr_string[1],bgr_string[0])+';} QCheckBox::indicator { width: 25px; height: 25px;};')
            probeCB.setChecked(True)
            callback = functools.partial(lambda p: self.probeCheckBoxChanged(p,probeV), p=p)
            probeCB.stateChanged.connect(callback) # connect to probe display status
            self.probeCheckBoxList.append(probeCB) # add probe checkbox to list
            self.probeDisplayList.append(1) # add probe display/hide status to list
            self.probeSelectLayout.addWidget(probeCB)
        self.ctlPanel.addLayout(self.probeSelectLayout)
        self.ctlPanel.addStretch() # push stacks to top

    def probeCheckBoxChanged(self,probe_index,probeV):
        bgr_string = [str(self.probeColor[probe_index][0]),str(self.probeColor[probe_index][1]),str(self.probeColor[probe_index][2])]
        if self.probeDisplayList[probe_index] == 1: # showing
            self.probeDisplayList[probe_index] = 0 # change to hide
            self.probeCheckBoxList[probe_index].setStyleSheet('QCheckBox{font: 15pt Helvetica MS; color: '+'rgb({0},{1},{2})'.format(bgr_string[2],bgr_string[1],bgr_string[0])+'; border-width: 5px; border-style: solid; border-color: rgb(255,255,255);} QCheckBox::indicator { width: 25px; height: 25px;};')
            # print('probe_'+str(probe_index+1),' HIDE')
        else: # hiding
            self.probeDisplayList[probe_index] = 1 # change to show
            self.probeCheckBoxList[probe_index].setStyleSheet('QCheckBox{font: 15pt Helvetica MS; color: '+'rgb({0},{1},{2})'.format(bgr_string[2],bgr_string[1],bgr_string[0])+'; border-width: 5px; border-style: solid; border-color: '+'rgb({0},{1},{2})'.format(bgr_string[2],bgr_string[1],bgr_string[0])+';} QCheckBox::indicator { width: 25px; height: 25px;};')
            # print('probe_'+str(probe_index+1),' SHOW')
        self.loadSlice(probeV)
        





