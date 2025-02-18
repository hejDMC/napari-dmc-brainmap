#!/usr/bin/python3
# -*- coding: utf-8 -*-

# from PyQt5.QtPrintSupport import QPrintDialog
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QMenu, QAction, QFileDialog, QApplication
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.ModeToggle import ModeToggle
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.DotObject import DotObject
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.MainWidget import MainWidget
import json
import numpy as np
from natsort import natsorted
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.AtlasModel import AtlasModel
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.controller.status import StatusContainer
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.Tools import RegistrationHelper

class RegistrationViewer(QMainWindow):
    def __init__(self, regViewerWidget, regi_dict) -> None:
        super().__init__()
        self.regViewerWidget = regViewerWidget
        self.app = regViewerWidget.viewer
        self.regi_dict = regi_dict
        QAppInstance = QApplication.instance()  # get current QApplication Instance
        self.screenSize = [QAppInstance.primaryScreen().size().width(), QAppInstance.primaryScreen().size().height()]
        self.applySizePolicy()
        # create widget container
        self.widget = MainWidget(self)
        # set mainWidget central
        self.setCentralWidget(self.widget)

        # create atlasModel
        self.atlasModel = AtlasModel(self)
        self.widget.viewerLeft.scene.changed.connect(lambda: self.atlasModel.updateDotPosition(mode="default"))
        self.widget.viewerRight.scene.changed.connect(lambda: self.atlasModel.updateDotPosition(mode="default"))

        # create statusContainer
        self.status = StatusContainer(self)
        self.widget.viewerLeft.view.leaveEvent = self.widget.viewerLeft.leaveLabel
        self.widget.viewerRight.view.leaveEvent = self.widget.viewerRight.leaveLabel
        self.widget.viewerLeft.view.enterEvent = self.widget.viewerLeft.hoverLeft
        self.widget.viewerRight.view.enterEvent = self.widget.viewerRight.hoverRight

        self.widget.viewerLeft.view.mouseMoved.connect(self.widget.viewerLeft.getCursorPos) # connect mouseTracking only for left viewer
        self.widget.viewerLeft.loadSlice()
        self.widget.viewerRight.loadSample()

        self.setFixedSize(self.fullWindowSize[0],self.fullWindowSize[1])
        self.setWindowTitle("Registration Viewer")
        self.createActions()
        self.createMenus()

        self.load_data()
    
    def applySizePolicy(self):
        xyz_dict = self.regi_dict['xyz_dict']
        self.atlas_resolution = [xyz_dict['x'][1], xyz_dict['y'][1]]
        if self.screenSize[0] > round(self.atlas_resolution[0]*2.2) and self.screenSize[1] > round(self.atlas_resolution[1] * 1.1):  # 2x width (plus margin) and 1x height (plus margin)
            self.scaleFactor = 1
            self.fullWindowSize = [self.atlas_resolution[0]*2+100, self.atlas_resolution[1]+150]
            self.singleWindowSize = self.atlas_resolution

        else: # [1920,1080] resolution
            # check for longer edge on x and y axis
            if self.atlas_resolution[0] >= self.atlas_resolution[1]:
                self.scaleFactor = round(self.screenSize[0]/(self.atlas_resolution[0] * 2.5), 2)
            else:
                self.scaleFactor = round(self.screenSize[1]/(self.atlas_resolution[1] * 1.7), 2)

            self.fullWindowSize = [int(round((self.atlas_resolution[0]*2.2) * self.scaleFactor)),
                                       int(round((self.atlas_resolution[1]*1.25) * self.scaleFactor))]
            self.singleWindowSize = [int(i*self.scaleFactor) for i in self.atlas_resolution]
            
        # set dotObject size
        if (int(10 * self.scaleFactor) % 2) != 0:
            self.dotRR = int(10 * self.scaleFactor) + 1
        else:
            self.dotRR = int(10 * self.scaleFactor)
        
        # get resolution pixel mapping
        low = np.arange(np.max(self.singleWindowSize))
        low_up = (low/self.scaleFactor).astype(int)
        self.res_up = {k:v for k,v in zip(low,low_up)}
        self.res_x_range = np.arange(self.singleWindowSize[0])
        self.res_y_range = np.arange(self.singleWindowSize[1])

        high = np.arange(np.max(self.atlas_resolution))
        high_down = (high*self.scaleFactor).astype(int)
        self.res_down = {k:v for k,v in zip(high,high_down)}
        # correct for res_down
        for k,v in self.res_up.items():
            self.res_down[v] = k
    

    def wheelEvent(self,event):
        self.status.wheelEventHandle(event)

    def mousePressEvent(self, event):
        self.status.mousePressEventHandle(event)

    def keyPressEvent(self, event):
        self.status.keyPressEventHandle(event)

    # menu related functions
    def load_data(self):
        self.status.folderPath = self.regi_dict['regi_dir']
        # self.status.folderPath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if (self.status.folderPath is None) | (self.status.folderPath == ''):
            pass
        else:
            self.status.imgFileName = natsorted([f.parts[-1] for f in self.status.folderPath.glob('*.tif')])
            self.status.sliceNum = len(self.status.imgFileName)
            if self.status.sliceNum == 0:
                pass
            else:
                for n in range(self.status.sliceNum):
                    self.status.imgNameDict[n] = self.status.imgFileName[n]
                # initiate image stack matrix
                self.atlasModel.getStack()
                # create widgets
                self.widget.createImageTitle() # create title first
                self.widget.createSampleSlider()
                self.widget.createTransformToggle()
                self.widget.connect_actions()

                self.widget.viewerRight.loadSample()
                self.status.sampleChanged() # manually call sampledChanged for loading first slice
                # self.loadAct.setEnabled(True) # enable load json option
                # check if registration JSON exist
                if self.status.folderPath.joinpath('registration.json').is_file():
                    # pop up window
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Registration JSON found!")
                    msg.setText("There is previous registration record. \nDo want to load them? \n* Choose 'No' will overwrite previous record.")
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg.setDefaultButton(QMessageBox.Yes)
                    feedback = msg.exec_()
                    if feedback == msg.Yes:
                        self.status.jsonPath = self.status.folderPath.joinpath('registration.json')
                        with open (self.status.jsonPath,'r') as jsonData:
                            jsonDict = json.load(jsonData)
                            self.status.atlasLocation = {int(k):v for k,v in jsonDict['atlasLocation'].items()}
                            self.status.atlasDots = {int(k):v for k,v in jsonDict['atlasDots'].items()}
                            self.status.sampleDots = {int(k):v for k,v in jsonDict['sampleDots'].items()}
                            self.status.imgNameDict = {int(k):v for k,v in jsonDict['imgName'].items()}
                        self.status.toggleChanged()
                        self.atlasModel.checkSaved()
                    else:
                        pass

    def loadJSON(self):
        self.status.jsonPath = QFileDialog.getOpenFileName(self, "Choose registration JSON","","JSON File (*.json)")[0]
        if self.status.jsonPath == "":
            pass
        else:
            with open (self.status.jsonPath,'r') as jsonData:
                jsonDict = json.load(jsonData)
                self.status.atlasLocation = {int(k):v for k,v in jsonDict['atlasLocation'].items()}
                self.status.atlasDots = {int(k):v for k,v in jsonDict['atlasDots'].items()}
                self.status.sampleDots = {int(k):v for k,v in jsonDict['sampleDots'].items()}
                self.status.imgNameDict = {int(k):v for k,v in jsonDict['imgName'].items()}

            self.status.toggleChanged()
            self.atlasModel.checkSaved()
            

    def createActions(self):
        self.helperAct = QAction("Registration H&elper", self, shortcut="Ctrl+H", triggered=self.helperPageOpen)


    def createMenus(self):
        self.helperMenu = QMenu("Tools", self)
        self.helperMenu.addAction(self.helperAct)
        self.helperAct.setEnabled(True)
        self.menuBar().addMenu(self.helperMenu)
    
    def helperPageOpen(self):
        self.helperAct.setEnabled(False)
        self.helperPage = RegistrationHelper(self)
        self.helperPage.show()
    
    def del_reghelper_instance(self):
        del self.helperPage.regViewer
        del self.helperPage.helperModel
        del self.helperPage.mainWidget
        del self.helperPage
        self.helperAct.setEnabled(True)


    def closeEvent(self, event) -> None:
        self.regViewerWidget.del_regviewer_instance()