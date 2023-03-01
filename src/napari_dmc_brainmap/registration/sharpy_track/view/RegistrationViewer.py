#!/usr/bin/python3
# -*- coding: utf-8 -*-

# from PyQt5.QtPrintSupport import QPrintDialog
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QMenu, QAction, QFileDialog
from sharpy_track.view.ModeToggle import ModeToggle
from sharpy_track.view.DotObject import DotObject
from sharpy_track.view.MainWidget import MainWidget
import json
from sharpy_track.model.AtlasModel import AtlasModel
from sharpy_track.model.calculation import *
from sharpy_track.controller.status import StatusContainer

class RegistrationViewer(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        # create statusContainer
        self.status = StatusContainer()
        # create atlasModel
        self.atlasModel = AtlasModel()

        self.setFixedSize(2350, 940) # a suitable size for 2K screen or higher
        self.setWindowTitle("Registration Viewer")
        self.createActions()
        self.createMenus()
        self.loadAct.setEnabled(False)
    
        # create widget container
        self.widget = MainWidget(self)
        # set mainWidget central
        self.setCentralWidget(self.widget.widget)
    


    def wheelEvent(self,event):
        self.status.wheelEventHandle(self,event)

    def mousePressEvent(self, event):
        self.status.mousePressEventHandle(self,event)

    def keyPressEvent(self, event):
        self.status.keyPressEventHandle(self,event)

    # menu related functions
    def openFolder(self):
        self.status.folderPath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if (self.status.folderPath is None) | (self.status.folderPath == ''):
            pass
        else:
            files_all = natsorted(os.listdir(self.status.folderPath))
            self.status.imgFileName = []
            for f in files_all:
                if not(f.startswith('.')) and (f.endswith('.tif') or f.endswith('.tiff')):
                    self.status.imgFileName.append(f)
                else:
                    pass
            self.status.sliceNum = len(self.status.imgFileName)
            if self.status.sliceNum == 0:
                pass
            else:
                for n in range(self.status.sliceNum):
                    self.status.imgNameDict[n] = self.status.imgFileName[n]
                # initiate image stack matrix
                self.atlasModel.getStack(self)
                # create widgets
                self.widget.createImageTitle(self) # create title first
                self.widget.createSampleSlider(self)
                self.widget.createTransformToggle(self)

                self.widget.viewerRight.loadSample(self)
                self.status.sampleChanged(self) # manually call sampledChanged for loading first slice
                self.loadAct.setEnabled(True) # enable load json option
                # check if registration JSON exist
                if os.path.isfile(os.path.join(self.status.folderPath,'registration.json')):
                    # pop up window
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Registration JSON found!")
                    msg.setText("There is previous registration record. \nDo want to load them? \n* Choose 'No' will overwrite previous record.")
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg.setDefaultButton(QMessageBox.Yes)
                    feedback = msg.exec_()
                    if feedback == msg.Yes:
                        self.status.jsonPath = os.path.join(self.status.folderPath,'registration.json')
                        with open (self.status.jsonPath,'r') as jsonData:
                            jsonDict = json.load(jsonData)
                            self.status.atlasLocation = {int(k):v for k,v in jsonDict['atlasLocation'].items()}
                            self.status.atlasDots = {int(k):v for k,v in jsonDict['atlasDots'].items()}
                            self.status.sampleDots = {int(k):v for k,v in jsonDict['sampleDots'].items()}
                            self.status.imgNameDict = {int(k):v for k,v in jsonDict['imgName'].items()}
                        self.status.toggleChanged(self)
                        self.atlasModel.checkSaved(self)
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

            self.status.toggleChanged(self)
            self.atlasModel.checkSaved(self)
            
    def about(self):
        QMessageBox.about(self, "About Registration Viewer",
                          "<p> A software to align sample microscopic image to AllenCCF </p>")

    def createActions(self):
        self.openAct = QAction("O&pen Folder", self, shortcut="Ctrl+O", triggered=self.openFolder)
        self.loadAct = QAction("L&oad JSON", self, shortcut="Ctrl+L",triggered=self.loadJSON)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.aboutAct = QAction("&About", self, triggered=self.about)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addAction(self.exitAct)
        
        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.helpMenu)

