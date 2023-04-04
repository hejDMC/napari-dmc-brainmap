import numpy as np
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import *
from PyQt5.QtCore import Qt
import json

class StatusContainer():
    def __init__(self,screenWidth,screenHeight) -> None:
        self.cursor = 0
        self.currentAP = 0
        self.MLangle = 0
        self.DVangle = 0
        self.sliceNum = 0
        self.currentSliceNumber = 0
        self.tMode = 0
        self.contour = 0
        self.imgNameDict = {}
        self.atlasLocation = {}
        self.atlasDots = {}
        self.sampleDots = {}
        self.blendMode = {}
        self.screenSize = [screenWidth,screenHeight]
        self.applySizePolicy()
        self.imgFileName = None
        self.folderPath = None

    def applySizePolicy(self):
        if (self.screenSize == [2560,1440]) or (self.screenSize == [3840,2160]):
            self.fullWindowSizeNarrow = [2350,940]
            self.fullWindowSizeWide = [2394,940]
            self.singleWindowSize = [1140,800]
            self.aspectRatio = 1
        else: # [1920,1080] resolution
            self.fullWindowSizeNarrow = [1762,705]
            self.fullWindowSizeWide = [1820,705]
            self.singleWindowSize = [855,600]
            self.aspectRatio = 0.75


    def sampleChanged(self,regViewer):
        self.currentSliceNumber = regViewer.widget.sampleSlider.value()
        regViewer.widget.imageTitle.setText(str(self.currentSliceNumber)+'---'+self.imgFileName[self.currentSliceNumber])
        regViewer.widget.viewerRight.loadSample(regViewer)
        regViewer.widget.viewerLeft.loadSlice(regViewer)
        if self.currentSliceNumber in self.atlasDots: # check if dots saving list is created
            pass
        else:
            self.atlasDots[self.currentSliceNumber] = []
            self.sampleDots[self.currentSliceNumber] = [] # if not create empty list

        while len(regViewer.widget.viewerLeft.itemGroup) > 0:
            regViewer.widget.removeRecentDot() # clear dots
        regViewer.atlasModel.checkSaved(regViewer)

    

    def apChanged(self,regViewer):
        self.currentAP = np.round(5.39-regViewer.widget.apSlider.value()/100,2)
        regViewer.widget.viewerLeft.loadSlice(regViewer)

    def mlChanged(self,regViewer):
        self.MLangle = np.round(regViewer.widget.mlSlider.value()/10,1)
        regViewer.widget.viewerLeft.loadSlice(regViewer)
    
    def dvChanged(self,regViewer):
        self.DVangle = np.round(regViewer.widget.dvSlider.value()/10,1)
        regViewer.widget.viewerLeft.loadSlice(regViewer)
    
    def toggleChanged(self,regViewer):
        if regViewer.widget.toggle.isChecked():
            self.tMode = 1 # ON
            regViewer.widget.apSlider.setDisabled(True) # lock Sliders, prevent user from changing 
            regViewer.widget.mlSlider.setDisabled(True) # when in transformation mode
            regViewer.widget.dvSlider.setDisabled(True)
            regViewer.widget.sampleSlider.setDisabled(True)
            regViewer.widget.viewerLeft.view.setInteractive(True)
            regViewer.widget.viewerRight.view.setInteractive(True)
            self.atlasLocation[self.currentSliceNumber] = [self.MLangle,self.DVangle,self.currentAP] # refresh atlasLocation
            
        else:
            self.tMode = 0 # OFF
            regViewer.widget.apSlider.setDisabled(False) # restore responsive Slider
            regViewer.widget.mlSlider.setDisabled(False) 
            regViewer.widget.dvSlider.setDisabled(False)
            regViewer.widget.sampleSlider.setDisabled(False)
            regViewer.widget.viewerLeft.view.setInteractive(False)
            regViewer.widget.viewerRight.view.setInteractive(False)


    def wheelEventHandle(self, regViewer, event):
        # filter mouse wheel event
        ## update viewerLeft
        if (self.cursor == -1) & (self.tMode == 0): # tMode OFF, inside viewerLeft
            if event.angleDelta().y() < 0: # scrolling towards posterior
                self.currentAP -= 0.01
                self.currentAP = np.round(self.currentAP,2)
            elif event.angleDelta().y() > 0: # scrolling towards anterior
                self.currentAP += 0.01
                self.currentAP = np.round(self.currentAP,2)
            else:
                pass

            # within range check
            if get_ap(self.currentAP) > 1319:
                self.currentAP = -7.8
                # print("Posterior End!")
            elif get_ap(self.currentAP) < 0:
                self.currentAP = 5.39
                # print("Anterior End!")
            else:
                pass
            regViewer.widget.apSlider.setSliderPosition(get_ap(self.currentAP))
            # regViewer.widget.viewerLeft.loadSlice(regViewer)
        ## update viewerRight
        elif (self.cursor == 1) & (self.sliceNum > 0) & (self.tMode == 0): # sample images loaded, inside viewerRight, tMode off
            if event.angleDelta().y() < 0: # scrolling towards posterior
                self.currentSliceNumber += 1
            elif event.angleDelta().y() > 0: # scrolling towards anterior
                self.currentSliceNumber -= 1
            else:
                pass
            # whinin range check
            if self.currentSliceNumber < 0:
                self.currentSliceNumber = 0
                # print("Already at First Slice!")
            elif self.currentSliceNumber >= self.sliceNum:
                self.currentSliceNumber = self.sliceNum - 1
                # print("Already The Last Slice!")
            else:
                pass
            regViewer.widget.sampleSlider.setSliderPosition(self.currentSliceNumber)
            # regViewer.widget.viewerRight.loadSample(regViewer)
        else:
            pass

    def mousePressEventHandle(self, regViewer, event):
        # only leftViewer clickable, when transformation mode is ON
        if (self.cursor == -1) & (self.tMode == 1):
            # left mouse click
            if event.button() == Qt.LeftButton:
                # map regViewer coordinates to view
                self.pressPos = regViewer.widget.viewerLeft.view.mapFrom(regViewer,event.pos()) 
                regViewer.widget.addDots(regViewer)

            elif event.button() == Qt.RightButton:
                # delete most recent added pair of dots
                regViewer.widget.removeRecentDot()
    
    def saveRegistration(self):
        with open(os.path.join(self.folderPath,'registration.json'), 'w') as f:
            reg_data = {'atlasLocation':self.atlasLocation,
                        'atlasDots':self.atlasDots,
                        'sampleDots':self.sampleDots,
                        'imgName':self.imgNameDict}
            json.dump(reg_data, f)


    def keyPressEventHandle(self, regViewer, event):
        if (event.key() == 50) & (self.tMode == 0): # pressed numpad 2
            self.DVangle = np.round(self.DVangle-0.1,1)
            regViewer.widget.dvSlider.setSliderPosition(int(self.DVangle*10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)
        elif (event.key() == 52) & (self.tMode == 0): # pressed numpad 4
            self.MLangle = np.round(self.MLangle-0.1,1)
            regViewer.widget.mlSlider.setSliderPosition(int(self.MLangle*10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)
        elif (event.key() == 54) & (self.tMode == 0): # pressed numpad 6
            self.MLangle = np.round(self.MLangle+0.1,1)
            regViewer.widget.mlSlider.setSliderPosition(int(self.MLangle*10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)
        elif (event.key() == 56) & (self.tMode == 0): # pressed numpad 8
            self.DVangle = np.round(self.DVangle+0.1,1)
            regViewer.widget.dvSlider.setSliderPosition(int(self.DVangle*10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)

        elif event.key() == Qt.Key_T: # T for transformation mode
            if hasattr(regViewer.widget, 'toggle'):
                regViewer.widget.toggle.click()
        
        elif event.key() == Qt.Key_A: # A for showing brain region outline
            # if hasattr(regViewer.atlasModel,'outlineBool'):
            if self.contour == 0:
                regViewer.atlasModel.displayContour(regViewer)
            else:
                regViewer.atlasModel.hideContour(regViewer)

        elif event.key() == Qt.Key_Z: # ZXC for blendMode
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 0 # all atlas
                regViewer.atlasModel.updateDotPosition(regViewer)
        elif event.key() == Qt.Key_X:
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 1 # overlay
                regViewer.atlasModel.updateDotPosition(regViewer)
        elif event.key() == Qt.Key_C:
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 2 # all sample
                regViewer.atlasModel.updateDotPosition(regViewer)


