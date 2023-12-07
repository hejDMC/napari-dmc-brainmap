import numpy as np
from napari_dmc_brainmap.utils import coord_mm_transform
from PyQt5.QtCore import Qt
import json
from PyQt5.QtWidgets import QApplication,QMessageBox

class StatusContainer():
    def __init__(self, regi_dict, z_idx, bregma) -> None:
        self.cursor = 0
        self.current_z = 0
        self.x_angle = 0
        self.y_angle = 0
        self.sliceNum = 0
        self.currentSliceNumber = 0
        self.tMode = 0
        self.contour = 0
        self.imgNameDict = {}
        self.atlasLocation = {}
        self.atlasDots = {}
        self.sampleDots = {}
        self.blendMode = {}
        self.xyz_dict = regi_dict['xyz_dict']
        self.atlas_resolution = [self.xyz_dict['x'][1], self.xyz_dict['y'][1]]
        self.z_idx = z_idx
        self.bregma = bregma
        QAppInstance = QApplication.instance()  # get current QApplication Instance
        self.screenSize = [QAppInstance.primaryScreen().size().width(), QAppInstance.primaryScreen().size().height()]
        self.applySizePolicy()
        self.imgFileName = None
        self.folderPath = None
        self.imageRGB = False

    def applySizePolicy(self):
        if self.screenSize[0] > round(self.atlas_resolution[0]*2.2) and self.screenSize[1] > round(self.atlas_resolution[1] * 1.1):  # 2x width (plus margin) and 1x height (plus margin)
            self.scaleFactor = 1
            self.fullWindowSize = [self.atlas_resolution[0]*2+100, self.atlas_resolution[1]+150]
            self.singleWindowSize = self.atlas_resolution

        else: # [1920,1080] resolution
            self.scaleFactor = round(self.screenSize[0]/(self.atlas_resolution[0] * 2.5), 2)
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

        high = np.arange(np.max(self.atlas_resolution))
        high_down = (high*self.scaleFactor).astype(int)
        self.res_down = {k:v for k,v in zip(high,high_down)}
        # correct for res_down
        for k,v in self.res_up.items():
            self.res_down[v] = k
        



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
        # clear dots record at atlasmodel
        regViewer.atlasModel.atlas_pts = []
        regViewer.atlasModel.sample_pts = []
        regViewer.atlasModel.checkSaved(regViewer)

    

    def z_changed(self, regViewer):
        self.current_z = np.round(coord_mm_transform([0], [self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]]) - regViewer.widget.z_slider.value() / (1000/self.xyz_dict['z'][2]), 2) # adapt Z step from atlas resolution
        regViewer.widget.viewerLeft.loadSlice(regViewer)

    def x_changed(self, regViewer):
        self.x_angle = np.round(regViewer.widget.x_slider.value() / 10, 1)
        regViewer.widget.viewerLeft.loadSlice(regViewer)
    
    def y_changed(self, regViewer):
        self.y_angle = np.round(regViewer.widget.y_slider.value() / 10, 1)
        regViewer.widget.viewerLeft.loadSlice(regViewer)
    
    def toggleChanged(self,regViewer):
        if regViewer.widget.toggle.isChecked():
            self.tMode = 1 # ON
            regViewer.widget.z_slider.setDisabled(True) # lock Sliders, prevent user from changing
            regViewer.widget.x_slider.setDisabled(True) # when in transformation mode
            regViewer.widget.y_slider.setDisabled(True)
            regViewer.widget.sampleSlider.setDisabled(True)
            regViewer.widget.viewerLeft.view.setInteractive(True)
            regViewer.widget.viewerRight.view.setInteractive(True)
            self.atlasLocation[self.currentSliceNumber] = [self.x_angle, self.y_angle, self.current_z] # refresh atlasLocation
            
        else:
            self.tMode = 0 # OFF
            regViewer.widget.z_slider.setDisabled(False) # restore responsive Slider
            regViewer.widget.x_slider.setDisabled(False)
            regViewer.widget.y_slider.setDisabled(False)
            regViewer.widget.sampleSlider.setDisabled(False)
            regViewer.widget.viewerLeft.view.setInteractive(False)
            regViewer.widget.viewerRight.view.setInteractive(False)


    def wheelEventHandle(self, regViewer, event):
        # filter mouse wheel event
        ## update viewerLeft
        if (self.cursor == -1) & (self.tMode == 0): # tMode OFF, inside viewerLeft
            if event.angleDelta().y() < 0: # scrolling towards posterior
                self.current_z -= self.xyz_dict['z'][2] / 1000
                self.current_z = np.round(self.current_z, 2)
            elif event.angleDelta().y() > 0: # scrolling towards anterior
                self.current_z += self.xyz_dict['z'][2] / 1000
                self.current_z = np.round(self.current_z, 2)
            else:
                pass

            # within range check
            z_coord = coord_mm_transform([self.current_z], [self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]], mm_to_coord=True)

            if z_coord > self.xyz_dict['z'][1] - 1:
                self.current_z = coord_mm_transform([z_coord],[self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]])

            elif z_coord < 0:
                self.current_z = coord_mm_transform([z_coord], [self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]])
                # print("Anterior End!")
            else:
                pass
            regViewer.widget.z_slider.setSliderPosition(coord_mm_transform([self.current_z], [self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]], mm_to_coord=True))
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
        with open(self.folderPath.joinpath('registration.json'), 'w') as f:
            reg_data = {'atlasLocation': self.atlasLocation,
                        'atlasDots': self.atlasDots,
                        'sampleDots': self.sampleDots,
                        'imgName': self.imgNameDict}
            json.dump(reg_data, f)


    def keyPressEventHandle(self, regViewer, event):
        if (event.key() == 50) & (self.tMode == 0): # pressed numpad 2
            self.y_angle = np.round(self.y_angle - (self.xyz_dict['y'][2] / 1000), 1)
            regViewer.widget.y_slider.setSliderPosition(int(self.y_angle * 10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)
        elif (event.key() == 52) & (self.tMode == 0): # pressed numpad 4
            self.x_angle = np.round(self.x_angle - (self.xyz_dict['x'][2] / 1000), 1)
            regViewer.widget.x_slider.setSliderPosition(int(self.x_angle * 10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)
        elif (event.key() == 54) & (self.tMode == 0): # pressed numpad 6
            self.x_angle = np.round(self.x_angle + (self.xyz_dict['x'][2] / 1000), 1)
            regViewer.widget.x_slider.setSliderPosition(int(self.x_angle * 10))
            regViewer.widget.viewerLeft.loadSlice(regViewer)
        elif (event.key() == 56) & (self.tMode == 0): # pressed numpad 8
            self.y_angle = np.round(self.y_angle + (self.xyz_dict['y'][2] / 1000), 1)
            regViewer.widget.y_slider.setSliderPosition(int(self.y_angle * 10))
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
                regViewer.atlasModel.updateDotPosition(regViewer,mode='force')
        elif event.key() == Qt.Key_X:
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 1 # overlay
                regViewer.atlasModel.updateDotPosition(regViewer,mode='force')
        elif event.key() == Qt.Key_C:
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 2 # all sample
                regViewer.atlasModel.updateDotPosition(regViewer,mode='force')
        
        # press D for deleting all paired dots at current slide
        elif event.key() == Qt.Key_D:
            if regViewer.widget.toggle.isChecked():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Remove all dots")
                msg.setText("Do you want to delete all paired dots at current slice? \n* Choose 'YES' will delete all dots at current slice.")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                feedback = msg.exec_()
                if feedback == msg.Yes:
                    while len(regViewer.widget.viewerLeft.itemGroup) > 0:
                        # remove dots from scene
                        regViewer.widget.viewerLeft.scene.removeItem(regViewer.widget.viewerLeft.itemGroup[-1])
                        regViewer.widget.viewerRight.scene.removeItem(regViewer.widget.viewerRight.itemGroup[-1])
                        # remove dots from itemGroup storage
                        regViewer.widget.viewerLeft.itemGroup = regViewer.widget.viewerLeft.itemGroup[:-1]
                        regViewer.widget.viewerRight.itemGroup = regViewer.widget.viewerRight.itemGroup[:-1]
                    regViewer.atlasModel.atlas_pts = []
                    regViewer.atlasModel.sample_pts = []
                    self.atlasDots[regViewer.status.currentSliceNumber] = []
                    self.sampleDots[regViewer.status.currentSliceNumber] = []
                    self.saveRegistration()
                    del self.blendMode[self.currentSliceNumber]

                else:
                    pass
            else:
                print("To remove all dots, turn on registration mode (T) first!")


