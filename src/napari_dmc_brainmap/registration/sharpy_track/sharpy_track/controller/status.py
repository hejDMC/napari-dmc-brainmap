import numpy as np
from napari_dmc_brainmap.utils.atlas_utils import coord_mm_transform, get_decimal
from PyQt5.QtCore import Qt
import json
from PyQt5.QtWidgets import QMessageBox

class StatusContainer():
    def __init__(self, regViewer) -> None:
        self.regViewer = regViewer
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
        self.xyz_dict = regViewer.regi_dict['xyz_dict']
        self.z_idx = regViewer.atlasModel.z_idx
        self.bregma = regViewer.atlasModel.bregma
        self.imgFileName = None
        self.folderPath = None
        self.imageRGB = False
        self.calculate_z_decimal()

    
    def calculate_z_decimal(self):
        step_float = self.xyz_dict['z'][2] / 1000
        self.decimal = get_decimal([self.xyz_dict['z'][2]])[0]
        self.z_step = np.round(step_float,self.decimal)



    def sampleChanged(self):
        self.currentSliceNumber = self.regViewer.widget.sampleSlider.value()
        self.regViewer.widget.imageTitle.setText(str(self.currentSliceNumber)+'---'+self.imgFileName[self.currentSliceNumber])
        self.regViewer.widget.viewerRight.loadSample()
        self.regViewer.widget.viewerLeft.loadSlice()
        if self.currentSliceNumber in self.atlasDots: # check if dots saving list is created
            pass
        else:
            self.atlasDots[self.currentSliceNumber] = []
            self.sampleDots[self.currentSliceNumber] = [] # if not create empty list

        while len(self.regViewer.widget.viewerLeft.itemGroup) > 0:
            self.regViewer.widget.removeRecentDot() # clear dots
        # clear dots record at atlasmodel
        self.regViewer.atlasModel.atlas_pts = []
        self.regViewer.atlasModel.sample_pts = []
        self.regViewer.atlasModel.checkSaved()


    def z_changed(self):
        self.current_z = np.round(coord_mm_transform([0], [self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]]) - 
                                      self.regViewer.widget.z_slider.value() /
                                        (1000/self.xyz_dict['z'][2]), self.decimal) # adapt Z step from atlas resolution
        self.regViewer.widget.viewerLeft.loadSlice()

    def x_changed(self):
        self.x_angle = np.round(self.regViewer.widget.x_slider.value() / 10, 1)
        self.regViewer.widget.viewerLeft.loadSlice()
    
    def y_changed(self):
        self.y_angle = np.round(self.regViewer.widget.y_slider.value() / 10, 1)
        self.regViewer.widget.viewerLeft.loadSlice()
    
    def toggleChanged(self):
        if self.regViewer.widget.toggle.isChecked():
            self.tMode = 1 # ON
            self.regViewer.widget.z_slider.setDisabled(True) # lock Sliders, prevent user from changing
            self.regViewer.widget.x_slider.setDisabled(True) # when in transformation mode
            self.regViewer.widget.y_slider.setDisabled(True)
            self.regViewer.widget.sampleSlider.setDisabled(True)
            self.regViewer.widget.viewerLeft.view.setInteractive(True)
            self.regViewer.widget.viewerRight.view.setInteractive(True)
            self.atlasLocation[self.currentSliceNumber] = [self.x_angle, self.y_angle, self.current_z] # refresh atlasLocation
            
        else:
            self.tMode = 0 # OFF
            self.regViewer.widget.z_slider.setDisabled(False) # restore responsive Slider
            self.regViewer.widget.x_slider.setDisabled(False)
            self.regViewer.widget.y_slider.setDisabled(False)
            self.regViewer.widget.sampleSlider.setDisabled(False)
            self.regViewer.widget.viewerLeft.view.setInteractive(False)
            self.regViewer.widget.viewerRight.view.setInteractive(False)


    def wheelEventHandle(self, event):
        # filter mouse wheel event
        ## update viewerLeft
        if (self.cursor == -1) & (self.tMode == 0) & (self.regViewer.widget.toggle.isEnabled()): # tMode OFF, inside viewerLeft
            if event.angleDelta().y() < 0: # scrolling towards posterior
                self.current_z -= self.z_step
                self.current_z = np.round(self.current_z, self.decimal)
            elif event.angleDelta().y() > 0: # scrolling towards anterior
                self.current_z += self.z_step
                self.current_z = np.round(self.current_z, self.decimal)
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
            self.regViewer.widget.z_slider.setSliderPosition(coord_mm_transform([self.current_z], [self.bregma[self.z_idx]],
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
            self.regViewer.widget.sampleSlider.setSliderPosition(self.currentSliceNumber)
            # regViewer.widget.viewerRight.loadSample(regViewer)
        else:
            pass

    def mousePressEventHandle(self, event):
        # only leftViewer clickable, when transformation mode is ON
        if (self.cursor == -1) & (self.tMode == 1):
            # left mouse click
            if event.button() == Qt.LeftButton:
                # map regViewer coordinates to view
                self.pressPos = self.regViewer.widget.viewerLeft.view.mapFrom(self.regViewer,event.pos()) 
                self.regViewer.widget.addDots()

            elif event.button() == Qt.RightButton:
                # delete most recent added pair of dots
                self.regViewer.widget.removeRecentDot()
    
    def saveRegistration(self):
        # if preview mode is on, will not save registration
        save_exec = 1
        if hasattr(self.regViewer,"interpolatePositionPage"):
            if self.regViewer.interpolatePositionPage.preview_mode == 1:
                save_exec = 0
            else:
                pass
        else:
            pass

        if save_exec:
            # the only place where writing to registration json happens
            with open(self.folderPath.joinpath('registration.json'), 'w') as f:
                reg_data = {'atlasLocation': self.atlasLocation,
                            'atlasDots': self.atlasDots,
                            'sampleDots': self.sampleDots,
                            'imgName': self.imgNameDict}
                json.dump(reg_data, f)
        else:
            pass

    def keyPressEventHandle(self, event):
        if (event.key() == 50) & (self.tMode == 0) & (self.regViewer.widget.toggle.isEnabled()): # pressed numpad 2
            self.y_angle = np.round(self.y_angle - (self.xyz_dict['y'][2] / 100), 1)
            self.regViewer.widget.y_slider.setSliderPosition(int(self.y_angle * 10))
            self.regViewer.widget.viewerLeft.loadSlice()

        elif (event.key() == 52) & (self.tMode == 0) & (self.regViewer.widget.toggle.isEnabled()): # pressed numpad 4
            self.x_angle = np.round(self.x_angle - (self.xyz_dict['x'][2] / 100), 1)
            self.regViewer.widget.x_slider.setSliderPosition(int(self.x_angle * 10))
            self.regViewer.widget.viewerLeft.loadSlice()

        elif (event.key() == 54) & (self.tMode == 0) & (self.regViewer.widget.toggle.isEnabled()): # pressed numpad 6
            self.x_angle = np.round(self.x_angle + (self.xyz_dict['x'][2] / 100), 1)
            self.regViewer.widget.x_slider.setSliderPosition(int(self.x_angle * 10))
            self.regViewer.widget.viewerLeft.loadSlice()

        elif (event.key() == 56) & (self.tMode == 0) & (self.regViewer.widget.toggle.isEnabled()): # pressed numpad 8
            self.y_angle = np.round(self.y_angle + (self.xyz_dict['y'][2] / 100), 1)
            self.regViewer.widget.y_slider.setSliderPosition(int(self.y_angle * 10))
            self.regViewer.widget.viewerLeft.loadSlice()


        elif event.key() == Qt.Key_T: # T for transformation mode
            if hasattr(self.regViewer.widget, 'toggle'):
                self.regViewer.widget.toggle.click()
        
        elif event.key() == Qt.Key_A: # A for showing brain region outline
            # if hasattr(self.regViewer.atlasModel,'outlineBool'):
            if self.contour == 0:
                self.regViewer.atlasModel.displayContour()
            else:
                self.regViewer.atlasModel.hideContour()

        elif event.key() == Qt.Key_Z: # ZXC for blendMode
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 0 # all atlas
                self.regViewer.atlasModel.updateDotPosition(mode='force')
        elif event.key() == Qt.Key_X:
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 1 # overlay
                self.regViewer.atlasModel.updateDotPosition(mode='force')
        elif event.key() == Qt.Key_C:
            if self.currentSliceNumber in self.blendMode:
                self.blendMode[self.currentSliceNumber] = 2 # all sample
                self.regViewer.atlasModel.updateDotPosition(mode='force')
        
        # press D for deleting all paired dots at current slide
        elif event.key() == Qt.Key_D:
            if self.regViewer.widget.toggle.isChecked():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Remove all dots")
                msg.setText("Do you want to delete all paired dots at current slice? \n* Choose 'YES' will delete all dots at current slice.")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                feedback = msg.exec_()
                if feedback == msg.Yes:
                    while len(self.regViewer.widget.viewerLeft.itemGroup) > 0:
                        # remove dots from scene
                        self.regViewer.widget.viewerLeft.scene.removeItem(self.regViewer.widget.viewerLeft.itemGroup[-1])
                        self.regViewer.widget.viewerRight.scene.removeItem(self.regViewer.widget.viewerRight.itemGroup[-1])
                        # remove dots from itemGroup storage
                        self.regViewer.widget.viewerLeft.itemGroup = self.regViewer.widget.viewerLeft.itemGroup[:-1]
                        self.regViewer.widget.viewerRight.itemGroup = self.regViewer.widget.viewerRight.itemGroup[:-1]
                    self.regViewer.atlasModel.atlas_pts = []
                    self.regViewer.atlasModel.sample_pts = []
                    self.atlasDots[self.regViewer.status.currentSliceNumber] = []
                    self.sampleDots[self.regViewer.status.currentSliceNumber] = []
                    self.saveRegistration()
                    del self.blendMode[self.currentSliceNumber]

                else:
                    pass
            else:
                print("To remove all dots, turn on registration mode (T) first!")
        
                
        # press M to add a new measurement to measurement page
        elif event.key() == Qt.Key_M:
            if self.regViewer.measurementAct.isEnabled():
                pass 
            else:
                assert hasattr(self.regViewer, 'measurementPage')
                if (self.regViewer.measurementPage.ui.pages.currentIndex() == 0
                ) & (self.regViewer.measurementPage.measurement_state == "ready"):
                    self.regViewer.measurementPage.ui.addMeasurementBtn.click()
                    # if cursor already in viewerRight, trigger enterEvent
                    if self.regViewer.status.cursor == 1:
                        self.regViewer.measurementPage.show_measurement_pointer()
                else:
                    pass




