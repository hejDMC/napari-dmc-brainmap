from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget,QStackedLayout,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QMainWindow,QMessageBox
from PyQt5 import QtGui
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.AnchorRow import AnchorRow
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.HelperModel import HelperModel

class RegistrationHelper(QMainWindow):
    def __init__(self, regViewer) -> None:
        super().__init__()
        self.regViewer = regViewer
        self.helperModel = HelperModel(regViewer)
        self.setWindowTitle("Registration Helper")
        self.setFixedSize(int(regViewer.fullWindowSize[0]/2),regViewer.fullWindowSize[1])
        self.mainWidget = QWidget()
        # setup layout
        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)
        self.buttonlayout = QHBoxLayout()
        self.stacklayout = QStackedLayout()
        self.mainLayout.addLayout(self.buttonlayout)
        self.mainLayout.addLayout(self.stacklayout)
        # buttons
        self.btn_ip = QPushButton("Interpolate Position")
        self.btn_re = QPushButton("Registration Editor")

        self.btn_ip.clicked.connect(self.activate_ipPage)
        self.btn_re.clicked.connect(self.activate_rePage)

        self.buttonlayout.addWidget(self.btn_ip)
        self.buttonlayout.addWidget(self.btn_re)
        # tab structure: Interpolate Position
        self.ip_widget = QWidget()
        self.ip_vbox = QVBoxLayout()
        # self.ip_vbox.addStretch() # align to the top
        self.ip_widget.setLayout(self.ip_vbox)
        self.stacklayout.addWidget(self.ip_widget) # only addChildLayout under stacklayout
        # location plot hbox
        self.locplot_hbox = QHBoxLayout()
        self.ip_vbox.addLayout(self.locplot_hbox)
        # preview button and add button vbox
        self.previewadd_vbox = QVBoxLayout()
        self.locplot_hbox.addLayout(self.previewadd_vbox)
            # preview button
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_action)
        self.preview_btn.setDisabled(True) # gray out by default
        self.previewadd_vbox.addWidget(self.preview_btn)
            # add button
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_action)
        self.previewadd_vbox.addWidget(self.add_btn)
            # section location illustration
        self.preview_label = QLabel()
        # self.preview_label.setFixedSize()

        # numpy array to QImage
        h,w,_ = self.helperModel.img0.shape
        previewimg_init = QtGui.QImage(self.helperModel.img0.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(previewimg_init))


        self.locplot_hbox.addWidget(self.preview_label)
            # abort and apply buttons in a vbox
        self.abort_apply_vbox = QVBoxLayout()
        self.locplot_hbox.addLayout(self.abort_apply_vbox)
        self.abort_btn = QPushButton("Abort")
        self.apply_btn = QPushButton("Apply")
        self.abort_btn.setDisabled(True) # gray out by default
        self.apply_btn.setDisabled(True) # gray out by default
        self.abort_btn.clicked.connect(self.abort_action)
        self.apply_btn.clicked.connect(self.apply_action)
        self.abort_apply_vbox.addWidget(self.abort_btn)
        self.abort_apply_vbox.addWidget(self.apply_btn)

        # anchor widget
        self.anchor_widget = QWidget()
        self.anchor_vbox = QVBoxLayout()
        self.anchor_widget.setLayout(self.anchor_vbox)
        self.ip_vbox.addWidget(self.anchor_widget)



        # tab structure: Registration Editor
        self.label_re = QLabel()
        self.label_re.setText("editor")
        self.stacklayout.addWidget(self.label_re)
        # set default display Interpolate Position tab
        self.stacklayout.setCurrentIndex(0)
        self.preview_mode = 0

    def activate_ipPage(self):
        self.stacklayout.setCurrentIndex(0)
    
    def activate_rePage(self):
        self.stacklayout.setCurrentIndex(1)
    
    def add_action(self):
        # create anchor object
        AnchorRow(self) # HelperModel takes care of update button availability
    
    def preview_action(self):
        # freeze anchor settings
        self.update_button_availability(status_code=2)
        # show atlas/sample locations in regViewer, lock change
        self.activate_preview_mode()
        # backup and overwrite atlasLocation dictionary
        self.atlasLocation_backup = self.regViewer.status.atlasLocation.copy()
        for k,v in self.helperModel.mapping_dict.items():
            self.regViewer.status.atlasLocation[k] = [self.regViewer.status.atlasLocation[k][0],
                                                      self.regViewer.status.atlasLocation[k][1],
                                                      v]
        # update atlas viewer
        self.regViewer.status.current_z = self.regViewer.status.atlasLocation[
            self.regViewer.status.currentSliceNumber][2]
        self.regViewer.widget.viewerLeft.loadSlice()
        # show transformation overlay
        if self.regViewer.status.currentSliceNumber in self.regViewer.status.blendMode:
            self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] = 1 # overlay
            self.regViewer.atlasModel.updateDotPosition(mode='force')


    def abort_action(self):
        # restore editing
        self.update_button_availability(status_code=3)
        # restore viewing
        self.deactivate_preview_mode()
        # restore previous atlasLocation dictionary
        self.regViewer.status.atlasLocation = self.atlasLocation_backup.copy()
        del self.atlasLocation_backup
        # update atlas viewer
        self.regViewer.status.current_z = self.regViewer.status.atlasLocation[
            self.regViewer.status.currentSliceNumber][2]
        self.regViewer.widget.viewerLeft.loadSlice()
        # show transformation overlay
        if self.regViewer.status.currentSliceNumber in self.regViewer.status.blendMode:
            self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] = 1 # overlay
            self.regViewer.atlasModel.updateDotPosition(mode='force')


    
    def apply_action(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Apply Interpolated Position Confirmation")
        msg.setText("Are you sure you want to overwrite slice location information?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        feedback = msg.exec_()
        print("User chose {}".format(feedback))
        if feedback == QMessageBox.Yes:
            self.update_button_availability(status_code=4)
        else:
            pass
    
    
    def update_button_availability(self,status_code):
        # status 1: more than 1 different anchors, ready for preview
        if status_code == 1:
            if (len(self.helperModel.mapping_dict.keys())<1 # empty mapping_dict
                ) | (self.regViewer.widget.toggle.isChecked()): # or transformation mode active
                self.preview_btn.setDisabled(True) # gray-out preview button
            else:
                self.preview_btn.setEnabled(True)

        # status 2: during preview, Add and Preview buttons, and anchorrows become unavailable,
        # while Abort and Apply buttons become available.
        elif status_code == 2:
            self.preview_btn.setDisabled(True)
            self.add_btn.setDisabled(True)
            self.abort_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
            # disable spinboxes and buttons in active anchors
            for anc in self.helperModel.active_anchor:
                anc.spinSliceIndex.setDisabled(True)
                anc.spinAPmm.setDisabled(True)
                anc.trash_btn.setDisabled(True)

        # status 3: pressed Abort during preview, restore default button state
        elif status_code == 3:
            self.preview_btn.setEnabled(True)
            self.add_btn.setEnabled(True)
            self.abort_btn.setDisabled(True)
            self.apply_btn.setDisabled(True)
            # disable spinboxes and buttons in active anchors
            for anc in self.helperModel.active_anchor:
                anc.spinSliceIndex.setEnabled(True)
                anc.spinAPmm.setEnabled(True)
                anc.trash_btn.setEnabled(True)
            
        # status 4: pressed Apply during preview, (prompt confirmation dialogue), disable all buttons and spinboxes
        elif status_code == 4:
            self.abort_btn.setDisabled(True)
            self.apply_btn.setDisabled(True)
        
        else:
            print("Warning: button availability updated without specified status code! "+
                  "Check and fix this!")
    
    def activate_preview_mode(self):
        self.preview_mode = 1
        self.regViewer.widget.x_slider.setDisabled(True)
        self.regViewer.widget.y_slider.setDisabled(True)
        self.regViewer.widget.z_slider.setDisabled(True)
        self.regViewer.widget.toggle.setDisabled(True)
        self.regViewer.widget.viewerLeft.view.setInteractive(False)
        self.regViewer.widget.viewerRight.view.setInteractive(False)
    
    def deactivate_preview_mode(self):
        self.preview_mode = 0
        self.regViewer.widget.x_slider.setEnabled(True)
        self.regViewer.widget.y_slider.setEnabled(True)
        self.regViewer.widget.z_slider.setEnabled(True)
        self.regViewer.widget.toggle.setEnabled(True)
        self.regViewer.widget.viewerLeft.view.setInteractive(True)
        self.regViewer.widget.viewerRight.view.setInteractive(True)


    







    




