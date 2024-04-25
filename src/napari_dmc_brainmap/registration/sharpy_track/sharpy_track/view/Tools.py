from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget,QStackedLayout,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QMainWindow,QMessageBox,QTableView,QDialog,QDialogButtonBox
from PyQt5 import QtGui
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.AnchorRow import AnchorRow
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.HelperModel import HelperModel
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.PandasModel import PandasModel
import pandas as pd

class RegistrationHelper(QMainWindow):
    def __init__(self, regViewer) -> None:
        super().__init__()
        self.regViewer = regViewer
        self.helperModel = HelperModel(regViewer)
        self.setWindowTitle("Registration Helper")
        self.setFixedSize(int(regViewer.fullWindowSize[0]/1.5),regViewer.fullWindowSize[1])
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
        self.btn_re = QPushButton("Keyboard shortcuts")

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
        self.label_re.setPixmap(QPixmap(str(
            self.regViewer.atlasModel.sharpy_dir.joinpath(
                'sharpy_track',
                'sharpy_track',
                'images',
                'helperpage_shortcuts.png'))))

        # Optional: Adjust the QLabel size to fit the image
        self.label_re.setScaledContents(True)
        self.label_re.resize(self.label_re.pixmap().size())
        

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
            self.regViewer.status.atlasLocation[k] = [self.regViewer.status.x_angle,
                                                      self.regViewer.status.y_angle,
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
        if self.regViewer.status.currentSliceNumber in self.regViewer.status.atlasLocation:
            self.regViewer.status.current_z = self.regViewer.status.atlasLocation[
                self.regViewer.status.currentSliceNumber][2]
        else:
            pass
        self.regViewer.widget.viewerLeft.loadSlice()
        # show transformation overlay
        if self.regViewer.status.currentSliceNumber in self.regViewer.status.blendMode:
            self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] = 1 # overlay
            self.regViewer.atlasModel.updateDotPosition(mode='force')
        else:
            pass


    
    def apply_action(self):
        # create change tracking list 
        change_tracking = []
        # go through mapping_dict
        for k,v in self.helperModel.mapping_dict.items():
            dict_temp = {} # columns=["slice_id","pre_AP","post_AP","type_of_change"]
            if k in self.atlasLocation_backup:
                if v == self.atlasLocation_backup[k][2]:
                    dict_temp = {"slice_id":k,
                                    "pre_AP":v,
                                    "post_AP":v,
                                    "type_of_change":"none"}
                    change_tracking.append(dict_temp)
                else:
                    dict_temp = {"slice_id":k,
                                    "pre_AP":self.atlasLocation_backup[k][2],
                                    "post_AP":v,
                                    "type_of_change":"modified"}
                    change_tracking.append(dict_temp)
            else:
                # create according to mapping dict
                dict_temp = {"slice_id":k,
                                "pre_AP":"none",
                                "post_AP":v,
                                "type_of_change":"added"}
                change_tracking.append(dict_temp)

        change_tracking = pd.DataFrame(change_tracking)
        registration_status = []
        for id in change_tracking["slice_id"]:
            if len(self.regViewer.status.atlasDots[id]) > 0:
                registration_status.append("YES")
            else:
                registration_status.append("NO")

        change_tracking["registered"] = registration_status

        # prompt user to solve conflict
        # create a dialog window
        self.confirmation_dialog = QDialog()
        self.confirmation_dialog.setWindowTitle("Confirm or cancel change(s)")
        buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonbox.accepted.connect(self.dialog_accept_action)
        buttonbox.rejected.connect(self.dialog_reject_action)
        selectall_btn = QPushButton("Select All")
        deselectall_btn = QPushButton("Deselect All")
        selectall_btn.clicked.connect(self.checkAll)
        deselectall_btn.clicked.connect(self.uncheckAll)
        dialog_layout = QVBoxLayout()
        select_btn_layout = QHBoxLayout()
        # create QTableView with pandas dataframe
        view = QTableView()
        view.setMinimumSize(800,600)
        view.horizontalHeader().setStretchLastSection(True)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QTableView.SelectRows)
        self.model = PandasModel(change_tracking)
        self.change_tracking = change_tracking

        view.setModel(self.model)

        # create confirmation dialog layout
        dialog_layout.addWidget(view)
        dialog_layout.addLayout(select_btn_layout)
        select_btn_layout.addWidget(selectall_btn) # select all button
        select_btn_layout.addWidget(deselectall_btn) # deselect all button
        select_btn_layout.addWidget(buttonbox)
        self.confirmation_dialog.setLayout(dialog_layout)
        self.confirmation_dialog.exec()

    def checkAll(self):
        for row in range(self.model.rowCount()):
            self.model.setData(self.model.index(row, self.model.columnCount() - 1), Qt.Checked, Qt.CheckStateRole)
    
    def uncheckAll(self):
        for row in range(self.model.rowCount()):
            self.model.setData(self.model.index(row, self.model.columnCount() - 1), Qt.Unchecked, Qt.CheckStateRole)

    def dialog_accept_action(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Apply Change Confirmation")
        msg.setText("Are you sure you want to overwrite selected slice(s) location?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        feedback = msg.exec_()
        if feedback == QMessageBox.Yes:
            self.update_button_availability(status_code=4)
            self.deactivate_preview_mode()
            self.change_tracking["select"] = self.change_tracking.index.map(self.model.check_states)
            # update when select is True, and typeofchange is modified or added
            update_queue = self.change_tracking[(self.change_tracking['select']==True) & 
                                                     ((self.change_tracking['type_of_change']=='modified') | 
                                                      (self.change_tracking['type_of_change']=='added'))].copy()
            # apply changes
            self.regViewer.status.atlasLocation = self.atlasLocation_backup.copy()
            del self.atlasLocation_backup
            for _,row in update_queue.iterrows():
                slice_id,_,post_AP,type_of_change,_,_ = row.tolist()
                if type_of_change == "added": # can be empty angle [empty,empty,AP], when type of change is 'added', use status angle info
                    self.regViewer.status.atlasLocation[slice_id] = [self.regViewer.status.x_angle,
                                                                    self.regViewer.status.y_angle,
                                                                    post_AP]
                else:
                    self.regViewer.status.atlasLocation[slice_id] = [self.regViewer.status.atlasLocation[slice_id][0],
                                                                    self.regViewer.status.atlasLocation[slice_id][1],
                                                                    post_AP]
            # update atlas viewer
            self.regViewer.status.current_z = self.regViewer.status.atlasLocation[
                self.regViewer.status.currentSliceNumber][2]
            self.regViewer.widget.viewerLeft.loadSlice()
            # show transformation overlay
            if self.regViewer.status.currentSliceNumber in self.regViewer.status.blendMode:
                self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] = 1 # overlay
                self.regViewer.atlasModel.updateDotPosition(mode='force')
            self.confirmation_dialog.close()
            # execute save json
            self.regViewer.status.saveRegistration()
        
        else:
            pass # go back to QTableView
    
    def dialog_reject_action(self):
        self.confirmation_dialog.close()
    
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
            
        # status 4: pressed Accept after confirm select, disable all buttons and spinboxes
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
    
    def closeEvent(self, event) -> None:
        if self.preview_mode == 1:
            self.abort_action()
        self.regViewer.del_reghelper_instance()

    







    




