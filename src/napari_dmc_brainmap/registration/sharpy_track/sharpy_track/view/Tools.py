from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QColor
from PyQt5.QtWidgets import QWidget,QStackedLayout,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QMainWindow,QMessageBox,QTableView,QDialog,QDialogButtonBox
from PyQt5 import QtGui
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.AnchorRow import AnchorRow
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.TreRow import TreRow
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.ui.AccuracyMeasurement import Ui_AccuracyMeasurement
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.HelperModel import HelperModel
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.PandasModel import PandasModel
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.MeasurementHandler import MeasurementHandler
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans
import pandas as pd

class ShortcutsInfo(QMainWindow):
    def __init__(self, regViewer) -> None:
        super().__init__()
        self.regViewer = regViewer
        self.setWindowTitle("Keyboard Shortcuts")
        self.setFixedSize(int(regViewer.fullWindowSize[0]/1.5),regViewer.fullWindowSize[1])
        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)
        
        # Create shortcuts label
        self.shortcuts_label = QLabel()
        self.shortcuts_label.setPixmap(QPixmap(str(
            self.regViewer.atlasModel.sharpy_dir.joinpath(
                'sharpy_track',
                'sharpy_track',
                'images',
                'keyboard_shortcuts.png'))))
        
        # Optional: Adjust the QLabel size to fit the image
        self.shortcuts_label.setScaledContents(True)
        self.shortcuts_label.resize(self.shortcuts_label.pixmap().size())
        
        self.mainLayout.addWidget(self.shortcuts_label)
    
    def closeEvent(self, event) -> None:
        self.regViewer.del_shortcuts_instance()


class InterpolatePosition(QMainWindow):
    def __init__(self, regViewer) -> None:
        super().__init__()
        self.regViewer = regViewer
        self.helperModel = HelperModel(regViewer)
        self.setWindowTitle("Interpolate Position")
        self.setFixedSize(int(regViewer.fullWindowSize[0]/1.5),regViewer.fullWindowSize[1])
        self.mainWidget = QWidget()
        # setup layout
        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)
        
        # location plot hbox
        self.locplot_hbox = QHBoxLayout()
        self.mainLayout.addLayout(self.locplot_hbox)
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
            # connect tranformation toggle changed signal to update button availability
        self.regViewer.widget.toggle.clicked.connect(self.toggle_changed_callback)
        self.toggle_changed_callback()

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
        self.mainLayout.addWidget(self.anchor_widget)

        self.preview_mode = 0

    def toggle_changed_callback(self):
        self.update_button_availability(status_code=5)

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
        # check if AccuracyMeasurement window is open
        if hasattr(self.regViewer, 'measurementPage'):
            # connect preview button press signal to AccuracyMeasurement flip_page
            self.regViewer.measurementPage.flip_page()


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

        # check if AccuracyMeasurement window is open
        if hasattr(self.regViewer, 'measurementPage'):
            # connect preview button press signal to AccuracyMeasurement flip_page
            self.regViewer.measurementPage.flip_page()


    
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

        # check if AccuracyMeasurement window is open
        if hasattr(self.regViewer, 'measurementPage'):
            # connect preview button press signal to AccuracyMeasurement flip_page
            self.regViewer.measurementPage.flip_page()

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

        # status 5: tranformation toggle turned, disable/ enable add button
        elif status_code == 5:
            if self.regViewer.widget.toggle.isChecked(): # off ==> on
                self.add_btn.setDisabled(True)
                self.preview_btn.setDisabled(True)
            else: # on ==> off
                if self.helperModel.active_anchor:
                    if self.helperModel.active_anchor[0].trash_btn.isEnabled():
                        self.add_btn.setEnabled(True)
                        self.update_button_availability(status_code=1)
                    else:
                        pass
                else:
                    self.add_btn.setEnabled(True)
                    self.preview_btn.setDisabled(True)

        else:
            print("Warning: button availability updated without specified status code!")
    
    def activate_preview_mode(self):
        self.preview_mode = 1
        self.regViewer.widget.x_slider.setDisabled(True)
        self.regViewer.widget.y_slider.setDisabled(True)
        self.regViewer.widget.z_slider.setDisabled(True)
        self.regViewer.widget.toggle.setDisabled(True)
    
    def deactivate_preview_mode(self):
        self.preview_mode = 0
        self.regViewer.widget.x_slider.setEnabled(True)
        self.regViewer.widget.y_slider.setEnabled(True)
        self.regViewer.widget.z_slider.setEnabled(True)
        self.regViewer.widget.toggle.setEnabled(True)
    
    def closeEvent(self, event) -> None:
        if self.preview_mode == 1:
            self.abort_action()
        self.regViewer.widget.toggle.clicked.disconnect(self.toggle_changed_callback)
        self.regViewer.del_interpolatePosition_instance()


class AccuracyMeasurement(QMainWindow):
    def __init__(self, regViewer) -> None:
        super().__init__()
        self.regViewer = regViewer
        self.setWindowTitle("Accuracy Measurement (TRE)")
        self.setFixedSize(int(regViewer.fullWindowSize[0]/3),regViewer.fullWindowSize[1])
        self.ui = Ui_AccuracyMeasurement()
        self.ui.setupUi(self)
        # widget specific state variables
        self.measurement_state = "ready"
        self.unset_tre_row = None
        self.unset_source_pos = None
        self.unset_target_pos = None
        self.unset_truth_pos = None
        self.active_rows = dict(source_coords=[],
                                target_coords=[],
                                source_obj=[],
                                row_obj=[],
                                truth_obj=[],
                                truth_coords=[],
                                tre_score=[],
                                tform_matrix=None,
                                imgIndex=None)
        self.measurement_handler = MeasurementHandler(self)
        # retrieve current file name
        self.ui.currentFileNameLabel.setText(
            self.regViewer.status.imgFileName[
            self.regViewer.status.currentSliceNumber])
        ## connect signals
        ## add measurement button signal
        self.ui.addMeasurementBtn.clicked.connect(self.modify_measurement)


    def update_name_label(self):
        self.ui.currentFileNameLabel.setText(
            self.regViewer.status.imgFileName[
            self.regViewer.status.currentSliceNumber])
        self.flip_page()
    

    def flip_page(self):
        # retrieve registration mode information
        registration_on = True if self.regViewer.status.tMode == 1 else False
        preview_on = False if self.regViewer.widget.toggle.isEnabled() else True
        if registration_on or preview_on:
            self.ui.pages.setCurrentIndex(1)
        else:
            if self.regViewer.status.currentSliceNumber not in self.regViewer.status.atlasDots:
                self.ui.pages.setCurrentIndex(2)
                self.measurement_handler.save_measurement_record()
            else:
                if len(self.regViewer.status.atlasDots[self.regViewer.status.currentSliceNumber]) < 5:
                    self.ui.pages.setCurrentIndex(2)
                    self.measurement_handler.save_measurement_record()
                else:
                    self.ui.pages.setCurrentIndex(0)
                    self.measurement_handler.save_measurement_record()
                    self.measurement_handler.load_measurement_record()

    def modify_measurement(self):
        # check status of measurement
        if self.measurement_state == "ready":
            # Disable hover selection for all existing rows when starting new measurement
            for row in self.active_rows["row_obj"]:
                row.setMouseTracking(False)
                # Reset any hover styles to normal
                if row.is_hovered:
                    row.apply_normal_style()
                    row.is_hovered = False
                    
            self.ui.addMeasurementBtn.setText("Place Marker on Sample")
            self.ui.addMeasurementBtn.setStyleSheet("background-color: rgb(255, 222, 33);") # yellow
            # enable pointer projection visualization
            self.display_target_projection()
            # setup abort action callback
            self.setup_abort_callback()

        elif any([self.measurement_state == "abort",
                  self.measurement_state == "waiting_source",
                  self.measurement_state == "waiting_truth"]):
            self.detach_abort_callback()
            if self.regViewer.widget.viewerRight.targetPointHover.childItems():
                self.hide_measurement_pointer()
            try:
                self.regViewer.widget.viewerRight.view.mouseEntered.disconnect(self.show_measurement_pointer)
                self.regViewer.widget.viewerRight.view.mouseLeft.disconnect(self.hide_measurement_pointer)
            except TypeError:
                pass
            # remove any unset row reference in case of sudden exit
            if self.unset_tre_row is not None:
                self.unset_tre_row.remove_unset_row()
                self.unset_tre_row = None
            self.ui.addMeasurementBtn.setText("Add Measurement")
            self.ui.addMeasurementBtn.setStyleSheet("background-color: rgb(0, 255, 0);") # green
            self.measurement_state = "ready"
            
            # Enable hover selection for all existing rows when returning to ready state
            for row in self.active_rows["row_obj"]:
                row.setMouseTracking(True)
        else:
            raise ValueError("Unknown measurement state: "+self.measurement_state)
            

    def display_target_projection(self):
        # initialize cursor
        pointer_image_y = QPixmap(str(
            self.regViewer.atlasModel.sharpy_dir.joinpath(
                'sharpy_track',
                'sharpy_track',
                'images',
                'crosshair_yellow.png')))
        self.cursor_y_64 = QtGui.QCursor(pointer_image_y, 32, 32) # 64 * 64 image cursor hotspot set to [32,32]

        pointer_image_truth = QPixmap(str(
            self.regViewer.atlasModel.sharpy_dir.joinpath(
                'sharpy_track',
                'sharpy_track',
                'images',
                'crosshair_red.png')))
        self.cursor_truth_64 = QtGui.QCursor(pointer_image_truth, 32, 32) # 64 * 64 image cursor hotspot set to [32,32]

        # initialize cursor
        pointer_image_r = QPixmap(str(
            self.regViewer.atlasModel.sharpy_dir.joinpath(
                'sharpy_track',
                'sharpy_track',
                'images',
                'error_red.png')))
        self.cursor_r_64 = QtGui.QCursor(pointer_image_r, 32, 32) # 64 * 64 image cursor hotspot set to [32,32]

        # initialize icon pixmap
        pointer_image_y_s = QPixmap(str(
            self.regViewer.atlasModel.sharpy_dir.joinpath(
                'sharpy_track',
                'sharpy_track',
                'images',
                'crosshair_yellow_s.png')))
        self.pixmap_y_32 = pointer_image_y_s

        self.regViewer.widget.viewerRight.view.mouseEntered.connect(self.show_measurement_pointer)
        self.regViewer.widget.viewerRight.view.mouseLeft.connect(self.hide_measurement_pointer)
        # update measurement state
        self.measurement_state = "waiting_source"

    def show_measurement_pointer(self):
        # Show the measurement pointer
        self.regViewer.widget.viewerRight.view.viewport().setCursor(self.cursor_y_64)
        # create TreRow upon entering viewerRight in measurement mode
        assert self.measurement_state == "waiting_source" and self.unset_tre_row is None
        self.unset_tre_row = self.create_new_row()

        # enable position tracking
        # project source position to target position using current transformation matrix
        self.regViewer.widget.viewerRight.tform = fitGeoTrans(self.regViewer.status.sampleDots[self.regViewer.status.currentSliceNumber], 
                                                              self.regViewer.status.atlasDots[self.regViewer.status.currentSliceNumber])
        # add targetPointHover group to scene
        self.regViewer.widget.viewerLeft.scene.addItem(self.regViewer.widget.viewerRight.targetPointHover)
        self.regViewer.widget.viewerRight.view.mouseMoved.connect(self.regViewer.widget.viewerRight.projectSourcePos)
        self.regViewer.widget.viewerRight.view.mouseClicked.connect(self.regViewer.widget.viewerRight.handleSourceClick)

    def hide_measurement_pointer(self, discard_row=True):
        # Hide the measurement pointer
        self.regViewer.widget.viewerRight.view.viewport().setCursor(Qt.ArrowCursor)
        # disable position tracking
        self.regViewer.widget.viewerRight.view.mouseMoved.disconnect(self.regViewer.widget.viewerRight.projectSourcePos)
        self.regViewer.widget.viewerRight.view.mouseClicked.disconnect(self.regViewer.widget.viewerRight.handleSourceClick)
        # clear tform and targetPointHover
        self.regViewer.widget.viewerRight.tform = None
        for item in self.regViewer.widget.viewerRight.targetPointHover.childItems():
            self.regViewer.widget.viewerRight.targetPointHover.removeFromGroup(item)
        self.regViewer.widget.viewerLeft.scene.removeItem(self.regViewer.widget.viewerRight.targetPointHover)
        # remove TreRow
        if discard_row: 
            if self.unset_tre_row is not None:
                self.unset_tre_row.remove_unset_row()
                self.unset_tre_row = None
        # register TreRow
        else:
            self.active_rows["row_obj"].append(self.unset_tre_row)
            self.unset_tre_row = None
    
    def display_truth_pointer(self):
        self.regViewer.widget.viewerLeft.view.mouseEntered.connect(self.show_truth_pointer)
        self.regViewer.widget.viewerLeft.view.mouseLeft.connect(self.hide_truth_pointer)
        self.regViewer.widget.viewerLeft.view.mouseClicked.connect(self.regViewer.widget.viewerLeft.handleTruthClick)

    def show_truth_pointer(self):
        self.regViewer.widget.viewerLeft.view.viewport().setCursor(self.cursor_truth_64)
        # color source dot to red
        self.active_rows["source_obj"][-1].setBrush(QColor(255, 0, 0))
        # connect mousemoved signal to update true_pos in row object
        self.regViewer.widget.viewerLeft.view.mouseMoved.connect(self.regViewer.widget.viewerLeft.update_true_pos)

    def hide_truth_pointer(self):
        self.regViewer.widget.viewerLeft.view.mouseMoved.disconnect(self.regViewer.widget.viewerLeft.update_true_pos)
        # restore source dot color
        self.active_rows["source_obj"][-1].setBrush(QColor(255, 140, 0))
        # restore true_pos field to [?]
        self.active_rows["row_obj"][-1].true_pos_label.setText("[?]")
        # restore cursor
        self.regViewer.widget.viewerLeft.view.viewport().setCursor(Qt.ArrowCursor)
        # clear true_pos field in TreRow
    
    def setup_abort_callback(self):
        self.regViewer.widget.sampleSlider.valueChanged.connect(self.abort_action)
        self.regViewer.widget.x_slider.valueChanged.connect(self.abort_action)
        self.regViewer.widget.y_slider.valueChanged.connect(self.abort_action)
        self.regViewer.widget.z_slider.valueChanged.connect(self.abort_action)
        self.regViewer.widget.toggle.clicked.connect(self.abort_action)


    def detach_abort_callback(self):
        for sig in [
            self.regViewer.widget.sampleSlider.valueChanged,
            self.regViewer.widget.x_slider.valueChanged,
            self.regViewer.widget.y_slider.valueChanged,
            self.regViewer.widget.z_slider.valueChanged,
            self.regViewer.widget.toggle.clicked
        ]:
            try:
                sig.disconnect(self.abort_action)
            except TypeError:
                pass

    def abort_action(self):
        self.measurement_state = "abort"
        self.modify_measurement()
    
    def create_new_row(self):
        return TreRow(self)

    # forward all keyPressEvent to regViewer
    def keyPressEvent(self, event):
        # Forward all key press events to the main registration viewer
        self.regViewer.keyPressEvent(event)

    def closeEvent(self, event) -> None:
        # disconnect signals
        self.regViewer.widget.sampleSlider.valueChanged.disconnect(self.update_name_label)
        self.regViewer.widget.toggle.clicked.disconnect(self.flip_page)
        if self.measurement_state == "waiting_source":
            self.abort_action()
        if self.ui.pages.currentIndex() == 0:
            self.measurement_handler.save_measurement_record()
        self.regViewer.del_measurement_instance()
    