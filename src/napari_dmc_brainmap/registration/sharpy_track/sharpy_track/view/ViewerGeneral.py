from PyQt5.QtWidgets import QLabel,QGraphicsScene,QFrame, QGraphicsItemGroup, QGraphicsPixmapItem, QGraphicsEllipseItem
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtCore import Qt
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.QGraphicsViewerMT import QGraphicsViewMT
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import mapPointTransform
import numpy as np

class ViewerGeneral():
    def __init__(self,regViewer) -> None:
        self.regViewer = regViewer
        self.labelImg = QLabel()
        self.labelImg.setFixedSize(regViewer.singleWindowSize[0],regViewer.singleWindowSize[1])
        self.scene = QGraphicsScene(0,0,regViewer.singleWindowSize[0],regViewer.singleWindowSize[1],parent=regViewer)
        self.scene.addWidget(self.labelImg)
        self.itemGroup = [] # create itemGroup, store DotObjects
        self.view = QGraphicsViewMT(self.scene) # QGraphicsView with mousetracking
        self.view.setFixedSize(regViewer.singleWindowSize[0],regViewer.singleWindowSize[1])
        self.view.setSceneRect(0,0,regViewer.singleWindowSize[0],regViewer.singleWindowSize[1])
        self.view.setFrameShape(QFrame.NoFrame)
        # TRE variables
        # project source position to target position using current transformation matrix
        self.tform = None
        self.targetPointHover = QGraphicsItemGroup()  # container for dynamically projected point

    def leaveLabel(self):
        """Slot for mouse leave signal"""
        self.regViewer.status.cursor = 0
    
    # to be connected with mouseMoved signal on the left viewer
    def getCursorPos(self):
        if self.regViewer.status.contour == 1: # only when contour active, update in status
            # cursor position within boundary check
            if (self.view.cursorPos[0] in self.regViewer.res_x_range) and (
                self.view.cursorPos[1] in self.regViewer.res_y_range):
                self.regViewer.status.hoverX = self.regViewer.res_up[self.view.cursorPos[0]] # save hover position to status, if single window size different from 1140*800 scale coordinates
                self.regViewer.status.hoverY = self.regViewer.res_up[self.view.cursorPos[1]]
                self.regViewer.atlasModel.treeFindArea()
                
            else:
                pass
        else:
            pass
    

    def projectSourcePos(self):
        x_src, y_src = self.regViewer.res_up[int(self.view.cursorPos[0])], self.regViewer.res_up[int(self.view.cursorPos[1])]
        x_target, y_target = mapPointTransform(x_src, y_src, self.tform)
        # round
        x_target, y_target = np.round(x_target).astype(int), np.round(y_target).astype(int)
        # within boundary check
        if any([
                x_target < 0,
                x_target >= self.regViewer.atlas_resolution[0],
                y_target < 0,
                y_target >= self.regViewer.atlas_resolution[1]
                ]):
            # target point out of boundary, indicate with red cursor on the right viewer
            self.regViewer.widget.viewerRight.view.viewport().setCursor(self.regViewer.measurementPage.cursor_r_64)
            # clear target point marker
            if self.targetPointHover.childItems():
                self.clearTargetPointHover()
            # PREVENT THE CREATION OF SOURCE DOT
        else:
            # switch cursor to yellow pointer
            self.regViewer.widget.viewerRight.view.viewport().setCursor(self.regViewer.measurementPage.cursor_y_64)
            # check if there is already a pixmap item in the list
            if not self.targetPointHover.childItems():
                # create new pixmap item
                item = QGraphicsPixmapItem(self.regViewer.measurementPage.pixmap_y_32)
                self.targetPointHover.addToGroup(item)
            self.targetPointHover.childItems()[0].setPos(self.regViewer.res_down[x_target] - 16, self.regViewer.res_down[y_target] - 16)

        # Update live labels if a row is active
        row = getattr(self.regViewer.measurementPage, 'unset_tre_row', None)
        if row is not None:
            row.source_pos_label.setText(f"({x_src}, {y_src})")
            if self.targetPointHover.childItems():
                row.target_pos_label.setText(f"({x_target}, {y_target})")
                # store XY coordinates of both source and target to regViewer.measurementPage
                self.regViewer.measurementPage.unset_source_pos = (x_src, y_src)
                self.regViewer.measurementPage.unset_target_pos = (x_target, y_target)
            else:
                row.target_pos_label.setText("[Out of Bounds]")


    def clearTargetPointHover(self):
        self.regViewer.widget.viewerLeft.scene.removeItem(self.targetPointHover.childItems()[0])


    # click handler to finalize source point when target projection is valid
    def handleSourceClick(self):
        # require a valid projected target marker present
        if not self.targetPointHover.childItems():
            return
        # retrieve source and target position from regViewer.measurementPage
        x_src, y_src = self.regViewer.measurementPage.unset_source_pos
        # add source dot on right viewer
        self.addSourceDot(self.regViewer.res_down[x_src], self.regViewer.res_down[y_src])
        # save to active_rows dictionary
        x_target, y_target = self.regViewer.measurementPage.unset_target_pos
        self.regViewer.measurementPage.active_rows["source_coords"].append([int(x_src), int(y_src)])
        self.regViewer.measurementPage.active_rows["target_coords"].append([int(x_target), int(y_target)])
        self.regViewer.measurementPage.active_rows["tform_matrix"] = self.tform.tolist()
        # update button text and style
        self.regViewer.measurementPage.ui.addMeasurementBtn.setText("Place Marker on Atlas")
        self.regViewer.measurementPage.ui.addMeasurementBtn.setStyleSheet("background-color: rgb(255, 140, 0);") # dark orange
        # disconnect mouse enter and left signal
        self.regViewer.widget.viewerRight.view.mouseEntered.disconnect(self.regViewer.measurementPage.show_measurement_pointer)
        self.regViewer.widget.viewerRight.view.mouseLeft.disconnect(self.regViewer.measurementPage.hide_measurement_pointer)
        # hide measurement pointer
        self.regViewer.measurementPage.hide_measurement_pointer(discard_row=False)
        # update measurement state
        self.regViewer.measurementPage.measurement_state = "waiting_truth"
        # continue at measurementPage.display_truth_pointer
        self.regViewer.measurementPage.display_truth_pointer()


    def addSourceDot(self, x: int, y: int, diameter: int = 8) -> None:
        ellipse = QGraphicsEllipseItem(0, 0, diameter, diameter)
        ellipse.setBrush(QColor(255, 140, 0))  # solid dark orange
        ellipse.setPen(QPen(Qt.NoPen))
        ellipse.setPos(x - diameter // 2, y - diameter // 2)
        # save to active_rows dictionary
        self.regViewer.measurementPage.active_rows["source_obj"].append(ellipse)
        # add to scene
        self.scene.addItem(self.regViewer.measurementPage.active_rows["source_obj"][-1])
    
    def update_true_pos(self):
        x_truth, y_truth = self.regViewer.res_up[int(self.view.cursorPos[0])], self.regViewer.res_up[int(self.view.cursorPos[1])]
        # save to measurementPage.unset_truth_pos
        self.regViewer.measurementPage.unset_truth_pos = (x_truth, y_truth)
        # update true_pos field in TreRow
        self.regViewer.measurementPage.active_rows["row_obj"][-1].true_pos_label.setText(f"({x_truth}, {y_truth})")
    
    def handleTruthClick(self):
        # disconnect signals
        self.regViewer.widget.viewerLeft.view.mouseMoved.disconnect(self.regViewer.widget.viewerLeft.update_true_pos)
        self.regViewer.widget.viewerLeft.view.mouseClicked.disconnect(self.regViewer.widget.viewerLeft.handleTruthClick)
        self.regViewer.widget.viewerLeft.view.mouseEntered.disconnect(self.regViewer.measurementPage.show_truth_pointer)
        self.regViewer.widget.viewerLeft.view.mouseLeft.disconnect(self.regViewer.measurementPage.hide_truth_pointer)

        self.regViewer.measurementPage.active_rows["source_obj"][-1].setBrush(QColor(255, 140, 0))
        self.regViewer.widget.viewerLeft.view.viewport().setCursor(Qt.ArrowCursor)
        
        x_truth, y_truth = self.regViewer.measurementPage.unset_truth_pos
        self.addTruthDot(self.regViewer.res_down[x_truth], self.regViewer.res_down[y_truth])
        # save to active_rows dictionary
        self.regViewer.measurementPage.active_rows["truth_coords"].append([int(x_truth), int(y_truth)])

        # update button text and style
        self.regViewer.measurementPage.ui.addMeasurementBtn.setText("Add Measurement")
        self.regViewer.measurementPage.ui.addMeasurementBtn.setStyleSheet("background-color: rgb(0, 255, 0);") 
        self.regViewer.measurementPage.measurement_state = "ready"
        # calculate TRE, save TRE to measurementPage.active_rows
        # TRE = sqrt((x_truth - x_target)**2 + (y_truth - y_target)**2)
        x_target, y_target = self.regViewer.measurementPage.active_rows["target_coords"][-1]
        TRE = np.sqrt((x_truth - x_target)**2 + (y_truth - y_target)**2)
        self.regViewer.measurementPage.active_rows["row_obj"][-1].tre_label.setText(f"{TRE:.2f}")
        # update TRE label
        self.regViewer.measurementPage.active_rows["tre_score"].append(np.round(TRE, 4).astype(str))
        # configure and enable delete button for current row object
        self.regViewer.measurementPage.active_rows["row_obj"][-1].connect_delete_btn()
        
        self.regViewer.measurementPage.active_rows["row_obj"][-1].remove_btn.setEnabled(True)
        
        for row in self.regViewer.measurementPage.active_rows["row_obj"]:
            row.setMouseTracking(True)
        


    
    def addTruthDot(self, x: int, y: int, diameter: int = 8) -> None:
        ellipse = QGraphicsEllipseItem(0, 0, diameter, diameter)
        ellipse.setBrush(QColor(255, 140, 0))  # solid red
        ellipse.setPen(QPen(Qt.NoPen))
        ellipse.setPos(x - diameter // 2, y - diameter // 2)
        # save to active_rows dictionary
        self.regViewer.measurementPage.active_rows["truth_obj"].append(ellipse)
        # add to scene
        self.scene.addItem(self.regViewer.measurementPage.active_rows["truth_obj"][-1])

