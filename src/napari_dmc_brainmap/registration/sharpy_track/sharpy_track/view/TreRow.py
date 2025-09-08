from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

class TreRow(QWidget):
    def __init__(self, measurementPage):
        super().__init__()
        self.measurementPage = measurementPage
        # create horizontal layout
        self.row_hbox = QHBoxLayout()
        self.setLayout(self.row_hbox)
        # inside ui.coordsDataVBox
        self.measurementPage.ui.coordsDataVBox.addWidget(self)
        self.source_pos_label = QLabel("[?]")
        self.target_pos_label = QLabel("[?]")
        self.true_pos_label = QLabel("[?]")
        self.tre_label = QLabel("[?]")
        self.row_hbox.addWidget(self.source_pos_label)
        self.row_hbox.addWidget(self.target_pos_label)
        self.row_hbox.addWidget(self.true_pos_label)
        self.row_hbox.addWidget(self.tre_label)
        # add delete button
        self.remove_btn = QPushButton("Delete")
        # disabled for now
        self.remove_btn.setEnabled(False)
        self.row_hbox.addWidget(self.remove_btn)
        
        # Enable mouse tracking for hover events
        self.setMouseTracking(True)
        self.is_hovered = False

    def enterEvent(self, event):
        """Handle mouse enter event for hover selection"""
        if self.measurementPage.measurement_state == "ready":
            self.is_hovered = True
            self.apply_hover_style()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave event for hover selection"""
        if self.measurementPage.measurement_state == "ready":
            self.is_hovered = False
            self.apply_normal_style()
        super().leaveEvent(event)

    def apply_hover_style(self):
        """Apply hover style - light orange background and red dots"""
        # Set light orange background for the row
        self.setStyleSheet("background-color: rgb(255, 200, 150);")
        
        # Find the index of this row in the active_rows
        if self in self.measurementPage.active_rows["row_obj"]:
            idx = self.measurementPage.active_rows["row_obj"].index(self)
            # Change source dot color to red
            self.measurementPage.active_rows["source_obj"][idx].setBrush(QColor(255, 0, 0))
            # Change truth dot color to red
            self.measurementPage.active_rows["truth_obj"][idx].setBrush(QColor(255, 0, 0))

    def apply_normal_style(self):
        """Apply normal style - default background and orange dots"""
        # Reset background to default
        self.setStyleSheet("")
        
        # Find the index of this row in the active_rows
        if self in self.measurementPage.active_rows["row_obj"]:
            idx = self.measurementPage.active_rows["row_obj"].index(self)
            # Change source dot color back to orange
            self.measurementPage.active_rows["source_obj"][idx].setBrush(QColor(255, 140, 0))
            # Change truth dot color back to orange
            self.measurementPage.active_rows["truth_obj"][idx].setBrush(QColor(255, 140, 0))

    def remove_unset_row(self):
        self.measurementPage.ui.coordsDataVBox.removeWidget(self)
        self.deleteLater()
    
    def remove_registered_row(self):
        self.measurementPage.ui.coordsDataVBox.removeWidget(self)
        idx_del = self.measurementPage.active_rows["row_obj"].index(self)
        self.measurementPage.active_rows["source_coords"].pop(idx_del)
        self.measurementPage.active_rows["target_coords"].pop(idx_del)
        self.measurementPage.active_rows["row_obj"].pop(idx_del)
        self.measurementPage.active_rows["truth_coords"].pop(idx_del)
        self.measurementPage.active_rows["tre_score"].pop(idx_del)
        # remove dot objects from scene
        self.measurementPage.regViewer.widget.viewerRight.scene.removeItem(self.measurementPage.active_rows["source_obj"][idx_del])
        self.measurementPage.regViewer.widget.viewerLeft.scene.removeItem(self.measurementPage.active_rows["truth_obj"][idx_del])
        # remove dot objects from list
        self.measurementPage.active_rows["source_obj"].pop(idx_del)
        self.measurementPage.active_rows["truth_obj"].pop(idx_del)
        self.deleteLater()

    def connect_delete_btn(self):
        self.remove_btn.clicked.connect(self.remove_registered_row)