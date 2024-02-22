from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSpinBox,QLabel,QPushButton,QDoubleSpinBox
from napari_dmc_brainmap.utils import coord_mm_transform
import numpy as np

class AnchorRow(QWidget):
    def __init__(self, regHelper):
        super().__init__()
        self.regHelper = regHelper
        # create horizontal layout
        self.anc_hbox = QHBoxLayout()
        self.setLayout(self.anc_hbox)
        regHelper.anchor_vbox.addWidget(self)
        # add widgets to horizontal layout
        self.spinSliceIndex = QSpinBox()
        self.spinSliceIndex.setMinimum(0)
        self.spinSliceIndex.setMaximum(regHelper.regViewer.status.sliceNum - 1) # total_num
        self.spinSliceIndex.setSingleStep(1)
        self.spinSliceIndex.setValue(regHelper.regViewer.status.currentSliceNumber) # slice_id
        self.spinSliceIndex.valueChanged.connect(self.slice_index_changed)

        self.anc_hbox.addWidget(self.spinSliceIndex)

        self.sliceNameLabel = QLabel()
        self.sliceNameLabel.setText(regHelper.regViewer.status.imgNameDict[
            regHelper.regViewer.status.currentSliceNumber])
        self.anc_hbox.addWidget(self.sliceNameLabel)

        self.spinAPmm = QDoubleSpinBox()
        self.spinAPmm.setDecimals(2)
        self.spinAPmm.setMinimum(-7.80)
        self.spinAPmm.setMaximum(5.40)
        self.spinAPmm.setSingleStep(0.01)
        self.spinAPmm.setValue(regHelper.regViewer.status.current_z) # ap_mm
        self.spinAPmm.setPrefix("AP: ")
        self.spinAPmm.setSuffix(" mm")
        self.spinAPmm.valueChanged.connect(self.ap_mm_changed)

        self.anc_hbox.addWidget(self.spinAPmm)

        self.trash_btn = QPushButton("Trash")
        self.trash_btn.clicked.connect(self.trash_action)
        self.anc_hbox.addWidget(self.trash_btn)
        
        # add anchor(self) to plot
        regHelper.helperModel.add_anchor(self,regHelper.regViewer.status.currentSliceNumber,
                                         regHelper.regViewer.status.current_z)
    
    def trash_action(self):
        # remove self from anchor_vbox layout
        self.regHelper.anchor_vbox.removeWidget(self)
        self.regHelper.helperModel.remove_anchor(self)


    def slice_index_changed(self):
        self.regHelper.regViewer.widget.sampleSlider.setValue(self.spinSliceIndex.value())
        self.sliceNameLabel.setText(self.regHelper.regViewer.status.imgNameDict[
            self.regHelper.regViewer.status.currentSliceNumber])
        self.spinAPmm.setValue(self.regHelper.regViewer.status.current_z)
        # update anchor
        self.regHelper.helperModel.update_anchor()
    
    def ap_mm_changed(self):
        self.regHelper.regViewer.status.current_z = np.round(self.spinAPmm.value(),self.regHelper.regViewer.status.decimal)
        self.regHelper.regViewer.widget.viewerLeft.loadSlice()
        # sync sample slice index
        self.regHelper.regViewer.widget.sampleSlider.setValue(self.spinSliceIndex.value())
        # update anchor
        self.regHelper.helperModel.update_anchor()
        