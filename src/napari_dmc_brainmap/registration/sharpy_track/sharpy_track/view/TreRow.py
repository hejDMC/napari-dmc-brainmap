from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
import random

class TreRow(QWidget):
    def __init__(self, measurementPage):
        super().__init__()
        self.measurementPage = measurementPage
        # create horizontal layout
        self.row_hbox = QHBoxLayout()
        self.setLayout(self.row_hbox)
        # inside ui.coordsDataVBox
        self.measurementPage.ui.coordsDataVBox.addWidget(self)
        self.source_pos_label = QLabel(str(random.randint(0, 100)))
        self.target_pos_label = QLabel(str(random.randint(0, 100)))
        self.true_pos_label = QLabel(str(random.randint(0, 100)))
        self.tre_label = QLabel(str(random.randint(0, 100)))
        self.row_hbox.addWidget(self.source_pos_label)
        self.row_hbox.addWidget(self.target_pos_label)
        self.row_hbox.addWidget(self.true_pos_label)
        self.row_hbox.addWidget(self.tre_label)
        # add delete button
        self.remove_btn = QPushButton("Delete")
        self.remove_btn.clicked.connect(self.trash_action)
        self.row_hbox.addWidget(self.remove_btn)

    def trash_action(self):
        # remove self from ui.coordsDataVBox
        self.measurementPage.ui.coordsDataVBox.removeWidget(self)
        # remove related data in model