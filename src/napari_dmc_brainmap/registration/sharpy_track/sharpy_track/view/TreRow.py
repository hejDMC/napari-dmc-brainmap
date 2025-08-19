from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton

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
        self.remove_btn.clicked.connect(self.remove_row)
        self.row_hbox.addWidget(self.remove_btn)

    def remove_row(self):
        self.measurementPage.ui.coordsDataVBox.removeWidget(self)
        self.deleteLater()