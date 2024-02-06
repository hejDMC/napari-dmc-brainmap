from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget,QStackedLayout,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QMainWindow,QComboBox
from PyQt5 import QtGui
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.AnchorRow import AnchorRow

class RegistrationHelper(QMainWindow):
    def __init__(self, regViewer) -> None:
        super().__init__()
        self.regViewer = regViewer
        self.setWindowTitle("Registration Helper")
        self.setFixedSize(int(regViewer.status.fullWindowSize[0]/2),regViewer.status.fullWindowSize[1])
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
        self.previewadd_vbox.addWidget(self.preview_btn)
            # add button
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_action)
        self.previewadd_vbox.addWidget(self.add_btn)
            # section location illustration
        self.preview_label = QLabel()
        # insert demo image, replace with real time rendering in the future
        demo_img_path = "C:\\Users\\xiao\\GitHub\\napari-dmc-brainmap\\src\\napari_dmc_brainmap\\registration\\sharpy_track\\sharpy_track\\images\\locationplot.png" 
        with open(demo_img_path,'rb') as f:
            demo_img_bin = f.read()
        demo_img = QtGui.QImage()
        demo_img.loadFromData(demo_img_bin)
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(demo_img))
        # self.preview_label.setFixedSize(750,300)

        self.locplot_hbox.addWidget(self.preview_label)
            # abort and apply buttons in a vbox
        self.abort_apply_vbox = QVBoxLayout()
        self.locplot_hbox.addLayout(self.abort_apply_vbox)
        self.abort_btn = QPushButton("Abort")
        self.apply_btn = QPushButton("Apply")
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

    def activate_ipPage(self):
        self.stacklayout.setCurrentIndex(0)
    
    def activate_rePage(self):
        self.stacklayout.setCurrentIndex(1)
    
    def add_action(self):
        # create anchor object
        AnchorRow(self)







    




