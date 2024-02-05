from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget,QStackedLayout,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QMainWindow

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

        self.btn_ip.pressed.connect(self.activate_ipPage)
        self.btn_re.pressed.connect(self.activate_rePage)

        self.buttonlayout.addWidget(self.btn_ip)
        self.buttonlayout.addWidget(self.btn_re)
        # tab contents
        self.label_ip = QLabel()
        
        self.label_ip.setText("interpolate")
        self.label_re = QLabel()
        self.label_re.setText("editor")
        self.stacklayout.addWidget(self.label_ip)
        self.stacklayout.addWidget(self.label_re)
        self.stacklayout.setCurrentIndex(0)

    def activate_ipPage(self):
        self.stacklayout.setCurrentIndex(0)
    
    def activate_rePage(self):
        self.stacklayout.setCurrentIndex(1)


    




