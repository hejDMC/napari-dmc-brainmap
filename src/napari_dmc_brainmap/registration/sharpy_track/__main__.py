from sharpy_track.view.RegistrationViewer import RegistrationViewer
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    RegViewer = RegistrationViewer(app)
    RegViewer.show()
    sys.exit(app.exec_())