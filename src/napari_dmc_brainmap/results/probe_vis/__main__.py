import sys
from PyQt5.QtWidgets import QApplication
from probe_vis.view.ProbeVisualizer import ProbeVisualizer

# sys.path.append("C:\\Users\\xiao\\GitHub\\probe4Xiao\\probe_vis")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    probeV = ProbeVisualizer()
    probeV.show()
    sys.exit(app.exec_())