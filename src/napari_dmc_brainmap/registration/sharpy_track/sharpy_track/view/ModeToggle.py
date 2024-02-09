from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect


class ModeToggle(QtWidgets.QPushButton):
    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setMinimumWidth(44)
        self.setMinimumHeight(132)

    def paintEvent(self, event):
        if self.isChecked():
            label = "I" # transformation ON
            bg_color = Qt.green
        else:
            if self.isEnabled():
                label = "O" # transformation OFF, non-preview
                bg_color = Qt.red
            else:
                label = "P" # transformation OFF, preview
                bg_color = Qt.blue

        radius = 20
        height = 64
        center = self.rect().center()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QtGui.QColor(0,0,0))

        pen = QtGui.QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(QRect(-radius, -height, 2*radius, 2*height), radius, radius)
        painter.setBrush(QtGui.QBrush(bg_color))
        sw_rect = QRect(-radius, -height, 2*radius, height)
        if not self.isChecked():
            sw_rect.moveBottom(height)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.drawText(sw_rect, Qt.AlignCenter, label)


