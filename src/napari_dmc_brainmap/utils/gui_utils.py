from qtpy.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QProgressBar
from pathlib import Path

class ProgressBar(QWidget):
    """
    A QWidget subclass representing a progress bar.
    """

    def __init__(self, parent: QWidget = None):
        """
        Initialize the ProgressBar widget.

        Parameters:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def set_value(self, value: int):
        """
        Set the current value of the progress bar.

        Parameters:
            value (int): The value to set (between the minimum and maximum values).
        """
        self.progress_bar.setValue(value)


def check_input_path(input_path: Path) -> bool:
    """
    Validate the input path to ensure it is a valid directory.

    If the path is not valid, a critical error message box is displayed.

    Parameters:
        input_path (Path): The input path to validate.

    Returns:
        bool: True if the input path is valid, False otherwise.
    """
    if not input_path.is_dir() or str(input_path) == '.':
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(
            f"Input path is not a valid directory. Please make sure this exists: >> '{str(input_path)}' <<"
        )
        msg_box.setWindowTitle("Invalid Path Error")
        msg_box.exec_()  # Show the message box
        return False
    else:
        return True
