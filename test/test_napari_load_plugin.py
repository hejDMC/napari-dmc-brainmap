import pytest
from unittest.mock import MagicMock
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from main import GreetingApp


app = QApplication([])


def test_greeting_update(qtbot):
    # Arrange
    window = GreetingApp()
    
    # Use qtbot to add the window to the test
    qtbot.addWidget(window)
    
    # Mock the text input
    window.input_box.setText("Alice")
    
    # Act: simulate clicking the button
    qtbot.mouseClick(window.submit_button, QtCore.Qt.LeftButton)
    
    # Assert: verify the greeting label text is updated as expected
    assert window.greeting_label.text() == "Hello, Alice, welcome"

def test_greeting_update_with_mock():
    # Arrange
    window = GreetingApp()
    
    # Mock the input box's text method to return a predefined value
    window.input_box.text = MagicMock(return_value="Bob")
    
    # Act: trigger the greeting update
    window.update_greeting()
    
    # Assert: check that the QLabel text is as expected
    assert window.greeting_label.text() == "Hello, Bob, welcome"
    window.input_box.text.assert_called_once()
