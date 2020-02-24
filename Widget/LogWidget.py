from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QSizePolicy, QLabel

# Simple widget used to display what the state of the application
# and the current user choices
class LogWidget(QLabel):
    
    def __init__(self, signal, text="Welcome to the RL mingrid app. Press 'process' to start the training",**kwargs):
        super().__init__(**kwargs)

        self.setText(text)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        # Connect to model
        signal.connect(self.set_text)

    @pyqtSlot(str)
    def set_text(self, text):
        self.setText(text)
