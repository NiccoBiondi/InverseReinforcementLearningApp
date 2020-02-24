import os
import sys

from Build.Ui_Dialog import Ui_Dialog

from PyQt5.QtWidgets import QDialog

# Pop up that is called when the user click on the About button 
class AboutDialog(QDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up the user interface from Designer.
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Connect Ok Button
        self.ui.buttonBox.clicked.connect(self.close)