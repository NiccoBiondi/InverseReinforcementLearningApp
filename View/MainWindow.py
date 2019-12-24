import os 
import sys

from Build.Ui_MainWindow import Ui_MainWindow

from Widget.SetupDialog import SetupDialog
from Widget.AboutDialog import AboutDialog

from Controller.controller import Controller
from Model.model import Model


from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))


# Create the Main window element. Initialize it with a setup layout 
# defined by the Desiger tool.     
class MainWindow(QMainWindow):
    def __init__(self, controller, model):
        super().__init__()
        
        # Set up model and controller
        self._model = model
        self._controller = controller

        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Create about dialog and wire actions.
        self._aboutDialog = AboutDialog()
        self.ui.action_About.triggered.connect(self._aboutDialog.exec_)

        # Create setup dialog and wire actions.
        self._setupModel = SetupDialog(self._model)

        # Connect quit and save action
        self.ui.action_Quit.triggered.connect(QApplication.exit)
        self.ui.action_Save_state.triggered.connect(self._controller.save_action)

        # Define buttons logics
        self.ui.startButton.clicked.connect(lambda : self._controller.start_button())
        self.ui.loadState.clicked.connect(lambda : self._controller.set_loadPath())
        self.ui.resetState.clicked.connect(lambda : self._controller.reset_loadPath())
        self.ui.InitParameters.clicked.connect(lambda : self._setupModel.exec_())

        # Connect signal
        self._model.pathLoadedSignal.connect(self.change_label)
        self._model.changeWindowSignal.connect(self.set_centralWidget)

    # Slot to change the label in Main window starting view
    @pyqtSlot(str)
    def change_label(self, string):
        if string != '':
            # Set the load state
            self.ui.stateLabel.setText(string)
            self.ui.stateLabel.setStyleSheet('color : green')
            
        elif 'FROM' in self.ui.stateLabel.text():
            # Reset the load state  if it is loaded
            self.ui.stateLabel.setText("MODEL NOT LOAD OR INITIALIZE")
            self.ui.stateLabel.setStyleSheet('color : red')

    # Replace the setting view with the AlgView used for the
    # Reinforcement Learning process.   
    @pyqtSlot(object)
    def set_centralWidget(self, widget):
        widget.setLayout(widget.createLayout())
        self.setCentralWidget(widget)
