import os 
import sys

from View.MainWindow import MainWindow

from Controller.controller import Controller
from Model.model import Model


from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))


app = QApplication(sys.argv)
model = Model()
controller = Controller(model)
window = MainWindow(controller, model)
window.show()
app.exec_()
sys.exit()
