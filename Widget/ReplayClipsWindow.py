import os
import sys
import cv2
from time import time 
from PIL import Image
import matplotlib.pyplot as plt

from Build.Ui_ReplayClipsWindow import Ui_ReplayClipsWindow
from Widget.DisplayWidget import Display

from PyQt5.QtCore import QTimer, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QDialog

# Simple window used in history window widget to reproduce
# the clips stored in annotation buffer.
class ReplayClipsWindow(QDialog):
    def __init__(self, path):
        super().__init__()

        # Define model and controller
        self._model = ReplayClipsWindowModel(path)
        self._controller = ReplayClipsWindowController(self._model) 

        # Define 
        self._display = Display(self._model.timer, self._model, self._model.replayModelImageSignal)

        self.ui = Ui_ReplayClipsWindow()
        self.ui.setupUi(self)
        self.ui.verticalLayout.addWidget(self._display)

        self.ui.startReplay.clicked.connect(lambda : self._controller.display_figure())

        # Connect model
        self._model.changeVisibilityButton.connect(self.changeVisibility)

    @pyqtSlot(bool)
    def changeVisibility(self, slot):
        self.ui.startReplay.setEnabled(slot)


class ReplayClipsWindowModel(QObject):
    replayModelImageSignal = pyqtSignal(list)
    changeVisibilityButton = pyqtSignal(bool)

    def __init__(self, path):
        super().__init__()

        # Oracle variable 
        self.oracle_active = False

        # Replay button variable
        self._choiceButton = True
        
        # Define Timer and interval period
        self._timer = QTimer()
        self._timer.setInterval(400)
        self._path = path

    @property
    def timer(self):
        return self._timer

    @property
    def choiceButton(self):
        return self._choiceButton

    @choiceButton.setter
    def choiceButton(self, slot):
        self._choiceButton = slot
        self.changeVisibilityButton.emit(slot)

    def load_image(self):
        images = []
        for img in sorted(os.listdir(self._path)):
            if '.png' in img:
                image = Image.open(self._path + '/' + img)
                images.append(image.convert("RGB").resize((800, 800)))
        return images


class ReplayClipsWindowController(QObject):

    def __init__(self, model):
        super().__init__()

        self._model = model

    @pyqtSlot()
    def display_figure(self):
        
        self._model.replayModelImageSignal.emit(self._model.load_image())
        self._model.choiceButton = False
        for i in range(5):
            self._model.timer.start()
        