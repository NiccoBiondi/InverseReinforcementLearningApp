import os
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from Build.Ui_ReplayClipsWindow import Ui_ReplayClipsWindow
from Widget.DisplayWidget import Display

from PyQt5.QtCore import QTimer, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QDialog

# Simple window used in history window widget to reproduce
# the clips stored in annotation buffer
# to understand if the preferency made is correct or not.
class ReplayClipsWindow(QDialog):
    def __init__(self, path):
        super().__init__()

        # Define model and controllar
        self._model = ReplayClipsWindowModel(path)
        self._controller = ReplayClipsWindowController(self._model) 


        # Define 
        self._display = Display(self._model.timer, self._model, self._model.replayModelImageSignal)


        self.ui = Ui_ReplayClipsWindow()
        self.ui.setupUi(self)
        self.ui.verticalLayout.addWidget(self._display)

        self.ui.startReplay.clicked.connect(lambda : self._controller.display_figure())


class ReplayClipsWindowModel(QObject):
    replayModelImageSignal = pyqtSignal(list)
    def __init__(self, path):
        super().__init__()
        
        # Define Timer and interval period
        self._timer = QTimer()
        self._timer.setInterval(400)
        self._path = path

        # Define variable for diplay clips
        self._display_image = self.load_image()

    @property
    def display_image(self):
        return self._display_image
    
    @property
    def timer(self):
        return self._timer

    
    def load_image(self):
        images = []
        for img in sorted(os.listdir(self._path)):
            if '.png' in img:
                #image = cv2.imread(self._path + '/' + img)
                image = Image.open(self._path + '/' + img)
                images.append(image.convert("RGB").resize((800, 700)))
                #images.append(cv2.resize(image, (800, 700)))
        return images


class ReplayClipsWindowController(QObject):

    def __init__(self, model):
        super().__init__()

        self._model = model

    @pyqtSlot()
    def display_figure(self):
        self._model.replayModelImageSignal.emit(self._model.display_image)
        for i in range(len(self._model.display_image)):
            self._model.timer.start()
        