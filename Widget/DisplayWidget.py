import cv2
import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap

class Display(QLabel):
    def __init__(self, timer, model, signal):
        super().__init__()

        # Connect the model
        self._model = model

        # Deifine variable to display clips
        self._count = 0
        self._displayImage = []

        # Define timer and connect it to display function
        self._timer = timer
        self._timer.timeout.connect(lambda : self.update_display())

        # Connect signal
        signal.connect(self.updateDisplayImage)

        self.setAlignment(Qt.AlignCenter)

    # Simple function t upload the displayed image
    @pyqtSlot()
    def update_display(self):

        if len(self._displayImage) == 0:
            self._timer.stop()

        else:
            if not self._model.oracle_active:

                if self._count != len(self._displayImage) :
                    image = cv2.cvtColor(np.array(self._displayImage[self._count]), cv2.COLOR_RGB2BGR)
                    h, w, ch = image.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                    self.setPixmap(QPixmap.fromImage(convertToQtFormat))
                    self._count += 1
                else:
                    self._count = 0
                    self._timer.stop()
                    self._model.choiceButton = True

            else:
                self._model.oracle_timer.start()

    # Funtion to upload the displayImage variable
    @pyqtSlot(list)
    def updateDisplayImage(self, image):
        self._displayImage = image
        self._displayImage.insert(0, Image.new('RGB', (800, 800), color=(255, 255, 255)))
        



        
        