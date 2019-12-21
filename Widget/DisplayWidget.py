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

    @pyqtSlot()
    def update_display(self):
        if self._count != len(self._displayImage):
            h, w, ch = self._displayImage[self.count].shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(self._displayImage[self.count], w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.setPixmap(QPixmap.fromImage(convertToQtFormat))
            self.resize(1000, 900)
            self._count += 1
        else:
            self._count = 0
            self._timer.stop()

    @pyqtSlot(list)
    def updateDisplayImage(self, image):
        self._displayImage = image




        
        