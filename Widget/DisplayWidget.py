from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap

class Display(QLabel):
    def __init__(self, timer, model):
        super().__init__()

        # Connect the model
        self._model = model

        # Deifine variable to display clips
        self.count = 0

        # Define timer and connect it to display function
        self.timer = timer
        self.timer.timeout.connect(lambda : self.update_display())

        self.setAlignment(Qt.AlignCenter)

    @pyqtSlot()
    def update_display(self):
        if self.count != len(self._model.displayImage):
            h, w, ch = self._model.displayImage[self.count].shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(self._model.displayImage[self.count], w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.setPixmap(QPixmap.fromImage(convertToQtFormat))
            self.resize(1000, 900)
            self.count += 1
        else:
            self.count = 0
            self.timer.stop()




        
        