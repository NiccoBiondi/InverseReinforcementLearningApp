from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton, QSizePolicy


class ReplayButton(QPushButton):
    ''' Choose button to decide the clips you prefer '''

    def __init__(self, timer, model, **kwargs):
        super().__init__(**kwargs)

        # Connect to the model and timer
        self._model = model
        self._timer = timer

        self.setText('{}'.format('replay video'))
        self.clicked.connect(lambda : self.replayVideo())
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.setEnabled(False)

        self._model.choiceButtonVisiblitySignal.connect(self.setVisability)
    
    def replayVideo(self):
        self._model.choiceButton = False
        for i in range(self._model.display_imageLen):
            self._timer.start()

    @pyqtSlot(bool)
    def setVisability(self, enable):
        self.setEnabled(enable)