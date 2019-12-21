from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton, QSizePolicy


class SpeedButton(QPushButton):
    ''' Define speed-up of a button '''

    def __init__(self, on_configure, model, **kwargs):
        super().__init__(**kwargs)

        self.setText('Speed : 1x')
        self.clicked.connect(lambda : on_configure())
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        # Connect Button with model
        model.setSpeedSignal.connect(self.increment_velocity) 

    @pyqtSlot(str)
    def increment_velocity(self, speed):
        
        self.setText('Speed : {}x'.format(speed))