from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton, QSizePolicy

class ProcessButton(QPushButton):
    ''' Button used to start the procss : 
            
            - policy train and creation of clips
            - clips annotation
            - reward model train

    '''

    def __init__(self, on_configure, model, **kwargs):
        super().__init__(**kwargs)

        self._name = 'process'
        self.setText('{}'.format(self._name))
        self.clicked.connect(on_configure)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        model.processButtonVisiblitySignal.connect(self.setVisability)
    
    @pyqtSlot()
    def setVisability(self):
        self.setEnabled(not self.isEnabled())
