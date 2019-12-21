from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton, QSizePolicy

class ChoiseButton(QPushButton):
    ''' Choose button to decide the clips you prefer '''

    def __init__(self, name, preferencies, on_configure, model, **kwargs):
        super().__init__(**kwargs)

        # Define button preferencies, it label the clisp when push the button
        self._preferencies = preferencies

        # Define update preferencies function
        self._on_configure = on_configure

        self._name = name
        self.setText('{}'.format(self._name))
        self.clicked.connect(lambda : self.update_preferencies())
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.setEnabled(False)

        model.choiseButtonVisiblitySignal.connect(self.setVisability)

    def update_preferencies(self):
        self._on_configure = self._preferencies
    
    @pyqtSlot()
    def setVisability(self):
        self.setEnabled(not self.isEnabled())
