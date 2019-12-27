from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton, QSizePolicy

class ChoiceButton(QPushButton):
    ''' Choose button to decide the clips you prefer '''

    def __init__(self, name, preferences, on_configure, model, **kwargs):
        super().__init__(**kwargs)

        # Define button preferences, it label the clisp when push the button
        self._preferences = preferences

        # Define update preferences function
        self._on_configure = on_configure

        self._name = name
        self.setText('{}'.format(self._name))
        self.clicked.connect(lambda : self.update_preferences())
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.setEnabled(False)

        model.choiceButtonVisiblitySignal.connect(self.setVisability)

    def update_preferences(self):
        self._on_configure(self._preferences)
    
    @pyqtSlot()
    def setVisability(self):
        self.setEnabled(not self.isEnabled())
