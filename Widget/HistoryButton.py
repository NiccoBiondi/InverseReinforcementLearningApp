from Widget.HistoryWindow import HistoryWindow
from PyQt5.QtWidgets import QPushButton, QSizePolicy

# Button in AlgView that open the History Window
class HistoryButton(QPushButton):
    
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)

        # Define the model
        self._model = model

        # Connect History button to its Window and define its ok_button configuration
        self._window = HistoryWindow(model)

        self._name ='history'
        self.setText('{}'.format(self._name))
        self.clicked.connect(lambda : self._window.exec_())
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        self._model.resetHistoryWindowSignal.connect(self.resetHistoryWindow)

    def resetHistoryWindow(self):
        self._window = HistoryWindow(self._model)