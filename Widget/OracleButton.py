from PyQt5.QtWidgets import QCheckBox, QSizePolicy

class OracleButton(QCheckBox):
    ''' Buton to set or unset the oracle used in annotation '''

    def __init__(self, on_configure, **kwargs):
        super().__init__(**kwargs)
        self._name = 'oracle'
        self._model = on_configure
        self.setText('{}'.format(self._name))
        self.clicked.connect(self.check_changed)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

    def check_changed(self):
        self._model(self.isChecked())