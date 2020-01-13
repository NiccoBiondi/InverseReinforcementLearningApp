from PyQt5.QtWidgets import QCheckBox, QSizePolicy

class OracleButton(QCheckBox):
    ''' Buton to set or unset the oracle used in annotation '''

    def __init__(self, model, on_configure, **kwargs):
        super().__init__(**kwargs)
        self._name = 'oracle'
        self._model = model
        self._func = on_configure
        self.setText('{}'.format(self._name))
        self.clicked.connect(self.check_changed)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

    def check_changed(self):
        if self.isChecked():
            self._model.choiceButton = False
            self._model.oracle_timer.start()
        else:
            if len(self._model.display_imageDx) > 0:
                self._model.timer_dx.start()
                self._model.timer_sx.start()

        self._func(self.isChecked())