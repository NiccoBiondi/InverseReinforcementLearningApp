from PyQt5.QtWidgets import QCheckBox, QSizePolicy

# Check Box to set or unset the Oracle used in annotation. 
# The user can select in every moment he wants this box 
# and the annotation will be performed by the aftificial 
# annotator.
class OracleButton(QCheckBox):

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

            if self._model.display_imageDx and self._model.display_imageSx:
                self._model.timer_dx.start()
                self._model.timer_sx.start()

        self._func(self.isChecked())