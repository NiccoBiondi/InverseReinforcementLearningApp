from PyQt5.QtWidgets import QPushButton, QSizePolicy

class ProcessButton(QPushButton):
    ''' Choose button to decide the clips you prefer '''

    def __init__(self, on_configure, **kwargs):
        super().__init__(**kwargs)

        self._name = 'process'
        self.setText('{}'.format(self._name))
        self.clicked.connect(on_configure)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
    
    #TODO:
    def setVisability(self):
        self.setEnabled(not self.isEnabled())
