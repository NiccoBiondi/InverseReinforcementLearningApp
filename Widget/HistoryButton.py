from Widget.HistoryWindow import HistoryWindow
from PyQt5.QtWidgets import QPushButton, QSizePolicy


class HistoryButton(QPushButton):
    ''' Choose button to decide the clips you prefer '''

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)

        # Define button functionality: open history dialog window and take enventually 
        # selected element and give them to the controller
        #self._on_configure = on_configure

        # Connect History button to its Window and define its ok_button configuration
        self._window = HistoryWindow(model)

        self._name ='history'
        self.setText('{}'.format(self._name))
        self.clicked.connect(lambda : self._window.exec_())
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
    '''
    def open_historyWindow(self):
        self._window.exec_()  
        if len(self._window.get_selectedElement()) > 0:
            self._on_configure(self._window.get_selectedElement())
        
        #self._window.reset_list()
    '''