import os
import sys
import time

from PyQt5.QtCore import QThread

class ClipsThread(QThread):

    def __init__(self, model):
        super().__init__()

        self._model = model


    def run(self):

        while(not os.listdir(self._model._clips_database)):
            self._model.logBarDxSignal.emit('Wait clips to annotate')
        
        for clips_folder in os.listdir(self._model._clips_database):
            for clip in os.listdir(self._model._clips_database + 'clips_folder/'):
                self._model.choiseButton = True
                time.sleep(0.5)
                self._model.choiseButton = False