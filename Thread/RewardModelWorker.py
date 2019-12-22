import os
import sys
import numpy as np

from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.ThreadUtility import clips_generator, save_clips, save_model

from PyQt5.QtCore import QRunnable, pyqtSlot, QObject, pyqtSignal

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    '''
    finishedSignal = pyqtSignal()

class RewardModelWorker(QRunnable):

    def __init__(self, model):
        super(RewardModelWorker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self._signals = WorkerSignals()
        self._model = model
        self._train = True if os.listdir(self._model._weigth_path) else False
        self._done = False

    @pyqtSlot()
    def run(self):
        self._model.processButton = True
  

        
        