import os
import sys
import shutil
import numpy as np

from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.ThreadUtility import save_model_parameters

from PyQt5.QtCore import QRunnable, pyqtSlot, QObject, pyqtSignal

def data_loader(annotation_buffer, batch):
    
    if len(annotation_buffer) < batch:
        return annotation_buffer

    else:
        index = np.random.randint(0, len(annotation_buffer), size = batch)
        train_data = [annotation_buffer[i] for i in index]  

        return train_data

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
        loss = []

        for k in range(self._model._model_parameters['K']):
            self._model.logBarSxSignal("Train reward model : k-batch " + str(k) + ' of ' + str(self.values['K']) )
            train_clips = data_loader(self._model._annotation_buffer, self._model._reward_batch)
            loss.append(self._model._reward_model.compute_rewards(self._model._reward_model, self._model._optimizer_r, train_clips, self._model._logBarDxSignal))

        self._model._iteration = 0
        self._model._model_parameters['idx'] = 0
        shutil.rmtree(self._model._auto_save_folder + '/annotation_buffer')
        save_model_parameters(self._model._auto_save_folder, self._model._model_parameters, self._model._iteration)
        
        self._model._reward_model.save_reward_weights(self._model._reward_model, self._model._auto_save_folder)
        self._model._logBarDxSignal.emit("End train reward model, the loss is : {:.3f}".format((sum(loss)/len(loss))))
        self._model._logBarSxSignal.emit("Press process to continue or quit application")
        self._model.processButton = True
  

        
        