import os
import sys
import shutil
import re
import numpy as np

from ReinforcementLearning.csvRewardModel import save_reward_weights
from Utility.utility import save_model_parameters, load_annotation_buffer

from PyQt5.QtCore import QThread, pyqtSlot, QObject, pyqtSignal


# Data loader. We create batch for train the reward model sampling
# annotation buffer items with bootstrapping. 
def data_loader(annotation_buffer, batch):
    if len(annotation_buffer) < batch:
        return annotation_buffer

    else:
        index = np.random.randint(0, len(annotation_buffer), size=batch)
        train_data = [annotation_buffer[i] for i in index]

        return train_data


class ThreadSignals(QObject):
    finishedSignal = pyqtSignal()


# Simple class that make the reward model train.
# For K times, from annotation buffer, is created different train 
# clips set that is given to the reward model. 
class RewardThread(QThread):

    def __init__(self, model):
        super(RewardThread, self).__init__()

        self._model = model
        self._signals = ThreadSignals()

    @pyqtSlot()
    def run(self):
        loss = []

        self._model.annotation_buffer = load_annotation_buffer(self._model.auto_save_folder + '/annotation_buffer/')

        for k in range(int(self._model.model_parameters['K'])):
            self._model.logBarSxSignal.emit(
                "Train reward model : k-batch " + str(k) + ' of ' + str(self._model.model_parameters['K']))
            train_clips = data_loader(self._model.annotation_buffer, self._model.reward_batch)
            l, _, _ = self._model.reward_model.compute_rewards(self._model.reward_model, self._model.optimizer_r,
                                                               train_clips)
            loss.append(l)
        self._model.reward_loss.append((sum(loss) / len(loss)))

        # Reset all the variables used during the current training protocol iteration (policy, annotation and reward model parameters)
        self._model._iteration = 0
        self._model._model_parameters['idx'] = 0
        self._model._ann_point = 0
        self._model.clip_point = 0
        self._model.annotation_buffer = []
        self._model._annotation = None
        self._model.resetHistoryWindowSignal.emit()

        # Reset all folders used for the current training epoch
        self._model.annotator.reset_clips_database(self._model.clips_database)
        self._model.annotator.reset_clips_database(self._model.history_database)
        if [self._model.auto_save_folder + '/' + el for el in os.listdir(self._model.auto_save_folder) if
            'annotation_buffer' in el]:
            shutil.rmtree([self._model.auto_save_folder + '/' + el for el in os.listdir(self._model.auto_save_folder) if
                           'annotation_buffer' in el][0])

        # Auto save policy weight, reward model weight and model parameters.
        save_model_parameters(self._model.auto_save_folder, self._model.model_parameters, self._model.iteration,
                              self._model.process)
        save_reward_weights(self._model.reward_model, self._model.auto_save_folder, self._model.weigth_path,
                            self._model.model_parameters['lr'], self._model.model_parameters['K'],
                            self._model.reward_loss)

        self._model.logBarSxSignal.emit("End train reward model, the loss is : {:.3f}".format((sum(loss) / len(loss))))
        self._model.logBarDxSignal.emit("Press process to continue or quit application")

        self._model.processButton = True
        self._signals.finishedSignal.emit()
