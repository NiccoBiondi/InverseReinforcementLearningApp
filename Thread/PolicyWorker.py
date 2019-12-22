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

class PolicyWorker(QRunnable):

    def __init__(self, model):
        super(PolicyWorker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self._signals = WorkerSignals()
        self._model = model
        self._train = True if os.listdir(self._model._weigth_path) else False
        self._done = False

    @pyqtSlot()
    def run(self):
        clips_generated = []

        while (not self._done):
            self._model.logBarSxSignal.emit('Policy processing :' +  str(self._model._iteration) + '/' + self._model.model_parameters['episodes'] + ' episodes')
        #for step in range(self._model._iteration, int(self._model._model_parameters['episodes'])):     
            (states, actions, dones) = run_episode(self._model._env, self._model._policy, int(self._model._model_parameters['episode_len']))

            clips = clips_generator(states, dones, int(self._model._model_parameters['clips_len']))

            # Sample the clips generated
            for index in np.random.randint(low = 0, high = len(clips), size= len(clips)//2):
                if len(clips_generated) == 50:
                    clips_path = save_clips(self._model._clips_database + '/clipsToAnnotate_' + str(self._model._model_parameters['idx']), clips_generated)
                    clips_generated = [clips[index]]
                    self._model._model_parameters['idx'] += 1
                    self._done = True

                clips_generated.append(clips[index])

            if self._model._iteration > 0 and self._model._iteration % self._model._auto_save_clock_policy == 0:
                    save_model(self._model._auto_save_folder, self._model._policy, self._model._model_parameters)
                    self.logBarSxSignal.emit('Auto-save in :' +  self._model._auto_save_folder)

            if not self._train:

                s = [obs['obs'] for obs in states]
                rewards = self._model._reward_model(s)
                l = Loss(self._model._policy, self._model._optimizer_p, states, actions, rewards)
                print("Train policy loss: {:.3f}".format((sum(l)/len(l))))
            
            self._model._iteration += 1 
    
        clips_generated = []
        if self._model._iteration == int(self._model._model_parameters['episodes']):
            save_model(self._model._auto_save_folder, self._model._policy, self._model._model_parameters, self._model._iteration)
            self._model._iteration = 0
        
        self._signals.finishedSignal.emit()

        
        