import os
import sys
import time
import cv2
import numpy as np

from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.utility import clips_generator, save_clips, save_model

from PyQt5.QtCore import QThread, pyqtSignal, QObject

class ThreadSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    '''
    startAnnotation = pyqtSignal()
    finishedSignal = pyqtSignal()

class PolicyThread(QThread):

    def __init__(self, model):
        super().__init__()

        self._model = model
        self._signals = ThreadSignals()
        self._max_len = 50
        self._train = True if os.listdir(self._model._weigth_path) else False
        self._done = False


    def run(self):

        clips_generated = []

        for step in range(self._model._iteration, int(self._model.model_parameters['episodes'])):
            
            (states, actions, dones) = run_episode(self._model._env, self._model._policy, int(self._model._model_parameters['episode_len']))

            clips = clips_generator(states, dones, int(self._model._model_parameters['clips_len']))

            # Sample the clips generated
            for index in np.random.randint(low = 0, high = len(clips), size= len(clips)//2):
                if len(clips_generated) == 50:
                    clips_path = save_clips(self._model._clips_database + '/clipsToAnnotate_' + str(self._model._model_parameters['idx']), clips_generated)
                    clips_generated = [clips[index]]
                    self._model._model_parameters['idx'] += 1
                    if not self._done:
                        self._signals.startAnnotation.emit()
                        self._done = True
                    #self._model.start_annotation = True

                clips_generated.append(clips[index])

            if step > 0 and step % self._model._auto_save_clock_policy == 0:
                    save_model(self._model._auto_save_folder, self._model.policy, self._model._model_parameters)
                    self._model.logBarSxSignal.emit('Auto-save in :' +  self._model._auto_save_folder)
                    self._model.annoatate = True
                    time.sleep(0.5) 

            if not self._train:

                s = [obs['obs'] for obs in states]
                rewards = self._model._reward_model(s)
                l = Loss(self._model._policy, self._model._optimizer_p, states, actions, rewards)
                print("Train policy loss: {:.3f}".format((sum(l)/len(l))))
            
            self._model._iteration += 1 
    
        clips_generated = []
        save_model(self._model._auto_save_folder, self._model.policy, self._model._model_parameters, self._model._iteration)
        self._model._iteration = 0