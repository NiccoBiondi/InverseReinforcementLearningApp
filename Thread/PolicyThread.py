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

# Simple thread that make the policy function.
# From a number of episodes it creates trajectory from witch
# it creates clips which are saved in Clips Database folder.
# Thanks to this the user can check the single clips.
class PolicyThread(QThread):

    def __init__(self, model):
        super().__init__()

        self._model = model
        self._signals = ThreadSignals()
        self._max_len = 50
        self._train = True if os.listdir(self._model.weigth_path) else False
        self._done = False



    def run(self):

        clips_generated = []
        
        for step in range(self._model.iteration, int(self._model.model_parameters['episodes'])):
            self._model.logBarSxSignal.emit('Policy processing :' +  str(self._model.iteration + 1) + '/' + str(self._model.model_parameters['episodes']) + ' episodes')
            
            (states, actions, dones) = run_episode(self._model.env, self._model.policy, int(self._model.model_parameters['episode_len']))

            clips = clips_generator(states, dones, int(self._model.model_parameters['clips_len']))

            # Sample the clips generated
            for index in np.random.randint(low = 0, high = len(clips), size= len(clips)//2):
                if len(clips_generated) == self._max_len and self._model.model_parameters['idx']  < int(self._model.model_parameters['n_annotation']):
                    clips_path = save_clips(self._model.clips_database + '/clipsToAnnotate_' + str(self._model.model_parameters['idx']), clips_generated)
                    self._model.folder = '/clipsToAnnotate_' + str(self._model.model_parameters['idx'])
                    clips_generated = [clips[index]]
                    self._model.model_parameters = ['idx', self._model.model_parameters['idx'] + 1]
                    if not self._done:
                        self._signals.startAnnotation.emit()
                        self._done = True

                clips_generated.append(clips[index])

            # Auto save controll.
            if step > 0 and step % self._model.auto_save_clock_policy == 0:
                    save_model(self._model.auto_save_folder, self._model.policy, self._model.model_parameters, self._model.iteration)
                    self._model.logBarSxSignal.emit('Auto-save in :' +  self._model.auto_save_folder)
                    self._model.annoatate = True
                    time.sleep(0.5) 

            # If the reward model is trained one time the policy can be trained.
            # To train the policy are used the rewards computed by the reward model.
            if not self._train:
                s = [obs['obs'] for obs in states]
                rewards = self._model.reward_model(s)
                l = Loss(self._model.policy, self._model.optimizer_p, states, actions, rewards)
                print("Train policy loss: {:.3f}".format((sum(l)/len(l))))
            
            self._model.iteration += 1 
        
        # When the policy makes all episodes reset all and save the weight
        save_model(self._model.auto_save_folder, self._model.policy, self._model.model_parameters, self._model.iteration)
        self._model.iteration = 0