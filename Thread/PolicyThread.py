import os
import gc
import sys
import time
import cv2
import copy
import re
import numpy as np

from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.utility import clips_generator, save_clips, save_model

from PyQt5.QtCore import QThread, pyqtSignal, QObject

class ThreadSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    '''
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
        self._train = False 
        self._done = False

    @property
    def done(self):
        return self._done
  
    @done.setter
    def done(self, _bool):
        self._done = _bool

    def trainable(self):
        pattern = re.compile('csv_reward_weight.*pth$')
        w_path = list(filter(pattern.match, os.listdir(self._model.auto_save_folder)))
        if len(w_path) > 0:
            self._train = True  
        else:
            self._train = False

    def run(self):

        clips_generated = []
        rewards = []
        l = []

        # Check if in weigth_init path there is the reward model weigth. In the positive case, the policy
        # can be trained, else the policy has to create the clips.
        self._train = True if 'csv_reward_weight_lr' + str(self._model.model_parameters['lr']) + '_k' + str(self._model.model_parameters['K']) + '.pth' in os.listdir(self._model.weigth_path) else False
        
        for step in range(self._model.iteration, int(self._model.model_parameters['episodes'])):
            
            self._model.logBarSxSignal.emit('Policy processing :' +  str(self._model.iteration + 1) + '/' + str(self._model.model_parameters['episodes']) + ' episodes')
            
            (states, actions, dones, grids) = run_episode(self._model.env, self._model.policy, int(self._model.model_parameters['episode_len']), self._model.grid_wrapper)

            states_copy = copy.deepcopy(states)
            clips, clips_grid = clips_generator(states_copy, dones, int(self._model.model_parameters['clips_len']), grids)
            idx = save_clips(self._model.clips_database, self._model.model_parameters['idx'], clips, clips_grid)
            self._model.model_parameters = ['idx', idx + 1]
                        
            # Auto save controll.
            if step > 0 and step % self._model.auto_save_clock_policy == 0:
                    save_model(self._model.auto_save_folder, self._model.policy, self._model.model_parameters, self._model.iteration)
                    self._model.logBarSxSignal.emit('Auto-save in :' +  self._model.auto_save_folder)
                    self._model.annoatate = True

            # If the reward model is trained one time the policy can be trained.
            # To train the policy are used the rewards computed by the reward model.
            if self._train:
                reward = self._model.reward_model([obs['obs'] for obs in states])
                #rewards += [reward[i].item() for i in range(len(reward))]
                #for i in range(len(reward)):
                #   reward[i] = ( reward[i].item() - np.mean(rewards) ) / 0.05 
                losses = Loss(self._model.policy, self._model.optimizer_p, states, actions, reward)
                for i in range(len(losses)):
                    l.append(losses[i])
                

            self._model.iteration += 1
            gc.collect()
        print("Train policy loss: {:.6f}".format((sum(l)/len(l))))   
        self._model.logBarSxSignal.emit('Training of policy finished')

        # When the policy makes all episodes save the weight and model parameters
        save_model(self._model.auto_save_folder, self._model.policy, self._model.model_parameters, self._model.iteration)
        
        self._signals.finishedSignal.emit()
