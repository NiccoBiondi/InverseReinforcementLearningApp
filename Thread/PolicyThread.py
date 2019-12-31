import os
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
        self._train = True if 'csv_reward_weight_lr' + self._model.model_parameters['lr'] + '_k' + self._model.model_parameters['K'] + '.pth' in os.listdir(self._model.weigth_path) else False
        clips_generated = []
        if self._model.model_parameters['idx'] == int(self._model.model_parameters['n_annotation']):
            self._signals.startAnnotation.emit()
        
        for step in range(self._model.iteration, int(self._model.model_parameters['episodes'])):
            
            self._model.logBarSxSignal.emit('Policy processing :' +  str(self._model.iteration + 1) + '/' + str(self._model.model_parameters['episodes']) + ' episodes')
            
            (states, actions, dones) = run_episode(self._model.env, self._model.policy, int(self._model.model_parameters['episode_len']))
            
            states_copy = copy.deepcopy(states)
            clips = clips_generator(states_copy, dones, int(self._model.model_parameters['clips_len']))

            # Sample the clips generated
            for index in np.random.randint(low = 0, high = len(clips), size= len(clips)//2):
                if len(clips_generated) == self._max_len and self._model.model_parameters['idx']  < int(self._model.model_parameters['n_annotation']):
                    clips_path = save_clips(self._model.clips_database + '/clipsToAnnotate_' + str(self._model.model_parameters['idx']), clips_generated)
                    self._model.folder = '/clipsToAnnotate_' + str(self._model.model_parameters['idx'])
                    clips_generated = [clips[index]]
                    self._model.model_parameters = ['idx', self._model.model_parameters['idx'] + 1]
                    if not self.done:
                        self._signals.startAnnotation.emit()
                        self.done = True

                clips_generated.append(clips[index])

            # Auto save controll.
            if step > 0 and step % self._model.auto_save_clock_policy == 0:
                    save_model(self._model.auto_save_folder, self._model.policy, self._model.model_parameters, self._model.iteration)
                    self._model.logBarSxSignal.emit('Auto-save in :' +  self._model.auto_save_folder)
                    self._model.annoatate = True
                    time.sleep(0.5) 

            # If the reward model is trained one time the policy can be trained.
            # To train the policy are used the rewards computed by the reward model.
            if self._train:
                s = [obs['obs'] for obs in states]
                rewards = self._model.reward_model(s)
                if len(states) != 81:
                    print(len(states), len(actions), len(rewards), 'qua qua ')
                l = Loss(self._model.policy, self._model.optimizer_p, states, actions, rewards)
                print("Train policy loss: {:.3f}".format((sum(l)/len(l))))
            
            self._model.iteration += 1 
        
        self._model.logBarSxSignal.emit('Training of policy finished')
        if int(self._model.model_parameters['idx']) < int(self._model.model_parameters['n_annotation']):
            self._model.model_parameters = ['n_annotation', self._model.model_parameters['idx']]

        # When the policy makes all episodes save the weight and model parameters
        save_model(self._model.auto_save_folder, self._model.policy, self._model.model_parameters, self._model.iteration)

        self._signals.finishedSignal.emit()