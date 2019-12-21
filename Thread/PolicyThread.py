import os
import sys
import time
import numpy as np

from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.PolicyThreadUtility import clips_generator, save_clips, save_model_parameters

from PyQt5.QtCore import QThread

def save_model(path, policy, model_parameters):
    if not os.path.exists:
        os.makedirs(path)
    save_policy_weights(policy, path)
    save_model_parameters(path, model_parameters)


class PolicyThread(QThread):

    def __init__(self, model):
        super().__init__()

        self._model = model
        self._max_len = 50
        self._train = True if os.listdir(self._model._weigth_path) else False
        

    def run(self):

        clips_generated = []
        
        for step in range(self._model._iteration, int(self._model.model_parameters['episodes'])):
            
            (states, actions, dones) = run_episode(self._model._env, self._model._policy, int(self._model._model_parameters['episode_len']))

            clips = clips_generator(states, dones, int(self._model._model_parameters['clips_len']))

            # Sample the clips generated
            for index in np.random.randint(low = 0, high = len(clips), size= len(clips)//2):
                if len(clips_generated) == 50:
                    save_clips(self._model._clips_database + '/clips2annotate_' + str(self._model._model_parameters['idx']), clips_generated)
                    clips_generated = [clips[index]]
                    self._model._model_parameters['idx'] += 1

                clips_generated.append(clips[index])

            if step > 0 and step % self._model._auto_save_clock_policy == 0:
                    save_model(self._model._auto_save_foder, self._model.policy, clips_generated, self._model._model_parameters)
                    self._model.logBarSxSignal.emit('Auto-save in :' +  self._model._auto_save_foder)
                    time.sleep(0.5)

            if self._train:

                s = [obs['obs'] for obs in states]
                rewards = self._model._reward_model(s)
                l = Loss(self._model._policy, self._model._optimizer_p, states, actions, rewards)

                print("Train policy loss: {:.3f}".format((sum(l)/len(l))))
            
            self._model._iteration += 1 
            self._model.logBarSxSignal.emit('Policy processing :' +  str(step) + '/' + self._model.model_parameters['episodes'] + ' episodes')