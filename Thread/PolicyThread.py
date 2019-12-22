import os
import sys
import time
import cv2
import numpy as np

from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.PolicyThreadUtility import clips_generator, save_clips, save_model_parameters, save_model, save_annotation

from PyQt5.QtCore import QThread


class PolicyThread(QThread):

    def __init__(self, model):
        super().__init__()

        self._model = model
        self._max_len = 50
        self._train = True if os.listdir(self._model._weigth_path) else False


    def run(self):

        clips_generated = []
        annotation = []
        clips_path = []
        ann_clips = []
        disp_figure = []

        for step in range(self._model._iteration, int(self._model.model_parameters['episodes'])):
            
            (states, actions, dones) = run_episode(self._model._env, self._model._policy, int(self._model._model_parameters['episode_len']))

            clips = clips_generator(states, dones, int(self._model._model_parameters['clips_len']))

            # Sample the clips generated
            for index in np.random.randint(low = 0, high = len(clips), size= len(clips)//2):
                if len(clips_generated) == 50:
                    clips_path = save_clips(self._model._clips_database + '/clips2annotate_' + str(self._model._model_parameters['idx']), clips_generated)
                    annotation = clips_generated
                    clips_generated = [clips[index]]
                    self._model._model_parameters['idx'] += 1

                clips_generated.append(clips[index])

            if len(annotation) > 0:

                for i, ann in enumerate(annotation):
                    c = [clip['obs'] for clip in ann]
                    ann_clips.append({ 'clip' : c, 'path': clips_path[i]})
                    disp_figure.append([cv2.resize(clip['image'], (800, 700)) for clip in ann])
                


                for idx in range(0, len(disp_figure), 2):
                    
                    self._model.display_imageLen = len(disp_figure[idx])
                    self._model.diplay_imageDx = disp_figure[idx]
                    self._model.display_imageSx = disp_figure[idx + 1]
                    self._model.choiseButton = True

                    self._model.logBarDxSignal.emit('Waiting annotation')
                    while(self._model._preferencies == None):
                        continue

                    self._model._annotation_buffer.append([ann_clips[idx]['clip'], ann_clips[idx + 1]['clip'], self._model._preferencies])
                    annotation = [ann_clips[idx]['path'], ann_clips[idx + 1]['path'], '[' + str(self._model._preferencies[0]) + ',' + str(self._model._preferencies[1]) + ']']
                    self._model.annotation = annotation
                    self._model.choiseButton = False
                save_annotation(self._model._auto_save_foder, self._model._annotation_buffer)
            
            annotation = []
            ann_clips = []
            disp_figure = []

            #clips, disp_figure = self._model._annotator.load_clips_figure(self._model._clips_database, clips_folder)


            if step > 0 and step % self._model._auto_save_clock_policy == 0:
                    save_model(self._model._auto_save_foder, self._model.policy, clips_generated, self._model._model_parameters)
                    self._model.logBarSxSignal.emit('Auto-save in :' +  self._model._auto_save_foder)
                    self._model.annoatate = True
                    time.sleep(0.5) 

            if not self._train:

                s = [obs['obs'] for obs in states]
                rewards = self._model._reward_model(s)
                l = Loss(self._model._policy, self._model._optimizer_p, states, actions, rewards)
            

                #print("Train policy loss: {:.3f}".format((sum(l)/len(l))))
            
            self._model._iteration += 1 
            self._model.logBarSxSignal.emit('Policy processing :' +  str(step) + '/' + self._model.model_parameters['episodes'] + ' episodes')
        
        self._model._iteration = 0