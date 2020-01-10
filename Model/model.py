import sys
import os
import re 
import shutil
import numpy as np
from datetime import date

import torch
import gym
import gym_minigrid

from ReinforcementLearning.Oracle import Oracle
from ReinforcementLearning.csvRewardModel import csvRewardModel
from ReinforcementLearning.policy import Policy
from ReinforcementLearning.wrapper import RGBImgObsWrapper, FullyObsWrapper
from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.annotator import Annotator

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QWidget

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))

class Model(QObject):
    preferenceChangedSignal = pyqtSignal()
    processButtonVisiblitySignal = pyqtSignal()
    resetHistoryWindowSignal = pyqtSignal()
    choiceButtonVisiblitySignal = pyqtSignal(bool)
    updateDisplaySxImageSignal = pyqtSignal(list)
    updateDisplayDxImageSignal = pyqtSignal(list)
    setClipsHistorySignal = pyqtSignal(list)
    pathLoadedSignal = pyqtSignal(str)
    setSpeedSignal = pyqtSignal(str)
    logBarSxSignal = pyqtSignal(str)
    logBarDxSignal = pyqtSignal(str)
    changeWindowSignal = pyqtSignal(object)
    
    
    def __init__(self):
        super().__init__()

        # Define botton visibility variable.
        self._processButton = True
        self._choiceButton = False

        # Define initialize model variable. If the user
        # don't initialize or load the model, he can't start
        # to annotate the clips.
        self._model_init = False
        self._model_load = False

        # Define default path where the clips are saved by the policy
        # thread, the application can make auto save action, and where the
        # reward model initial weight is saved. "_load_path" variable
        # is used to memorize if the user loads a checkpoint model.
        self._weigth_path = DIR_NAME +  '/ReinforcementLearning/reward_model_init_weight/'
        self._auto_save_folder = DIR_NAME + '/SAVE_FOLDER/'
        self._clips_database = DIR_NAME + '/Clips_Database/'
        self._history_database = DIR_NAME + '/History_Database/'
        self._load_path = ''

        # Define variable to train policy and reward model.
        self._annotation = None      # Is used to update the history window widget with the current annotation made
        self._annotation_buffer = [] # Memorize the clips annotate and the preference
        self._oracle_active = False         # Boolean variable used to understand if the oracle is used or not

        # Define util variable.
        self._iteration = 0                # Memorize the episodes where the policy arrived.
        self._ann_point = 0                # Memorize the folder where the annotator arrive.
        self._auto_save_clock_policy = 100 # Define auto save cock period for the policy thread.
        self._annotator = Annotator()      # Utility class to reload the csv and the image which represent clips to annotate.
        self._model_parameters = {}        # Define model initial parameters like learning rate, environment name, trajectory length etc.
        self._preferences = None           # Utility function used to take the preferences of the user during annotation
        self._start_ann_disp = 0           # Variable used only for graphic aim : take updated the history window 'List pos' column in correct way

        # Define variable used to the annotation phase.
        self._clips = []       # Contain clips to annotate
        self._disp_figure = [] # Contain clips images to annotate

        # Define Inverse Reinforcement Learning element
        self._obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
        self._act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
        self._inner_size = 64   # Number of neurons in two hidden layers.
        self._reward_batch = 16 # Reward model batch size

        self._env = None           
        self._reward_model = csvRewardModel(obs_size = self._obs_size, inner_size = self._inner_size).cuda()
        self._policy = Policy(obs_size = self._obs_size, act_size = self._act_size, inner_size = self._inner_size).cuda()
        self._optimizer_p = None
        self._optimizer_r = None 

        # Define oracle variable
        self._oracle = None       
        self._grid_wrapper = None  
        self._oracle_timer = QTimer()
        self._oracle_timer.setInterval(500)

        # Define the two Display and replay buttons timers
        self._timer_dx = QTimer()
        self._timer_dx.setInterval(350)
        self._timer_sx = QTimer() 
        self._timer_sx.setInterval(350)
        self._currentInterval = 350
        self._speed = 1
        self._display_imageLen = 0
        self._display_imageDx = []
        self._display_imageSx = []

    # Define a collection of property and property.setter.

    @property
    def oracle_timer(self):
        return self._oracle_timer

    @property
    def grid_wrapper(self):
        return self._grid_wrapper

    @property
    def start_ann_disp(self):
        return self._start_ann_disp

    @property
    def annotator(self):
        return self._annotator

    @property
    def env(self):
        return self._env

    @property
    def policy(self):
        return self._policy

    @property
    def clips_database(self):
        return self._clips_database

    @property
    def history_database(self):
        return self._history_database
    
    @property
    def auto_save_clock_policy(self):
        return self._auto_save_clock_policy

    @property
    def auto_save_folder(self):
        return self._auto_save_folder
    
    @property
    def annotation_buffer(self):
        return self._annotation_buffer

    @property
    def reward_model(self):
        return self._reward_model
    
    @property
    def reward_batch(self):
        return self._reward_batch
    
    @property
    def optimizer_p(self):
        return self._optimizer_p
        
    @property
    def optimizer_r(self):
        return self._optimizer_r

    @property
    def clips(self):
        return self._clips

    @property
    def disp_figure(self):
        return self._disp_figure

    @property
    def annotation(self):
        return self._annotation

    @property
    def display_imageLen(self):
        return self._display_imageLen

    @property
    def display_imageDx(self):
        return self._display_imageDx

    @property
    def display_imageSx(self):
        return self._display_imageSx

    @property
    def processButton(self):
        return self._processButton

    @property
    def choiceButton(self):
        return self._choiceButton

    @property
    def oracle_active(self):
        return self._oracle_active
    
    @property
    def timer_dx(self):
        return self._timer_dx
    
    @property
    def timer_sx(self):
        return self._timer_sx

    @property
    def speed(self):
        return self._speed

    @property
    def model_init(self):
        return self._model_init

    @property
    def model_load(self):
        return self._model_load

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def load_path(self):
        return self._load_path
    
    @property
    def preferences(self):
        return self._preferences

    @property 
    def iteration(self):
        return self._iteration

    @property
    def weigth_path(self):
        return self._weigth_path

    @property
    def ann_point(self):
        return self._ann_point

    @property
    def oracle(self):
        return self._oracle

    @grid_wrapper.setter
    def grid_wrapper(self, slot):
        self._grid_wrapper = slot

    @weigth_path.setter
    def weigth_path(self, path):
        self._weigth_path = path

    @start_ann_disp.setter
    def start_ann_disp(self, slot):
        self._start_ann_disp = slot

    @optimizer_p.setter
    def optimizer_p(self, slot):
        self._optimizer_p = slot

    @optimizer_r.setter
    def optimizer_r(self, slot):
        self._optimizer_r = slot

    @clips_database.setter
    def clips_database(self, slot):
        self._clips_database = slot
    
    @history_database.setter
    def history_database(self, slot):
        self._history_database = slot

    @env.setter
    def env(self, slot):
        self._env = slot

    @ann_point.setter
    def ann_point(self, slot):
        self._ann_point = slot

    @annotation_buffer.setter
    def annotation_buffer(self, slot):
        self._annotation_buffer = slot

    @auto_save_folder.setter
    def auto_save_folder(self, path):
        self._auto_save_folder = path

    @clips.setter
    def clips(self, slot):
        self._clips = slot

    @disp_figure.setter
    def disp_figure(self, slot):
        self._disp_figure = slot

    @annotation.setter
    def annotation(self, slot):
        self._annotation = slot
        self.setClipsHistorySignal.emit(slot)
    
    @display_imageLen.setter
    def display_imageLen(self, slot):
        self._display_imageLen = slot

    @display_imageDx.setter
    def display_imageDx(self, images):
        self._display_imageDx = images
        self.updateDisplayDxImageSignal.emit(images)
        self._timer_dx.start()

    @display_imageSx.setter
    def display_imageSx(self, images):
        self._display_imageSx = images
        self.updateDisplaySxImageSignal.emit(images)
        self._timer_sx.start()

    @processButton.setter
    def processButton(self, val):
        self._processButton = val
        self.processButtonVisiblitySignal.emit() 

    @choiceButton.setter
    def choiceButton(self, val):
        self._choiceButton = val
        self.choiceButtonVisiblitySignal.emit(val)

    @oracle_active.setter
    def oracle_active(self, slot):
        self._oracle_active = slot
    
    @iteration.setter
    def iteration(self, slot):
        self._iteration = slot

    @oracle.setter
    def oracle(self, slot):
        self._oracle = slot
        
    @model_init.setter
    def model_init(self, value):

        self._model_init = value
        self._model_load = not value
        self.load_path = ''

        # Init env and oracle matrix
        env = gym.make(self._model_parameters['minigrid_env'])
        self._env = RGBImgObsWrapper(env)
        self._grid_wrapper = FullyObsWrapper(env)
        self._oracle = Oracle(self._grid_wrapper, env)
        self._env.reset()

        # Create the auto save folder for a specific minigrifd env. If this folder still exists, then i delete it.
        self._auto_save_folder = self._auto_save_folder + self._model_parameters['minigrid_env'] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
        if not os.path.exists(self._auto_save_folder):
            os.mkdir(self._auto_save_folder)
        else:
            shutil.rmtree(self._auto_save_folder)
            os.mkdir(self._auto_save_folder)

        # Use the Adam optimizer.
        self._optimizer_p = torch.optim.Adam(params=self._policy.parameters(), lr = 1e-4)
        self._optimizer_r = torch.optim.Adam(params=self._reward_model.parameters(), lr = float(self._model_parameters['lr']), weight_decay=0.01)
        
        self._clips_database = self._clips_database + self._model_parameters['minigrid_env']
        self._history_database = self._history_database + self._model_parameters['minigrid_env']

        if not os.path.exists(self._clips_database):
            os.makedirs(self._clips_database)

        else:
            self._annotator.reset_clips_database(self._clips_database)

        if not os.path.exists(self._history_database):
            os.makedirs(self._history_database)
            
        else:
            self._annotator.reset_clips_database(self._history_database)

        # When a model is initialized I control if a previous reward model
        # weight is saved in reward_model_init_weight folder. So in the first moment 
        # is created a folder where inside there are all the reward model weight 
        # created for that environment. Then is set the reward model weight looking to the
        # learning rate and the K hyperparameters.
        self._weigth_path = self._weigth_path + self._model_parameters['minigrid_env'] 
        
        if not os.path.exists(self._weigth_path):
            os.makedirs(self._weigth_path)
        
        #if this path still exists, It is checked if inside there is a saved weight inside the folder.
        else:
            if 'csv_reward_weight_lr' + str(self._model_parameters['lr']) + '_k' + str(self._model_parameters['K']) + '.pth' in os.listdir(self._weigth_path):
                self.reward_model.load_state_dict(torch.load(self._weigth_path + '/csv_reward_weight_lr' + str(self._model_parameters['lr']) + '_k' + str(self._model_parameters['K']) + '.pth' ))

        self.pathLoadedSignal.emit('MODEL LOADED')        

    @model_parameters.setter
    def model_parameters(self, model_par):

        if isinstance(model_par, dict):
            self._model_parameters = model_par
            if not 'idx' in self._model_parameters.keys():
                self._model_parameters['idx'] = 0
        else:
            self._model_parameters[model_par[0]] = model_par[1]


    @load_path.setter
    def load_path(self, path):

        if path != '':
            self._load_path = path
            self._model_init, self._model_load = False, True
            splits = self._load_path.split("/")
            self.pathLoadedSignal.emit("MODEL LOADED FROM : " + splits[-1])
        else:
            self._load_path = ''
            self._model_load = False
            self.pathLoadedSignal.emit('')

    @preferences.setter
    def preferences(self, slot):
        self._preferences = slot
        self.preferenceChangedSignal.emit()

    @pyqtSlot(object)
    def set_newWindow(self, window):
        self.changeWindowSignal.emit(window)

    @pyqtSlot(list)
    def set_timerSpeed(self, slot):
        if slot[1] < 0:
            self._speed += slot[0]
            self._currentInterval = self._currentInterval + slot[1]
            self._timer_dx.setInterval(self._currentInterval)
            self._timer_sx.setInterval(self._currentInterval)
        else:
            self._currentInterval = slot[1]
            self._speed = slot[0]

        self.setSpeedSignal.emit(str(self.speed))

    @pyqtSlot(list)
    def refreshAnnotationBuffer(self, annotation):
        c = 0
        
        for el in annotation:

            
            triple = self._annotation_buffer[int(el[0]) - c]
            self._annotation_buffer.pop(int(el[0]) - c)
            c += 1
            
            
            self.clips.append({ 'clip' : triple[0], 'path' : el[1]})
            self.clips.append({ 'clip' : triple[1], 'path' : el[2]})
            self.disp_figure.append(self._annotator.reload_figure(self._history_database, el[1]))
            self.disp_figure.append(self._annotator.reload_figure(self._history_database, el[2]))
            

    
        
