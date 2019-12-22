import sys
import os
import numpy as np
from datetime import date

import torch
import gym
import gym_minigrid


from ReinforcementLearning.csvRewardModel import csvRewardModel
from ReinforcementLearning.policy import Policy
from ReinforcementLearning.wrapper import RGBImgObsWrapper
from ReinforcementLearning.policy import run_episode, Loss, save_policy_weights

from Utility.ThreadUtility import clips_generator, save_clips, save_model
from Utility.annotator import Annotator

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QWidget

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))

class Model(QObject):
    refreshHistorySignal = pyqtSignal()
    processButtonVisiblitySignal = pyqtSignal()
    choiseButtonVisiblitySignal = pyqtSignal()
    preferenceChangedSignal = pyqtSignal()
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

        # Define botton visibility
        self._processButton = True
        self._choiseButton = False

        # Define if a initialize model
        self._model_init = False
        self._model_load = False

        # Define default path  
        self._weigth_path = DIR_NAME +  '/ReinforcementLearning/reward_model_init_weight'
        self._auto_save_folder = DIR_NAME + '/SAVEFOLDER/'
        self._clips_database = DIR_NAME + '/Clips_Database/'
        self._load_path = ''

        # Define variable to train policy and reward model
        self._annotation = None 
        self._annotation_buffer = []
        self._annotation_buffer_index = 0
        self._oracle = False

        # Define util variable
        self._iteration = 0 # memorize the episodes where the policy arrived
        self._ann_point = 0 # memorize the folder where the annotator arrive
        self._auto_save_clock_policy = 2000 
        self._annotator = Annotator()
        self._model_parameters = {}
        self._preferencies = None

        # Define variable used to the annotation phase
        self._clips = []
        self._disp_figure = []

        # Define Inverse Reinforcement Learning element
        self._obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
        self._act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
        self._inner_size = 64   # Number of neurons in two hidden layers.
        self._reward_batch = 16

        self._env = None 
        self._reward_model = csvRewardModel(obs_size = self._obs_size, inner_size = self._inner_size)
        self._policy = Policy(obs_size = self._obs_size, act_size = self._act_size, inner_size = self._inner_size)
        self._optimizer_p = None
        self._optimizer_r = None

        # Define the two Display and replay buttons timers
        self._timer_dx = QTimer()
        self._timer_dx.setInterval(400)
        self._timer_sx = QTimer() 
        self._timer_sx.setInterval(400)
        self._currentInterval = 400
        self._speed = 1
        self._display_imageDx = []
        self._display_imageSx = []
        self._display_imageLen = 0

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
    def choiseButton(self):
        return self._choiseButton

    @property
    def oracle(self):
        return self._oracle
    
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
    def preferencies(self):
        return self._preferencies

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

    @choiseButton.setter
    def choiseButton(self, val):
        self._choiseButton= val
        self.choiseButtonVisiblitySignal.emit()   

    @oracle.setter
    def oracle(self, slot):
        self._oracle = slot
        
    @model_init.setter
    def model_init(self, value):

        self._model_init = value
        self._model_load = not value
        self.load_path = ''

        # Init env
        self._env = RGBImgObsWrapper(gym.make(self._model_parameters['minigrid_env']))
        self._env.reset()
        self._auto_save_folder += self._model_parameters['minigrid_env'] + '_(' + date.today().strftime("%d/%m/%Y") + ')/'

        # load reward model starting weight if they exists reward model
        if os.path.exists(self._weigth_path + 'csv_reward_weght.pth'):
            self._reward_model.load_state_dict(torch.load( self._weigth_path + 'csv_reward_weght.pth' ))

        self._policy.cuda()
        self._reward_model.cuda()

        # Use the Adam optimizer.
        self._optimizer_p = torch.optim.Adam(params=self._policy.parameters(), lr = float(self._model_parameters['lr']))
        self._optimizer_r = torch.optim.Adam(params=self._reward_model.parameters(), lr = float(self._model_parameters['lr']), weight_decay=0.01)

        if not os.path.exists(DIR_NAME + '/Clips_Database/' + self._model_parameters['minigrid_env']):
            os.makedirs(DIR_NAME + '/Clips_Database/' + self._model_parameters['minigrid_env'])

        self._clips_database = DIR_NAME + '/Clips_Database/' + self._model_parameters['minigrid_env']
        self._annotator.reset_clips_database(self._clips_database)


    @model_parameters.setter
    def model_parameters(self, model_par):

        if isinstance(model_par, dict):
            self._model_parameters = model_par
        else:
            self._model_parameters[model_par[0]] = model_par[1]

        self._model_parameters['idx'] = 0

    @load_path.setter
    def load_path(self, path):

        if path != '':
            self._load_path = path
            self._model_init, self._model_load = False, True
            splits = self._load_path.split("/")
            print(splits[-1])
            self.pathLoadedSignal.emit(splits[-1])
        else:
            self._model_load = False
            self.pathLoadedSignal.emit('')

    @preferencies.setter
    def preferencies(self, slot):
        self._preferencies = slot
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
        for el in annotation:
            triple = self._annotation_buffer[el[0]]
            del self._annotation_buffer[el[0]]
            self.clips.insert(0, triple[0])
            self.clips.insert(0, triple[1])
            self.disp_figure.insert(0, self._annotator.reload_figure(self._clips_database, el[1]))
            self.disp_figure.insert(0, self._annotator.reload_figure(self._clips_database, el[2]))

        self.refreshHistorySignal.emit()

        
