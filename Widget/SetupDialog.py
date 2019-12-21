import os
import sys

from Build.Ui_SetupDialog import Ui_SetupModel

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject
from PyQt5.QtWidgets import QDialog

class SetupDialog(QDialog):
    def __init__(self, model):
        super().__init__()
        
        # Set the QTdesigner view
        self.ui = Ui_SetupModel()
        self.ui.setupUi(self)

        # Define model and controller
        self._model = SetupDialogModel(model)
        self._controller = SetupDialogController(self._model)

        # Connect button to controller
        self.ui.ok_button.clicked.connect(lambda : self._controller.ok_button())
        self.ui.defaultButton.clicked.connect(lambda : self._controller.set_default())

        # Connect view with model
        self._model.changeSettingSignal.connect(self.set_default)
        self._model.doneSignal.connect(self.close_Window)

        # Connect changing
        self.ui.minigrid_env.currentTextChanged.connect(self._controller.change_env) 
        self.ui.episode_len_line.textChanged.connect(self._controller.change_episode)
        self.ui.lr_line.textChanged.connect(self._controller.change_lr)
        self.ui.clips_len_line.textChanged.connect(self._controller.change_clips_len)
        self.ui.episodes_line.textChanged.connect(self._controller.change_episodes)
        self.ui.K_line.textChanged.connect(self._controller.change_K)



    @pyqtSlot(dict)
    def set_default(self, default_param):
        self.ui.minigrid_env.setCurrentIndex(1)
        self.ui.episode_len_line.setText(default_param['episode_len'])
        self.ui.lr_line.setText(default_param['lr'])
        self.ui.clips_len_line.setText(default_param['clips_len'])
        self.ui.episodes_line.setText(default_param['episodes'])
        self.ui.K_line.setText(default_param['K'])
    
    def close_Window(self):
        self.ui.ok_button.setEnabled(False)
        self.close

class SetupDialogModel(QObject):
    changeSettingSignal = pyqtSignal(dict)
    doneSignal = pyqtSignal()

    def __init__(self, model):
        super().__init__()
        
        self._model = model
        self._default_parameters = {}

    @property
    def default_parameters(self):
        return self._default_parameters
    
    @property
    def env(self):
        return self._default_parameters['minigrid_env']
    
    @property
    def episode_len(self):
        return self._default_parameters['episode_len']
    
    @property
    def lr(self):
        return self._default_parameters['lr']
    
    @property
    def clips_len(self):
        return self._default_parameters['clips_len']
    
    @property
    def episodes(self):
        return self._default_parameters['episodes']
    
    @property
    def K(self):
        return self._default_parameters['K']

    @env.setter
    def env(self, env):
        self._default_parameters['minigrid_env'] = env
        self._model.model_parameters = ['minigrid_env', env]

    @episode_len.setter
    def episode_len(self, episode_line):
        self._default_parameters['episode_len'] = episode_line
        self._model.model_parameters = ['episode_len', episode_line]

    @lr.setter
    def lr(self, lr):
        self._default_parameters['lr'] = lr
        self._model.model_parameters = ['lr', lr]

    @clips_len.setter
    def clips_len(self, clips_len):
        self._default_parameters['clips_len'] = clips_len
        self._model.model_parameters = ['clips_len', clips_len]

    @episodes.setter
    def episodes(self, episodes):
        self._default_parameters['episodes'] = episodes
        self._model.model_parameters = ['episodes', episodes]

    @K.setter
    def K(self, K):
        self._default_parameters['K'] = K
        self._model.model_parameters = ['K', K]
    
    @default_parameters.setter
    def default_parameters(self, params):
        for key in params.keys():
            self._default_parameters[key] = params[key]

        self.changeSettingSignal.emit(params)
        self._model.model_parameters = params

    @pyqtSlot()
    def done(self):
        self._model.model_init = True
        self.doneSignal.emit()


class SetupDialogController(QObject):
    def __init__(self, model):
        super().__init__()

        self._model = model

    @pyqtSlot()
    def set_default(self):
        default_param = {}
        default_param['minigrid_env'] = 'MiniGrid-Empty-6x6-v0'
        default_param['episode_len'] = str(80)
        default_param['lr'] = str(0.001)
        default_param['clips_len'] = str(5)
        default_param['episodes'] = str(50)
        default_param['K'] = str(100)
        self._model.default_parameters = default_param

    @pyqtSlot(str)
    def change_env(self, value):
        self._model.env = value

    @pyqtSlot(str)
    def change_episode(self, value):
        self._model.episode_line = value

    @pyqtSlot(str)
    def change_lr(self, value):
        self._model.lr = value

    @pyqtSlot(str)
    def change_clips_len(self, value):
        self.clips_len = value
    
    @pyqtSlot(str)
    def change_episodes(self, value):
        self._model.episodes = value
    
    @pyqtSlot(str)
    def change_K(self, value):
        self._model.K = value

    @pyqtSlot()
    def ok_button(self):
        self._model.done()

