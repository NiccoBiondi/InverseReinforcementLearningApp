import os 
import sys 
import gym
import gym_minigrid
import torch
import time
import shutil
from datetime import date

from View.AlgView import AlgView

from Thread.PolicyThread import PolicyThread
from Thread.RewardThread import RewardThread

from ReinforcementLearning.csvRewardModel import save_reward_weights
from ReinforcementLearning.wrapper import RGBImgObsWrapper

from Utility.utility import save_annotation
from Utility.utility import save_model

from Utility.utility import load_values
from Utility.utility import load_annotation_buffer

from PyQt5.QtCore import QObject, pyqtSlot, QEventLoop
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox


DIR_NAME = os.path.dirname(os.path.abspath('__file__'))


class Controller(QObject):
    def __init__(self, model):
        super().__init__()

        self._model = model

        # Define threads and connect the policy thread to annotation function.
        self._policy_t = PolicyThread(self._model)
        self._policy_t._signals.startAnnotation.connect(lambda : self.annotation())
        self._reward_t = RewardThread(self._model)
        
    # This function connect the SetupDialog model with main model.
    # Transfer the parameters setted in setup  window.
    @pyqtSlot(dict)
    def init_values(self, values):
        self._model.model_parameters = values

    # Simple function to create a folder where the user can save policy and reward model weight,
    # annotation buffer and model parameters. 
    @pyqtSlot()
    def save_action(self):
        file_name = ''
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getSaveFileName(caption="Define folder to save element", directory=DIR_NAME + "/SAVE_FOLDER/", options=options)
        if fileName:
            save_path = fileName[0] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_model(save_path, self._model.policy, self._model.model_parameters, self._model.iteration)
            save_reward_weights(self._model.reward_model, save_path)
            if len(self._model.annotation_buffer):
                save_annotation(save_path, self._model.annotation_buffer, self._model.ann_point)
            
    # Simple function connect to 'Load checkpoint' button.
    # Give the possibility to the user to select a checkpoint work folder and
    # load reward model and policy weight, model parameters defined in checkpoint work and, if exist, the annotation
    # buffer with all the anotations made.
    @pyqtSlot()
    def set_loadPath(self):
        fileName = ''

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getExistingDirectory(caption="Choose checkpoint to load", directory=DIR_NAME + "/SAVE_FOLDER/", options=options)

        if fileName:
            if fileName != self._model.load_path:

                if 'csv_reward_weight.pth' in os.listdir(fileName):
                    self._model.reward_model.load_state_dict(torch.load( fileName + '/csv_reward_weight.pth' ))

                if 'policy_weight.pth' in os.listdir(fileName):
                    self._model.policy.load_state_dict(torch.load( fileName + '/policy_weight.pth' ))

                if 'values.csv' in os.listdir(fileName):
                    self._model.model_parameters, self._model.iteration = load_values(fileName + '/values.csv')
                    self._model.env = RGBImgObsWrapper(gym.make(self._model.model_parameters['minigrid_env']))
                    self._model.env.reset() 
                    self._model.auto_save_folder = self._model.auto_save_folder + self._model.model_parameters['minigrid_env'] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
                    self._model.clips_database = self._model.clips_database + self._model.model_parameters['minigrid_env']
                    self._model.optimizer_p = torch.optim.Adam(params=self._model.policy.parameters(), lr = float(self._model.model_parameters['lr']))
                    self._model.optimizer_r = torch.optim.Adam(params=self._model.reward_model.parameters(), lr = float(self._model.model_parameters['lr']), weight_decay=0.01)
                    
                if 'annotation_buffer' in os.listdir(fileName):
                    self._model.annotation_buffer, self._model.ann_point = load_annotation_buffer(fileName + '/annotation_buffer/')

                # Restart from where the user stop the annotation.
                # From the clips2annotate folder we take the folder index where the user stop the annotation.
                # The we initialize the model.folder with the clipsToannotate folder from index onwards.
                if len(os.listdir(self._model.clips_database)) > 0:
                    folders = sorted(os.listdir(self._model.clips_database))
                    #index = folders.index([f for f in sorted(folders) if str(self._model.ann_point) in f][0])                    
                    for f in folders[self._model.ann_point:]:
                        self._model.folder = f
                self._model.load_path = fileName
                
    # This function define oracle button functionality.
    @pyqtSlot(bool)
    def change_oracle(self, slot):
        self._model.oracle = slot

    # Reset checkpoint button function.
    @pyqtSlot()
    def reset_loadPath(self):
        self._model.load_path = ''

    # start button function. When the button is pressed,
    # the new view, AlgView, is created only if the user
    # load or initialize the model.
    @pyqtSlot()
    def start_button(self):
        if not self._model.model_init and not self._model.model_load:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Error in choosing hyperparameters or load checkpoint")
            msg.setDetailedText("You have to choose if init parameters or load a checkpoint and then press start process")
            msg.exec_()
            
        else:

            self._model.set_newWindow(AlgView(self._model, self))

    # Speed button function. Set the
    # timer interval to display clips image.
    @pyqtSlot()
    def change_speed(self):
        if (self._model.speed + 1) % 4 == 0:
            self._model.set_timerSpeed([1, 350])
        else:
            self._model.set_timerSpeed([1, -75])

    # Choice button function. The four button (left, right, both, discard)
    # have different preferences which are given to the clips.
    # The preferences are the annotation made by the user.
    @pyqtSlot(list)
    def changepreferences(self, pref):
        self._model.preferences = pref

    # Process button function.
    @pyqtSlot()
    def process(self):
        self._model.logBarSxSignal.emit('Policy processing...')
        self._policy_t.start()
        self._model.processButton = False

    # Simple QEventLoop used to wait the choice button clicked event.
    def wait_signal(self):
        loop = QEventLoop()
        self._model.preferenceChangedSignal.connect(loop.quit)
        loop.exec_()

    # Funtion that give to the user the possibility
    # to annotate the clips. It add the annotation
    # to annotation buffer and to the history window.
    # End the clips to be annotate and the policy thread,
    # it starts the reward model thread.
    @pyqtSlot()
    def annotation(self):
        
        folders = []
        index = 0     
        i = self._model.ann_point + 1
        while (len(self._model.folder) > 0):

            if len(self._model.folder) > 0:
                self._model.clips, self._model.disp_figure = self._model.annotator.load_clips_figure(self._model.clips_database, self._model.folder.pop())

                for idx in range(0, len(self._model.disp_figure), 2):

                    self._model.logBarDxSignal.emit( 'Folder ' + str(i) + '/' + str(self._model.model_parameters['n_annotation']) + ': remain ' + str(idx) + '/' + str(len(self._model.disp_figure)) )
                    self._model.display_imageLen = len(self._model.disp_figure[idx])
                    self._model.display_imageSx = self._model.disp_figure[idx]
                    self._model.display_imageDx = self._model.disp_figure[idx + 1]
                    self._model.choiceButton = True
                    self._model.logBarDxSignal.emit('Folder ' + str(i) + '/' + str(self._model.model_parameters['n_annotation']) + ': remain ' + str(idx) + '/' + str(len(self._model.disp_figure)) + '..Waiting annotation...')
                    self.wait_signal()
                    self._model.choiceButton = False

                    try:

                        self._model.annotation_buffer.append([self._model.clips[idx]['clip'], self._model.clips[idx + 1]['clip'], self._model.preferences])
                        annotation = [self._model.clips[idx]['path'], self._model.clips[idx + 1]['path'], '[' + str(self._model.preferences[0]) + ',' + str(self._model.preferences[1]) + ']']
                        self._model.annotation = annotation

                    except Exception as e:
                        self._model.annotation_buffer = self._model.annotation_buffer[:-1]
                        save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, self._model.ann_point)
                        sys.exit()

                    self._model.preferences = None

                self._model.ann_point = self._model.ann_point + 1
                save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, self._model.ann_point)
                i += 1
        
        self._policy_t._signals.finishedSignal.connect(lambda: self._reward_t.start())
            
