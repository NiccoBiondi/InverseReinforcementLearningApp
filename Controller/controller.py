import gym
import gym_minigrid

import os
import gc 
import sys 
import torch
import time
import shutil
from datetime import date
import matplotlib.pyplot as plt

from View.AlgView import AlgView

from Thread.PolicyThread import PolicyThread
from Thread.RewardThread import RewardThread

from ReinforcementLearning.Oracle import Oracle
from ReinforcementLearning.csvRewardModel import save_reward_weights
from ReinforcementLearning.wrapper import RGBImgObsWrapper, FullyObsWrapper

from Utility.utility import save_annotation
from Utility.utility import save_model
from Utility.utility import save_model_parameters

from Utility.utility import load_annotation_point
from Utility.utility import load_values
from Utility.utility import load_losses

from PyQt5.QtCore import QObject, pyqtSlot, QEventLoop, QTimer
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox


DIR_NAME = os.path.dirname(os.path.abspath('__file__'))


class Controller(QObject):
    def __init__(self, model):
        super().__init__()

        self._model = model

        # Define threads and connect the policy thread to annotation function.
        self._policy_t = PolicyThread(self._model)
        self._policy_t._signals.finishedSignal.connect(lambda : self.annotation())
        self._reward_t = RewardThread(self._model)
        self._reward_t._signals.finishedSignal.connect(lambda : self._reward_t.quit())

        self._clips_number = 0      # count the number of clips to be annotated


    # Simple function to save the reward and policy graphic loss
    def save_graphic_loss(self, name):
        if self._model.model_parameters.keys():
            
            if not os.path.exists('Graphic_Images/' + self._model.model_parameters['minigrid_env']):
                os.makedirs('Graphic_Images/' + self._model.model_parameters['minigrid_env'])
            
            if 'reward' in name:
                plt.plot([i for i in range(len(self._model.reward_loss))], self._model.reward_loss)
                plt.xlabel('Iterations')
                
            else:
                plt.plot([i for i in range(len(self._model.policy_loss))], self._model.policy_loss)
                plt.xlabel('Agent Step')

            plt.ylabel('Loss')
            plt.title(name.split('.')[0])
            save_path ='Graphic_Images/' + self._model.model_parameters['minigrid_env'] + '/' + name
            plt.savefig(save_path)
            plt.show()

    # Funtion to take the oracle preferecies between pairs of clips
    # it starts/stops when the user check the Oracle Checkbox
    def setOraclePreferences(self):
        self._model.oracle_timer.stop()
        if len(self._model.clips) > 0:
            self._model.preferences = self._model.oracle.takeReward(self._model.clips_database, self._model.clips[0], self._model.clips[1], self._model.env)
        
    # This function connect the SetupDialog model with main model.
    # It transfers the parameters setted in setup  window.
    @pyqtSlot(dict)
    def init_values(self, values):
        self._model.model_parameters = values

    # Simple function to create a folder where the user can save policy,
    # reward model weight, annotation buffer and model parameters.
    @pyqtSlot()
    def save_action(self):
        file_name = ''
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getSaveFileName(caption="Define folder where to save element", directory=DIR_NAME + "/SAVE_FOLDER/", options=options)
        if fileName:
            save_path = fileName[0] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_model(save_path, self._model.policy, self._model.model_parameters, self._model.iteration, self._model.policy_loss)
            save_reward_weights(self._model.reward_model, save_path, self._model.weigth_path, self._model.model_parameters['lr'], self._model.model_parameters['K'], self._model.reward_loss)

            if os.path.exists(self._model.auto_save_folder + '/annotation_buffer/') and not os.path.exists(save_path + '/annotation_buffer/'):
                shutil.copytree(self._model.auto_save_folder + '/annotation_buffer/', save_path + '/annotation_buffer/')
            elif os.path.exists(self._model.auto_save_folder + '/annotation_buffer/'):
                shutil.rmtree(save_path + '/annotation_buffer/')
                shutil.copytree(self._model.auto_save_folder + '/annotation_buffer/', save_path + '/annotation_buffer/')
                
            if len(self._model.annotation_buffer):
                save_annotation(save_path, self._model.annotation_buffer, self._model.ann_point)
            
    # Simple function connected to 'Load checkpoint' button.
    # Give the possibility to the user to select a checkpoint folder and load previous system parameters.
    # In particular the annotation buffer with all the anotations made (if it exists).
    @pyqtSlot()
    def set_loadPath(self):
        fileName = ''

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getExistingDirectory(caption="Choose checkpoint to load", directory=DIR_NAME + "/SAVE_FOLDER/", options=options)

        if fileName:

            if fileName != self._model.load_path:

                # load hyperparameters and set the environment
                if [path for path in os.listdir(fileName) if 'values' in path]:

                    self._model.model_parameters, self._model.iteration, self._model.process = load_values(fileName + [ '/' + path for path in os.listdir(fileName) if 'values' in path][0])
                    env = gym.make(self._model.model_parameters['minigrid_env'])
                    self._model.env = RGBImgObsWrapper(env)
                    self._model.grid_wrapper = FullyObsWrapper(env)
                    self._model.oracle = Oracle(self._model.grid_wrapper, env, self._model)
                    self._model.env.reset() 
                    self._model.optimizer_p = torch.optim.Adam(params=self._model.policy.parameters(), lr = 1e-3, weight_decay=0.01)
                    self._model.optimizer_r = torch.optim.Adam(params=self._model.reward_model.parameters(), lr = float(self._model.model_parameters['lr']), weight_decay=0.01)
                
                # load reward model weigth
                if [path for path in os.listdir(fileName) if 'csv_reward_weight' in path]:
                    self._model.reward_model.load_state_dict(torch.load( fileName + [ '/' + path for path in os.listdir(fileName) if 'csv_reward_weight' in path][0] ))

                # In this case the user, in previous works, has already train a reward model with the same parameters defined in the current loaded work.
                # So if in the loaded folder there isn't the reward model weigth (the policy doesn't finish or the user doesn't finish to annotate),
                # is checked in a reward model weigth folder if there is a weight. This prevents the training the reward model in the first iteration 
                # and allows the policy training in the first iteration.
                if len([path for path in os.listdir(fileName) if 'csv_reward_weight' in path]) == 0  and 'csv_reward_weight_lr' + str(self._model.model_parameters['lr']) + '_k' + str(self._model.model_parameters['K']) + '.pth' in os.listdir(self._model.weigth_path) :
                    self._model.reward_model.load_state_dict(torch.load( self._model.weigth_path + '/csv_reward_weight_lr' + str(self._model.model_parameters['lr']) + '_k' + str(self._model.model_parameters['K']) + '.pth' ))

                # load policy weigth
                if [path for path in os.listdir(fileName) if 'policy_weight' in path]:
                    self._model.policy.load_state_dict(torch.load( fileName + [ '/' + path for path in os.listdir(fileName) if 'policy_weight' in path][0] ))

                # load reward model losses 
                if 'reward_model_losses.csv' in os.listdir(fileName):
                    self._model.reward_loss = load_losses(fileName + '/reward_model_losses.csv')

                # load policy losses 
                if 'policy_losses.csv' in os.listdir(fileName):
                    self._model.policy_loss = load_losses(fileName + '/policy_losses.csv')

                if os.path.exists(fileName + '/annotation_buffer'):
                    self._model.ann_point = load_annotation_point(fileName + '/annotation_buffer/')
                
                # Set the default path and create them if not exist.
                self._model.auto_save_folder = DIR_NAME + '/SAVE_FOLDER/' + self._model.model_parameters['minigrid_env'] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
                self._model.clips_database = DIR_NAME + '/Clips_Database/' + self._model.model_parameters['minigrid_env'] 
                self._model.history_database = DIR_NAME + '/History_Database/' + self._model.model_parameters['minigrid_env'] 
                self._model.weigth_path = DIR_NAME +  '/ReinforcementLearning/reward_model_init_weight/' + self._model.model_parameters['minigrid_env']
                
                if not os.path.exists(self._model.auto_save_folder):
                    os.makedirs(self._model.auto_save_folder)

                if not os.path.exists(self._model.clips_database):
                    os.makedirs(self._model.clips_database)
                
                if not os.path.exists(self._model.history_database):
                    os.makedirs(self._model.history_database)

                if not os.path.exists(self._model.weigth_path):
                    os.makedirs(self._model.weigth_path)

                # if the loaded folder is not an autosave folder, then we create an autosave folder with the same elements
                if fileName != self._model.auto_save_folder:
                    shutil.rmtree(self._model.auto_save_folder)
                    shutil.copytree(fileName, self._model.auto_save_folder)
                
                self._model.load_path = fileName
                
    # This function define oracle button functionality.
    @pyqtSlot(bool)
    def change_oracle(self, slot):
        self._model.oracle_active = slot

    # Reset checkpoint button function.
    @pyqtSlot()
    def reset_loadPath(self):
        self._model.load_path = ''

    # start button function. When the button is pressed, the AlgView is created only if the user load or initialize the model.
    @pyqtSlot()
    def start_button(self):
        if not self._model.model_init and not self._model.model_load:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Error in choosing hyperparameters or load checkpoint")
            msg.setDetailedText("You have to choose if init parameters or load a checkpoint and then press start process")
            msg.exec_()
            
        else:
            
            # When the model is initialize, the hyperparameters are saved in autosave folder.
            if not [self._model.auto_save_folder + '/' + f for f in os.listdir(self._model.auto_save_folder) if 'values' in f]:
                save_model_parameters(self._model.auto_save_folder, self._model.model_parameters, 0, 0)

            self._model.set_newWindow(AlgView(self._model, self))

    # Speed button function. Set the timer interval to display clips image faster or slower.
    @pyqtSlot()
    def change_speed(self):
        if (self._model.speed + 1) % 4 == 0:
            self._model.set_timerSpeed([1, 350])
        else:
            self._model.set_timerSpeed([1, -75])

    # Choice button function. The four button (left, right, both, discard)
    # have different preferences which are given to the clips.
    # The preferences are the human annotation made by the user.
    @pyqtSlot(list)
    def changepreferences(self, pref):
        self._model.preferences = pref

    # Process button function, that implements one training protocol iteration.
    # It is composed by tra policy training, the annotation phase and the reward model training.
    @pyqtSlot()
    def process(self):

        self._model.logBarSxSignal.emit('')
        self._model.logBarDxSignal.emit('')
        self._model.processButton = False

        # Any time this function is called, we check how many times the user clicks the 
        # process button. This is usefull to decrement the number of clips to annotate
        # during the reward model train period. After 5 "process" the number of annotation
        # is decremented by 20 until its value does not 20.
        if self._model.process > 0 and self._model.process % 5 == 0 and self._model.ann_point == 0:
            if self._model.model_parameters['n_annotation'] > 20:
                self._model.model_parameters['n_annotation'] = ['n_annotation', self._model.model_parameters['n_annotation'] - 20]
                save_model_parameters(self._model.auto_save_folder, self._model.model_parameters, self._model.iteration, self._model.process)
                
        elif self._model.ann_point == 0:
            self._model.process += 1

        # Define the number of clips to annotate
        self._clips_number = int( ( ( len(os.listdir(self._model.clips_database)) + len(os.listdir(self._model.history_database)) ) * ( int( self._model.model_parameters['n_annotation'] ) / 100  ) )  / 2 )

        if self._model.iteration < int(self._model.model_parameters['episodes']):
            self._policy_t.done = False
            self._policy_t.start()

        elif self._model.ann_point < self._clips_number:
            self.annotation()
        
        else:
            self._reward_t.start()

    # Simple QEventLoop used to wait the choice button clicked event, in essence it waits for the human annotation.
    def wait_signal(self):
        loop = QEventLoop()
        self._model.preferenceChangedSignal.connect(loop.quit)
        loop.exec_()

    # Funtion that give to the user the possibility to annotate the clips. 
    # It adds the annotation to annotation buffer and to the history window.
    # At the end it starts the reward model thread.
    @pyqtSlot()
    def annotation(self):
        self._policy_t.quit()
        
        # Connect the oracle timer with function to take oracle preferences
        self._model.oracle_timer.timeout.connect(lambda : self.setOraclePreferences())
        
        # Define the number of clips to annotate
        self._clips_number = int( ( ( len(os.listdir(self._model.clips_database)) + len(os.listdir(self._model.history_database)) ) * ( int( self._model.model_parameters['n_annotation'] ) / 100  ) )  / 2 )

        for i in range(self._model.ann_point, self._clips_number):

            self._model.clips, self._model.disp_figure = self._model.annotator.load_clips_figure(self._model.clips_database)
            self._model.logBarSxSignal.emit( 'Annotation: ' + str(i) + '/' + str(self._clips_number) )

            while(len(self._model.disp_figure) > 0):
                
                self._model.display_imageLen = len(self._model.disp_figure[0])
                self._model.display_imageSx = self._model.disp_figure.pop(0)
                self._model.display_imageDx = self._model.disp_figure.pop(0)
                
                self.wait_signal()
                self._model.choiceButton = False
                
                clip_1 = self._model.clips.pop(0)
                clip_2 = self._model.clips.pop(0)

                try:

                    # A triple is a list where the first two elemens are clips (minigrid states list of len clip_len)
                    self._model.annotation_buffer.append([clip_1['clip'], clip_2['clip'], self._model.preferences])
                    
                    # An annotation is a list where the first two elements are the paths of the corresponding clips,
                    # the third element is the given label. 
                    
                    annotation = [str(len(self._model.annotation_buffer) - 1), clip_1["path"], clip_2["path"], '[' + str(self._model.preferences[0]) + ',' + str(self._model.preferences[1]) + ']']

                    if self._model.preferences != None:
                        if not os.path.exists(self._model.history_database + '/' + clip_1["path"]):
                            shutil.move(self._model.clips_database + '/' + clip_1["path"], self._model.history_database)

                        if not os.path.exists(self._model.history_database + '/' + clip_2["path"]):
                            shutil.move(self._model.clips_database + '/' + clip_2["path"], self._model.history_database)

                    self._model.annotation = annotation
                    
                except Exception as e:
                    print(e)
                    self._model.annotation_buffer = self._model.annotation_buffer[:-1]
                    save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, self._model.ann_point)
                    sys.exit()

                self._model.preferences = None
                gc.collect()

            # we save the annotation buffer every 80 annotations and we clear it
            if self._model.ann_point > 0 and self._model.ann_point % 80 == 0:
                save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, self._model.ann_point)
                self._model.annotation_buffer = []

            self._model.ann_point = self._model.ann_point + 1

        if len(self._model.annotation_buffer) > 0:
            save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, self._model.ann_point)
            
        self._model.logBarDxSignal.emit('Annotation phase finished')

        self._model.display_imageSx = []
        self._model.display_imageDx = []
        self._model.choiceButton = False

        self._reward_t.start()         
        
