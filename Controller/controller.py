import gym
import gym_minigrid

import os
import gc 
import sys 
import torch
import time
import shutil
from datetime import date

from View.AlgView import AlgView

from Thread.PolicyThread import PolicyThread
from Thread.RewardThread import RewardThread

from ReinforcementLearning.Oracle import Oracle
from ReinforcementLearning.csvRewardModel import save_reward_weights
from ReinforcementLearning.wrapper import RGBImgObsWrapper, FullyObsWrapper

from Utility.utility import save_annotation
from Utility.utility import save_model
from Utility.utility import save_model_parameters

from Utility.utility import load_values
from Utility.utility import load_annotation_buffer

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

        

    # Simple function that when the timer end set the preferencies
    # like the oracle predict..(frase a caso)
    def setOraclePreferencies(self):
        self._model.oracle_timer.stop()
        self._model.preferences = self._model.oracle.takeReward(self._model.clips_database, self._model.clips[0], self._model.clips[1], self._model.env)
        
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
        fileName = QFileDialog.getSaveFileName(caption="Define folder where to save element", directory=DIR_NAME + "/SAVE_FOLDER/", options=options)
        if fileName:
            save_path = fileName[0] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            save_model(save_path, self._model.policy, self._model.model_parameters, self._model.iteration)
            save_reward_weights(self._model.reward_model, save_path, self._model.weigth_path, self._model.model_parameters['lr'], self._model.model_parameters['K'])
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
                
                if [path for path in os.listdir(fileName) if 'csv_reward_weight' in path]:
                    self._model.reward_model.load_state_dict(torch.load( fileName + [ '/' + path for path in os.listdir(fileName) if 'csv_reward_weight' in path][0] ))

                if [path for path in os.listdir(fileName) if 'policy_weight' in path]:
                    self._model.policy.load_state_dict(torch.load( fileName + [ '/' + path for path in os.listdir(fileName) if 'policy_weight' in path][0] ))
                
                if [path for path in os.listdir(fileName) if 'values' in path]:

                    self._model.model_parameters, self._model.iteration = load_values(fileName + [ '/' + path for path in os.listdir(fileName) if 'values' in path][0])
                    env = gym.make(self._model.model_parameters['minigrid_env'])
                    self._model.env = RGBImgObsWrapper(env)
                    self._model.grid_wrapper = FullyObsWrapper(env)
                    self._model.oracle = Oracle(self._model.grid_wrapper, env)
                    self._model.env.reset() 
                    self._model.clips_database = self._model.clips_database + self._model.model_parameters['minigrid_env']
                    self._model.history_database = self._model.history_database + self._model.model_parameters['minigrid_env']
                    self._model.optimizer_p = torch.optim.Adam(params=self._model.policy.parameters(), lr = float(self._model.model_parameters['lr']))
                    self._model.optimizer_r = torch.optim.Adam(params=self._model.reward_model.parameters(), lr = float(self._model.model_parameters['lr']), weight_decay=0.01)
                    self._model.weigth_path = self._model.weigth_path + self._model.model_parameters['minigrid_env']
                    
                if [path for path in os.listdir(fileName) if 'annotation_buffer' in path]:
                    self._model.annotation_buffer, self._model.ann_point = load_annotation_buffer(fileName + [ '/' + path + '/' for path in os.listdir(fileName) if 'annotation_buffer' in path][0])
                    self._model.start_ann_disp = len(self._model.annotation_buffer)

                if len([path for path in os.listdir(fileName) if 'csv_reward_weight' in path]) == 0  and 'csv_reward_weight_lr' + str(self._model.model_parameters['lr']) + '_k' + str(self._model.model_parameters['K']) + '.pth' in os.listdir(self._model.weigth_path) :
                    self._model.reward_model.load_state_dict(torch.load( self._model.weigth_path + '/csv_reward_weight_lr' + str(self._model.model_parameters['lr']) + '_k' + str(self._model.model_parameters['K']) + '.pth' ))

                
                # Create the auto_save path. If the load path used to restore the previous work is the same of
                # the autosave_path variable, the folder is not reset.
                # Create the auto save folder for a specific minigrifd env. If this folder still exists, then i delete it.
                self._model.auto_save_folder = self._model.auto_save_folder + self._model.model_parameters['minigrid_env'] + '_(' + date.today().strftime("%d-%m-%Y") + ')'
                
                if not os.path.exists(self._model.auto_save_folder):
                    os.mkdir(self._model.auto_save_folder)
                    
                elif self._model.auto_save_folder != fileName:
                    shutil.rmtree(self._model.auto_save_folder)
                    os.mkdir(self._model.auto_save_folder)
                
                self._model.load_path = fileName
                
    # This function define oracle button functionality.
    @pyqtSlot(bool)
    def change_oracle(self, slot):
        self._model.oracle_active = slot

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
        self._policy_t.done = False
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
        self._policy_t.quit()
        
        # Connect the oracle timer with a function
        self._model.oracle_timer.timeout.connect(lambda : self.setOraclePreferencies())

        # Define the number of clips to annotate
        clips_number = int( ( ( len(os.listdir(self._model.clips_database)) + len(os.listdir(self._model.history_database)) ) * ( int( self._model.model_parameters['n_annotation'] ) / 100  ) )  / 2 )

        for i in range(self._model.ann_point, clips_number):

            self._model.clips, self._model.disp_figure = self._model.annotator.load_clips_figure(self._model.clips_database)
            self._model.logBarDxSignal.emit( 'Annotation: ' + str(i) + '/' + str(clips_number) )
            
            while(len(self._model.disp_figure) > 0):
                
                self._model.display_imageLen = len(self._model.disp_figure[0])
                self._model.display_imageSx = self._model.disp_figure.pop(0)
                self._model.display_imageDx = self._model.disp_figure.pop(0)


                self.wait_signal()
                self._model.choiceButton = False

                '''
                # If the oracle is not active then the user 
                # has to decide the best clip
                if not self._model.oracle_active:
                    self.wait_signal()
                    self._model.choiceButton = False

                #the oracle make the decision
                else:
                    self._timer.start()
                    self.wait_signal()

                
                # If the user actives the oracle and there is no preferencies, i make it
                if self._model.preferences == None and self._model.oracle_active:
                    self._model.preferences = self._model.oracle.takeReward(self._model.clips_database, self._model.clips[0], self._model.clips[1], self._model.env)

                # If the user disables the oracle and there is no preferencies, then the user
                # has to choose the clip
                elif self._model.preferences == None and not self._model.oracle_active:
                    self._model.choiceButton = True
                    self.wait_signal()
                '''
                

                clip_1 = self._model.clips.pop(0)
                clip_2 = self._model.clips.pop(0)

                try:

                    # A triple is a list where the first two elemens are clips (minigrid states list of len clip_len)
                    self._model.annotation_buffer.append([clip_1['clip'], clip_2['clip'], self._model.preferences])
                    
                    # An annotation is a list where the first two elements are the paths of the corresponding clips,
                    # the third element is the preferency made. 
                    
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
                    save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, i)
                    save_model_parameters(self._model.auto_save_folder, self._model.model_parameters, i)
                    sys.exit()

                self._model.preferences = None
                gc.collect()

            self._model.ann_point = self._model.ann_point + 1
            save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, i)

        self._model.logBarDxSignal.emit('Annotation phase finished')   

        self._reward_t.start()         
        