import os 
import sys 
import gym
import gym_minigrid
import torch
from datetime import date

from View.AlgView import AlgView

from Thread.PolicyWorker import PolicyWorker
from Thread.RewardModelWorker import RewardModelWorker

from ReinforcementLearning.wrapper import RGBImgObsWrapper

from Utility.ThreadUtility import save_annotation
from Utility.utility import load_values, load_annotation_buffer

from PyQt5.QtCore import QObject, pyqtSlot, QThreadPool, QEventLoop
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))


class Controller(QObject):
    def __init__(self, model):
        super().__init__()

        self._model = model

        # Define threads
        self._policy_t = PolicyWorker(self._model)
        self._treadpool = QThreadPool()

    @pyqtSlot(dict)
    def init_values(self, values):
        self._model.model_parameters = values

    @pyqtSlot()
    def set_loadPath(self):
        fileName = ''

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        dialog = QFileDialog()
        dialog.setOptions(options)
        
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setDirectory(DIR_NAME + "/SAVEFOLDER/")

        if dialog.exec_() == QDialog.Accepted:
            fileName = dialog.selectedFiles()[0] 
        
        if fileName:
            #FIXME: fix the iteration and ann folder load .....
            if fileName != self._model.load_path:

                if 'csv_reward_weight.pth' in os.listdir(fileName):
                    self._model._reward_model.load_state_dict(torch.load( fileName + '/csv_reward_weight.pth' ))
                
                if 'policy_weight.pth' in os.listdir(fileName):
                    self._model._policy.load_state_dict(torch.load( fileName + '/policy_weight.pth' ))

                if 'values.csv' in os.listdir(fileName):
                    self._model._model_parameters = load_values(fileName + '/values.csv')
                    self._model._env = RGBImgObsWrapper(gym.make(self._model_parameters['minigrid_env']))
                    self._model._env.reset() 
                    self._model._auto_save_foder += self._model.model_parameters['minigrid_env'] + date.today().strftime("%d/%m/%Y") + '/'
                    print(self._model._auto_save_foder)
                
                if 'annotation_buffer' in os.listdir(fileName):
                    self._model._annotation_buffer, self._model._iteration = load_annotation_buffer(fileName + '/annotation_buffer/')

                self._model.load_path = fileName
        
    @pyqtSlot(bool)
    def change_oracle(self, slot):
        self._model.oracle = slot

    @pyqtSlot()
    def reset_loadPath(self):
        self._model.load_path = ''


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

    @pyqtSlot()
    def change_speed(self):
        if (self._model.speed + 1) % 4 == 0:
            self._model.set_timerSpeed([1, 450])
        else:
            self._model.set_timerSpeed([1, -150])

    @pyqtSlot(list)
    def changePreferencies(self, pref):
        self._model.preferencies = pref

    @pyqtSlot()
    def process(self):
        if self._model._iteration < int(self._model._model_parameters['episodes']):
            self._model.logBarSxSignal.emit('Start the policy work')
            self._model.logBarDxSignal.emit('Wait for clips to annotate')
            self._policy_t._signals.finishedSignal.connect(self.annotation)
            self._model.processButton = False

            self._treadpool.start(self._policy_t)
        
        else:
            self._model.logBarDxSignal.emit('')
            self._model.logBarSxSignal.emit('Train reward model')
            reward_t = RewardModelWorker(self._model)
            
    def wait_signal(self):
        loop = QEventLoop()
        self._model.preferenceChangedSignal.connect(loop.quit)
        loop.exec_()


    @pyqtSlot()
    def annotation(self):
        
        # riparto dal folder da cui mi ero fermato ad annotare
        folders = os.listdir(self._model._clips_database)
        folders = [folders[i] for i in range([i for i in range(len(folders)) if str(self._model._ann_point) in folders[i]][0], len(folders))] 

        for folder in folders:
    
            clips, disp_figure = self._model._annotator.load_clips_figure(self._model._clips_database, folder)

            for idx in range(0, len(disp_figure), 2):
                self._model.choiseButton = True
            
                self._model.display_imageLen = len(disp_figure[idx])
                self._model.display_imageDx = disp_figure[idx]
                self._model.display_imageSx = disp_figure[idx + 1]
                
                self._model.logBarDxSignal.emit('Waiting annotation')
                self.wait_signal()

                self._model._annotation_buffer.append([clips[idx]['clip'], clips[idx + 1]['clip'], self._model._preferencies])
                annotation = [clips[idx]['path'], clips[idx + 1]['path'], '[' + str(self._model._preferencies[0]) + ',' + str(self._model._preferencies[1]) + ']']
                self._model.annotation = annotation
                self._model.choiseButton = False
            
            self._model._ann_point += 1
            save_annotation(self._model._auto_save_folder, self._model._annotation_buffer, self._model._ann_point)

        self._model.choiseButton = False
        self.process()
            
            
