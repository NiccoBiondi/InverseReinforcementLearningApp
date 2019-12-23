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

from Utility.utility import save_annotation
from Utility.utility import load_values
from Utility.utility import load_annotation_buffer

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
            if fileName != self._model.load_path:

                if 'csv_reward_weight.pth' in os.listdir(fileName):
                    self._model.reward_model.load_state_dict(torch.load( fileName + '/csv_reward_weight.pth' ))
                
                if 'policy_weight.pth' in os.listdir(fileName):
                    self._model.policy.load_state_dict(torch.load( fileName + '/policy_weight.pth' ))

                if 'values.csv' in os.listdir(fileName):
                    self._model.model_parameters, self._model.iteration = load_values(fileName + '/values.csv')
                    self._model.env = RGBImgObsWrapper(gym.make(self._model.model_parameters['minigrid_env']))
                    self._model.env.reset() 
                    self._model.auto_save_folder = self._model.auto_save_folder + self._model.model_parameters['minigrid_env'] + '_(' + date.today().strftime("%d-%m-%Y") + ')/'
                    print(self._model.auto_save_folder)
                if 'annotation_buffer' in os.listdir(fileName):
                    self._model.annotation_buffer, self._model.ann_point = load_annotation_buffer(fileName + '/annotation_buffer/')

        
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
            self._model.set_timerSpeed([1, 350])
        else:
            self._model.set_timerSpeed([1, -75])

    @pyqtSlot(list)
    def changePreferencies(self, pref):
        self._model.preferencies = pref

    @pyqtSlot()
    def process(self):
        print(self._model.model_parameters.keys())
        if self._model.iteration < int(self._model.model_parameters['episodes']):
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
        self._model.choiseButton = True
        loop = QEventLoop()
        self._model.preferenceChangedSignal.connect(loop.quit)
        loop.exec_()


    @pyqtSlot()
    def annotation(self):
        
        # riparto dal folder da cui mi ero fermato ad annotare
        folders = os.listdir(self._model.clips_database)
        folders = [folders[i] for i in range([i for i in range(len(folders)) if str(self._model.ann_point) in folders[i]][0], len(folders))] 

        for folder in folders:
    
            self._model.clips, self._model.disp_figure = self._model.annotator.load_clips_figure(self._model.clips_database, folder)

            for idx in range(0, len(self._model.disp_figure), 2):
            
                self._model.display_imageLen = len(self._model.disp_figure[idx])
                self._model.display_imageDx = self._model.disp_figure[idx]
                self._model.display_imageSx = self._model.disp_figure[idx + 1]
                
                self._model.logBarDxSignal.emit('Waiting annotation')
                self.wait_signal()
                self._model.choiseButton = False
                
                try:

                    self._model.annotation_buffer.append([self._model.clips[idx]['clip'], self._model.clips[idx + 1]['clip'], self._model.preferencies])
                    annotation = [self._model.clips[idx]['path'], self._model.clips[idx + 1]['path'], '[' + str(self._model.preferencies[0]) + ',' + str(self._model.preferencies[1]) + ']']
                    self._model.annotation = annotation

                except Exception:
                    print(Exception)
                
                finally:
                    sys.exit()
            
            self._model.ann_point = self._model.ann_point + 1
            save_annotation(self._model.auto_save_folder, self._model.annotation_buffer, self._model.ann_point)

        self._model.choiseButton = False
        self.process()
            
            
