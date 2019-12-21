import os 
import sys 
import torch
from datetime import date

from View.AlgView import AlgView

from Thread.PolicyThread import PolicyThread
from Thread.ClipsThread import ClipsThread

from Utility.utility import load_values, load_annotation_buffer

from PyQt5.QtCore import QObject, pyqtSlot 
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox

DIR_NAME = os.path.dirname(os.path.abspath('__file__'))


class Controller(QObject):
    def __init__(self, model):
        super().__init__()

        self._model = model

        # Define threads
        self._policy_t = PolicyThread(self._model)
        self._clips_t = ClipsThread(self._model)

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
                    self._model._reward_model.load_state_dict(torch.load( fileName + '/csv_reward_weight.pth' ))
                
                if 'policy_weight.pth' in os.listdir(fileName):
                    self._model._policy.load_state_dict(torch.load( fileName + '/policy_weight.pth' ))

                if 'values.csv' in os.listdir(fileName):
                    self._model._model_parameters = load_values(fileName + '/values.csv')
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

    #TODO: display image setter

    @pyqtSlot(list)
    def changePreferencies(self, pref):
        self._model.preferencies = pref

    @pyqtSlot()
    def process(self):
        self._model.processButton = False
        self._policy_t.start()
        self._clips_t.start()
