import os
import sys
import csv
import time

from PyQt5.QtCore import QThread

def save_annotation(save_path, annotation_buffer):
    for i, triple in enumerate(annotation_buffer):
            with open(save_path + '/annotation_buffer/triple_' + str(i) + '.csv', 'w') as csvfile:
                filewriter = csv.writer(csvfile)
                for idx, clip in enumerate(triple[0]):
        
                    filewriter.writerow([clip, triple[1][idx], triple[2]])

class ClipsThread(QThread):

    def __init__(self, model):
        super().__init__()

        self._model = model


    def run(self):

        self._model.logBarDxSignal.emit('Wait clips to annotate')
        while(not self._model.annotate):
            continue
        
        for clips_folder in os.listdir(self._model._clips_database):

            clips, disp_figure = self._model._annotator.load_clips_figure(self._model._clips_database, clips_folder)
            for idx in range(len(disp_figure), 2):
                self._model.updateDisplayImages(disp_figure[idx], disp_figure[idx + 1])
                self._model.choiseButton = True

                while(self._model._preferencies == None):
                    self._model.logBarDxSignal.emit('Waiting annotation')

                self._model._annotation_buffer.append([clips[idx]['clip'], clips[idx + 1]['clip'], self._model._preferencies])
                annotation = [clips[idx]['path'], clips[idx+ 1]['path'], '[' + str(self._model._preferencies[0]) + ',' + str(self._model._preferencies[1]) + ']']
                self._model.updateHistoryList(annotation)
                self._model.choiseButton = False
                save_annotation(self._model._auto_save_foder, self._model._annotation_buffer)


            