import os
import sys
import cv2
import shutil

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Utility.utility import read_csv_clips

class Annotator():
    def __init__(self, data_path):

        # Clips database
        self.data_path = data_path

        self.clips = []
        self.disp_figure = []

    def get_path(self):
        return self.data_path

    def reload_figure(self, path):
        img = []
        for figure in sorted(os.listdir(self.data_path + '/' + path)):
            tmp = cv2.imread(self.data_path + '/' + path + '/' + figure)
            img.append(cv2.resize(tmp, (800, 700)))
        return img


    def choose_clips(self, idx):

        if len(self.clips) > 0:
            self.clips = []
            self.disp_figure = []

        
        folders = os.listdir(self.data_path)
        folder = [x for x in folders if str(idx) in x][0]
        for fig in os.listdir(self.data_path + '/' + folder):
            tmp = []
            for idx, figure in enumerate(sorted(os.listdir(self.data_path + '/' + folder + '/' + fig))):
                if '.png' in figure:
                    f = cv2.imread(self.data_path + '/' + folder + '/' + fig + '/' + figure)
                    tmp.append(cv2.resize(f, (800, 700)))
                elif '.csv' in figure:
                    self.clips.append({ 'clip' : read_csv_clips(self.data_path + '/' + folder + '/' + fig + '/' + figure), 'path' : folder + '/' + fig})

            self.disp_figure.append(tmp)
    
        return self.clips, self.disp_figure
    
    def reset_clips_database(self):
        shutil.rmtree(self.data_path)
        os.mkdir(self.data_path)

            
