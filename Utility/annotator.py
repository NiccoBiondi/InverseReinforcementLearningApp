import os
import sys
import cv2
import shutil

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Utility.utility import read_csv_clips

class Annotator():


    def get_path(self):
        return self.data_path


    def load_clips_figure(self, data_path, folder):

        clips = []
        disp_figure = []

        for fig in os.listdir(data_path + '/' + folder):
            tmp = []
            for idx, figure in enumerate(sorted(os.listdir(data_path + '/' + folder + '/' + fig))):
                if '.png' in figure:
                    f = cv2.imread(data_path + '/' + folder + '/' + fig + '/' + figure)
                    tmp.append(cv2.resize(f, (800, 700)))
                elif '.csv' in figure:
                    clips.append({ 'clip' : read_csv_clips(data_path + '/' + folder + '/' + fig + '/' + figure), 'path' : folder + '/' + fig})

            disp_figure.append(tmp)
    
        return clips, disp_figure
    
    def reset_clips_database(self, data_path):
        shutil.rmtree(data_path)
        os.mkdir(data_path)

            
