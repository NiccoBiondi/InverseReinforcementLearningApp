import os
import sys
import shutil
import random
import numpy as np
from PIL import Image

sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))

from Utility.utility import read_csv_clips

# Simple utility class to reload figure and clips
class Annotator():

    # Reload figure for history window widget
    def reload_figure(self, data_path, path):
        img = []
        for figure in sorted(os.listdir(data_path + '/' + path)):
            if '.png' in figure:
                tmp = Image.open(data_path + '/' + path + '/' + figure)
                img.append(tmp.convert("RGB").resize((800, 800)))

        return img

    # reload clips from csv and images for annotation
    def load_clips_figure(self, data_path):

        clips = []
        disp_figure = []
        folders = random.choices(os.listdir(data_path), k=2)

        for folder in folders:
            tmp = []
            for idx, figure in enumerate(sorted(os.listdir(data_path + '/' + folder))):
                if '.png' in figure:
                    f = Image.open(data_path + '/' + folder + '/' + figure)
                    tmp.append(f.convert("RGB").resize((800, 800)))
                elif '.csv' in figure:
                    clips.append({ 'clip' : read_csv_clips(data_path + '/' + folder + '/' + figure), 'path' : folder})

            disp_figure.append(tmp)

        return clips, disp_figure

    # Reset the Clips Database folder  
    def reset_clips_database(self, data_path):
        shutil.rmtree(data_path)
        os.mkdir(data_path)

            
