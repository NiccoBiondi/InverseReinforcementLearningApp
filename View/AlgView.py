import sys
import os

import numpy as np 

from Widget.DisplayWidget import Display
from Widget.ReplayButton import ReplayButton
from Widget.ChoiceButtonWidget import ChoiceButton
from Widget.ProcessButton import ProcessButton
from Widget.OracleButton import OracleButton
from Widget.HistoryButton import HistoryButton
from Widget.SpeedButton import SpeedButton
from Widget.LogWidget import LogWidget

from PyQt5.QtCore import  Qt, QTimer
from PyQt5.QtWidgets import QWidget, QGridLayout, QSizePolicy, QVBoxLayout

# Widget that contains a number of CounterButtons arranged in a GridLayout.
class ButtonPane(QGridLayout):
    
    def __init__(self, buttons, cols=5, **kwargs):
        super().__init__(**kwargs)


        for i in range(len(buttons)):
            self.addWidget(buttons[i], i // cols, i % cols)

# Widget that contains a number of dislay arranged in a GridLayout.
class VideoPane(QGridLayout):
    
    def __init__(self, model, cols=2, **kwargs):
        super().__init__(**kwargs)

        display = [Display(model.timer_sx, model, model.updateDisplaySxImageSignal), Display(model.timer_dx, model, model.updateDisplayDxImageSignal), ReplayButton(model.timer_sx, model), ReplayButton(model.timer_dx, model)]

        for i in range(len(display)):
            self.addWidget(display[i], i // cols, i % cols)


# Define the view for policy work, annotation work and reward model train.
class AlgView(QWidget):
    def __init__(self, model, controller):
        super().__init__()

        # Define model and controller        
        self._model = model
        self._controller = controller

        # Define choice button configuration
        self._choiceButtonConf = {'left' : [1, 0], 'right' : [0, 1], 'both' : [0.5, 0.5], 'discard' : [0, 0]}

    # Simple funtion that create the AlgView layout
    def createLayout(self):
        main_layout = QVBoxLayout()

        upper_widget = QWidget()
        upper_widget.setLayout(VideoPane(self._model))

        mid_widget = QWidget()
        buttons = []

        for key in self._choiceButtonConf.keys():
            buttons.append(ChoiceButton(key, self._choiceButtonConf[key], self._controller.changepreferences, self._model))

        mid_widget.setLayout(ButtonPane(buttons))
        mid_widget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        buttom_widget = QWidget()
        b = [ProcessButton(self._controller.process, self._model), OracleButton(self._model, self._controller.change_oracle), HistoryButton(self._model), 
            SpeedButton(self._controller.change_speed, self._model), LogWidget(self._model.logBarSxSignal), LogWidget(self._model.logBarDxSignal, text='')]

        buttom_widget.setLayout(ButtonPane(b, cols = 4))
        buttom_widget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        main_layout.addWidget(upper_widget)
        main_layout.addWidget(mid_widget)
        main_layout.addWidget(buttom_widget)

        return main_layout

    