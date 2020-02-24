# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ReplayClipsWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ReplayClipsWindow(object):
    def setupUi(self, ReplayClipsWindow):
        ReplayClipsWindow.setObjectName("ReplayClipsWindow")
        ReplayClipsWindow.resize(979, 885)
        self.verticalLayout = QtWidgets.QVBoxLayout(ReplayClipsWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.startReplay = QtWidgets.QPushButton(ReplayClipsWindow)
        self.startReplay.setObjectName("startReplay")
        self.verticalLayout.addWidget(self.startReplay)

        self.retranslateUi(ReplayClipsWindow)
        QtCore.QMetaObject.connectSlotsByName(ReplayClipsWindow)

    def retranslateUi(self, ReplayClipsWindow):
        _translate = QtCore.QCoreApplication.translate
        ReplayClipsWindow.setWindowTitle(_translate("ReplayClipsWindow", "Replay Clips Window"))
        self.startReplay.setText(_translate("ReplayClipsWindow", "Start the replay"))

