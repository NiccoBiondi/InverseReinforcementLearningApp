# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AnnotationHistory.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_historyWindow(object):
    def setupUi(self, historyWindow):
        historyWindow.setObjectName("historyWindow")
        historyWindow.resize(1596, 1302)
        self.verticalLayout = QtWidgets.QVBoxLayout(historyWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.annotationList = QtWidgets.QTreeWidget(historyWindow)
        self.annotationList.setObjectName("annotationList")
        self.annotationList.headerItem().setTextAlignment(0, QtCore.Qt.AlignCenter)
        self.annotationList.headerItem().setTextAlignment(1, QtCore.Qt.AlignCenter)
        self.annotationList.headerItem().setTextAlignment(2, QtCore.Qt.AlignCenter)
        self.annotationList.headerItem().setTextAlignment(3, QtCore.Qt.AlignCenter)
        self.annotationList.headerItem().setTextAlignment(4, QtCore.Qt.AlignCenter)
        self.verticalLayout.addWidget(self.annotationList)
        self.buttonBox = QtWidgets.QDialogButtonBox(historyWindow)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(historyWindow)
        self.buttonBox.accepted.connect(historyWindow.accept)
        self.buttonBox.rejected.connect(historyWindow.reject)
        QtCore.QMetaObject.connectSlotsByName(historyWindow)

    def retranslateUi(self, historyWindow):
        _translate = QtCore.QCoreApplication.translate
        historyWindow.setWindowTitle(_translate("historyWindow", "Annotation History"))
        self.annotationList.headerItem().setText(0, _translate("historyWindow", "CheckBox"))
        self.annotationList.headerItem().setText(1, _translate("historyWindow", "List pos"))
        self.annotationList.headerItem().setText(2, _translate("historyWindow", "Clip 1"))
        self.annotationList.headerItem().setText(3, _translate("historyWindow", "Clip 2"))
        self.annotationList.headerItem().setText(4, _translate("historyWindow", "Preferencies"))
