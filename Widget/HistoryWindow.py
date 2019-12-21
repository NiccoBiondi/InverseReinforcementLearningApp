import csv

from Widget.DisplayWidget import Display
from Widget.ReplayButton import ReplayButton

from Build.Ui_AnnotationHistory import Ui_historyWindow
from Build.Ui_ReplayClipsWindow import Ui_ReplayClipsWindow

from Widget.ReplayClipsWindow import ReplayClipsWindow

from PyQt5.QtCore import Qt, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QDialog, QPushButton, QSizePolicy, QTreeWidgetItem

class HistoryWindowButton(QPushButton):
    
    def __init__(self, name, path):
        super().__init__()
        self.setText(name)
        self.window = ReplayClipsWindow(path + '/' + name)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.clicked.connect(self.window.exec_)

class HistoryWindow(QDialog):
    def __init__(self, model):
        super().__init__()

        # Define path of minigrid environment
        self.data_path = 'model.annotator.get_path()'

        # Define model and controller 
        self._model = HistoryWindowModel()
        self._controller = HistoryWindowController(self._model, model.refreshAnnotationBuffer)

        self.ui = Ui_historyWindow()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self._controller.ok_button)
        self.ui.buttonBox.rejected.connect(lambda : self.close())

        # Connect window to principal model update
        model.setClipsHistorySignal.connect(self.add_annotation)
        model.refreshHistorySignal.connect(self._controller.refresh)

        # Connect qtree list to update of key-value hash element
        self.ui.annotationList.currentItemChanged.connect(self._controller.update_annotation_list)
        self.ui.annotationList.itemChanged.connect(self._controller.upload_selected_element)

        # Connect window with its model
        self._model.listUpdateSignal.connect(self.refreshTreeList)

    @pyqtSlot(list)
    def add_annotation(self, new_item):
        item = QTreeWidgetItem()
        item.setCheckState(0, Qt.Unchecked)
        item.setText(3, new_item[0][2])
        self.ui.annotationList.addTopLevelItem(item)
        self.ui.annotationList.setItemWidget(item, 1, HistoryWindowButton(new_item[0][0], self.data_path))
        self.ui.annotationList.setItemWidget(item, 2, HistoryWindowButton(new_item[0][1], self.data_path))

    @pyqtSlot(list)
    def refreshTreeList(self, el_list):
        for el in el_list:
            itemIndex = self.ui.annotationList.indexOfTopLevelItem(el)
            self.ui.annotationList.takeTopLevelItem(itemIndex)


class HistoryWindowModel(QObject):
    listUpdateSignal = pyqtSignal(list)

    def __init__(self):
        super().__init__()

        # Define window utilities: the first define the element to add to annotation buffer
        # The second define the key-value hash to connect QtreeItems to annotation buffer items
        self._selected_element = []
        self._annotation_list = {}
    
    @property
    def selected_element(self):
        return self._selected_element

    @property
    def annotation_list(self):
        return self._annotation_list

    @annotation_list.setter
    def annotation_list(self, slot):
        self._annotation_list[slot[0]] = slot[1]

    @selected_element.setter
    def selected_element(self, element):
        if element in self.selected_element:
            self._selected_element.append(element)
        else:
            self._selected_element.remove(element)

    

class HistoryWindowController(QObject):
    
    treeRefreshSignal = pyqtSignal(list)

    def __init__(self, model, configure):
        super().__init__()

        # Define model and o_button configuration. 
        # When press ok, i give the selected element to principal model
        # Then i refresh the annotation buffer
        self._model = model
        self._on_configure = configure

    @pyqtSlot()
    def ok_button(self):
        sel_el = []
        for el in self._model.selected_element:
            identifier = str(id(el))
            sel_el.append([self._model.annotation_list[identifier], self._model.el[i].text(1), el[i].text(2)])
            
        self._on_configure(sel_el)
        self._model.treeRefreshSignal.emit(sel_el)

    @pyqtSlot(QTreeWidgetItem, QTreeWidgetItem)
    def update_annotation_list(self, current, previous):
        self._model.annotation_list = [str(id(current)), self._model.annotation_list[str(id(previous))] + 1]

    @pyqtSlot()
    def refresh(self):
        for i, key in enumerate(self._model.annotation_list.keys()):
            self._model.annotation_list = [key, i]

    @pyqtSlot(QTreeWidgetItem, int)
    def upload_selected_element(self, item, column):
        self._model.selected_element = item








