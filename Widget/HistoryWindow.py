import csv

from Widget.DisplayWidget import Display
from Widget.ReplayButton import ReplayButton

from Build.Ui_AnnotationHistory import Ui_historyWindow
from Build.Ui_ReplayClipsWindow import Ui_ReplayClipsWindow

from Widget.ReplayClipsWindow import ReplayClipsWindow

from PyQt5.QtCore import Qt, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QDialog, QPushButton, QSizePolicy, QTreeWidgetItem

# Is a button to reproduce the clips associated with the button.
class HistoryWindowButton(QPushButton):
    
    def __init__(self, name, path):
        super().__init__()
        self.setText(name)

        self.window = ReplayClipsWindow(path + '/' + name)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.clicked.connect(self.window.exec_)

# Window that contain all the clips in annotation buffer with the correlated preferences
class HistoryWindow(QDialog):
    def __init__(self, model):
        super().__init__()
        
        self.ui = Ui_historyWindow()
        self.ui.setupUi(self)

        # Define path of minigrid environment
        self.data_path = model._clips_database

        # Define model and controller 
        self._model = HistoryWindowModel()
        self._controller = HistoryWindowController(self._model, model.refreshAnnotationBuffer, self.ui.annotationList)
        
        # Connect history window button
        self.ui.buttonBox.accepted.connect(self._controller.ok_button)
        self.ui.buttonBox.rejected.connect(lambda : self.close())

        # Connect window to principal model update
        model.setClipsHistorySignal.connect(self.add_annotation)

        # Connect qtree list to update of key-value hash element
        self.ui.annotationList.itemChanged.connect(self._controller.upload_selected_element)

        # Connect window with its model
        self._model.listUpdateSignal.connect(self.refreshTreeList)

    # Add new anotation to QTreeWidget
    @pyqtSlot(list)
    def add_annotation(self, new_item):
        item = QTreeWidgetItem()
        item.setCheckState(0, Qt.Unchecked)
        item.setText(1, new_item[0])
        item.setText(2, new_item[1])
        item.setText(3, new_item[2])
        item.setText(4, new_item[3])
        self.ui.annotationList.addTopLevelItem(item)
        self.ui.annotationList.setItemWidget(item, 2, HistoryWindowButton(new_item[1], self.data_path))
        self.ui.annotationList.setItemWidget(item, 3, HistoryWindowButton(new_item[2], self.data_path))
        

    # Refresh the QTreeWidget after it's modified
    @pyqtSlot(dict)
    def refreshTreeList(self, el_list):
        init_pos = 0
        i = 0
        items = [el_list[i][1] for i in el_list if el_list[i][0]]
        for it in items:

            itemIndex = self.ui.annotationList.indexOfTopLevelItem(it)
            self.ui.annotationList.takeTopLevelItem(itemIndex)
 
        for i in range(self.ui.annotationList.topLevelItemCount()):
            item = self.ui.annotationList.topLevelItem(i)

            if i == 0:
                init_pos = int(item.text(1))

            if [i for i in el_list.keys() if item.text(1) == el_list[i][1].text(1) and not el_list[i][0]]:
                item.setText(1, str(init_pos))
                item.setCheckState(0, Qt.Unchecked)
                init_pos += 1


class HistoryWindowModel(QObject):
    listUpdateSignal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # Define window utilities: the first define the element to add to annotation buffer
        # The second define the key-value hash to connect QtreeItems to annotation buffer items
        self._selected_element = {}
    
    @property
    def selected_element(self):
        return self._selected_element


    @selected_element.setter
    def selected_element(self, element):
        self._selected_element[element[0]] = element[1]

    

class HistoryWindowController(QObject):
    

    def __init__(self, model, configure, tree):
        super().__init__()

        # Define model and o_button configuration. 
        # When press ok, i give the selected element to principal model
        # Then i refresh the annotation buffer
        self._model = model
        self._on_configure = configure
        self._tree = tree

    # The ok button has to take the selected element from QTreeWidget,
    # remove them from QTree widget and from annotation buffer and put the
    # selected element in the clips list which have to be annotate.
    @pyqtSlot()
    def ok_button(self):

        identifier = [identifier for identifier in self._model.selected_element.keys() if self._model.selected_element[identifier][0]]
        idx = [self._model.selected_element[i][1].text(1) for i in identifier]
        sel_el = [ [ idx[i], self._model.selected_element[identifier[i]][1].text(2), self._model.selected_element[identifier[i]][1].text(3) ] for i in range(len(idx)) ]
        self._on_configure(sel_el)
        self._model.listUpdateSignal.emit(self._model.selected_element)
        self._model._selected_element = {} 

    # When a element is selected or deselected the 
    # selected_element variable is updated.
    @pyqtSlot(QTreeWidgetItem, int)
    def upload_selected_element(self, item, column):

        for i in range(self._tree.invisibleRootItem().childCount()):
            item = self._tree.invisibleRootItem().child(i)
            self._model.selected_element =  [str(id(item)), [True if item.checkState(0) == Qt.Checked else False, item]]
            
            








