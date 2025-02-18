from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex


class PandasModel(QAbstractTableModel):
    def __init__(self, dataframe, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe
        # Initialize check states for each row, False means unchecked
        self.check_states = {i: False for i in range(len(self._dataframe))}

    def rowCount(self, parent=QModelIndex()):
        return len(self._dataframe) if not parent.isValid() else 0

    def columnCount(self, parent=QModelIndex()):
        # Add one for the checkbox column
        return len(self._dataframe.columns) + 1 if not parent.isValid() else 0

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole and index.column() < len(self._dataframe.columns):
            return str(self._dataframe.iloc[index.row(), index.column()])
        if role == Qt.CheckStateRole and index.column() == len(self._dataframe.columns):
            # Return the check state for the checkbox column
            return Qt.Checked if self.check_states[index.row()] else Qt.Unchecked
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if index.column() == len(self._dataframe.columns) and role == Qt.CheckStateRole:
            self.check_states[index.row()] = not self.check_states[index.row()]
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True
        return False

    def flags(self, index):
        flags = super().flags(index)
        if index.column() == len(self._dataframe.columns):
            # Add the flag for items in the checkbox column to be editable
            flags |= Qt.ItemIsEditable | Qt.ItemIsUserCheckable
        return flags

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self._dataframe.columns):
                    return str(self._dataframe.columns[section])
                else:
                    return "Select"  # Header for the checkbox column
            elif orientation == Qt.Vertical:
                return str(self._dataframe.index[section])
        return None
    
