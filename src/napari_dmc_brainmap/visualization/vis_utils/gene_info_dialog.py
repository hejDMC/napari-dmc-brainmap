from qtpy.QtWidgets import QWidget, QDialog, QLabel, QPushButton, QLineEdit, QFileDialog, QVBoxLayout, QSpinBox
from napari_dmc_brainmap.utils.general_utils import split_to_list
from typing import List, Optional, Tuple

class GeneInfoDialog(QDialog):
    """
    Dialog for entering gene information, including file selection, gene list input, and rounding options.
    """
    def __init__(self, parent: Optional[QWidget] = None, only_gene: bool = False, round_expression: bool = False) -> None:
        """
        Initialize the GeneInfoDialog.

        Parameters:
            parent (Optional[QWidget]): Parent widget. Defaults to None.
            only_gene (bool): Whether to return only a single gene name input. Defaults to False.
            round_expression (bool): Whether to enable rounding of gene expression data. Defaults to False.
        """
        super().__init__(parent)
        self.setWindowTitle("Enter Gene Information")
        self.file_path = None
        self.only_gene = only_gene
        self.round_expression = round_expression
        self.setup_ui()


    def setup_ui(self) -> None:
        """
        Set up the user interface elements of the dialog.
        """
        layout = QVBoxLayout()

        # File upload
        self.file_label = QLabel("Gene Expression File:")
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setToolTip("Click to select a CSV file containing gene expression data. The file needs to "
                                    "contain gene expression as rows (per spot) an genes as columns, plus one column"
                                    "named 'spot_id' containing the spot ID.")
        layout.addWidget(self.file_label)
        layout.addWidget(self.file_button)

        # Gene List
        if self.only_gene:
            self.gene_list_label = QLabel("Name of Gene:")
            self.gene_list_input = QLineEdit()
            self.gene_list_input.setToolTip(
                "Enter name of gene")
        else:
            self.gene_list_label = QLabel("List of Genes (comma separated):")
            self.gene_list_input = QLineEdit()
            self.gene_list_input.setToolTip("Enter gene names separated by commas (e.g., GeneA,GeneB,GeneC) - no spaces!")
        layout.addWidget(self.gene_list_label)
        layout.addWidget(self.gene_list_input)
        if self.round_expression:
            self.round_label = QLabel("Round Gene Expression Data?")
            self.round_spinner = QSpinBox()
            self.round_spinner.setRange(0, 42)
            self.round_spinner.setValue(0)
            self.round_spinner.setToolTip(
                "Round expression data for visualization to x digits after the decimal point. Set to 0 for no rounding.")
            layout.addWidget(self.round_label)
            layout.addWidget(self.round_spinner)


        # OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def select_file(self) -> None:
        """
        Open a file dialog to select a gene expression file and update the file path.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Gene Expression File", "",
                                                   "CSV Files (*.csv);;All Files (*)")
        if file_name:
            self.file_label.setText(f"Selected File: {file_name}")
            self.file_path = file_name

    def get_gene_info(self) -> Tuple[Optional[str], List[str], Optional[int]]:
        """
        Retrieve the selected file path, gene list, and rounding value.

        Returns:
            Tuple[Optional[str], List[str], Optional[int]]:
                - File path of the selected gene expression file (or None if not selected).
                - List of gene names.
                - Rounding value (or None if rounding is not enabled).
        """
        round_value = self.round_spinner.value() if self.round_expression else None
        return self.file_path, split_to_list(self.gene_list_input.text()), round_value


#%%
