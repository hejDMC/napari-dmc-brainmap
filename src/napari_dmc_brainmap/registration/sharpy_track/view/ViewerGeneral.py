from PyQt5.QtWidgets import QLabel,QGraphicsScene,QGraphicsView,QFrame

class ViewerGeneral():
    def __init__(self,regViewer) -> None:
        self.labelImg = QLabel()
        self.labelImg.setFixedSize(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1])
        self.scene = QGraphicsScene(0,0,regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1],parent=regViewer)
        self.scene.changed.connect(lambda: regViewer.atlasModel.updateDotPosition(regViewer)) # here update
        self.scene.addWidget(self.labelImg)
        self.itemGroup = [] # create itemGroup, store DotObjects
        self.view = QGraphicsView(self.scene)
        self.view.leaveEvent = lambda event: self.leaveLabel(regViewer)
        self.view.setFixedSize(1140,800)
        self.view.setSceneRect(0,0,1140,800)
        self.view.setFrameShape(QFrame.NoFrame)

    def leaveLabel(self,regViewer):
        regViewer.status.cursor = 0
        