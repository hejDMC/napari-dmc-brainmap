from PyQt5.QtWidgets import QSlider,QWidget,QGridLayout,QLabel,QGraphicsItemGroup
from PyQt5.QtCore import Qt
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.GraphicViewers import ViewerLeft,ViewerRight
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.ModeToggle import ModeToggle
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.DotObject import DotObject
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import predictPointSample

class MainWidget(QWidget):
    def __init__(self, regViewer):
        super().__init__()
        self.regViewer = regViewer
        self.layoutGrid = QGridLayout()
        self.setLayout(self.layoutGrid)

        # add left viewer
        self.viewerLeft = ViewerLeft(regViewer)
        self.layoutGrid.addWidget(self.viewerLeft.view,1,1)
        # add right viewer
        self.viewerRight = ViewerRight(regViewer)
        self.layoutGrid.addWidget(self.viewerRight.view,1,3)
        # create volume slider
        self.create_z_slider()
        self.create_x_slider()
        self.create_y_slider()
    
    def connect_actions(self):
        self.z_slider.valueChanged.connect(self.regViewer.status.z_changed)
        self.regViewer.status.z_changed()
        self.x_slider.valueChanged.connect(self.regViewer.status.x_changed)
        self.regViewer.status.x_changed()
        self.y_slider.valueChanged.connect(self.regViewer.status.y_changed)
        self.regViewer.status.y_changed()
        self.sampleSlider.valueChanged.connect(self.regViewer.status.sampleChanged)
        self.regViewer.status.sampleChanged()
        self.toggle.clicked.connect(self.regViewer.status.toggleChanged)
        self.regViewer.status.toggleChanged()
        
    def create_z_slider(self):
        z_size = self.regViewer.regi_dict['xyz_dict']['z'][1]
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(z_size - 1)
        self.z_slider.setSingleStep(1)
        self.z_slider.setSliderPosition(int(round(z_size / 2)))
        
        self.layoutGrid.addWidget(self.z_slider, 2, 1)

    def create_x_slider(self):
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(-450) # +-45 degrees should be enough
        self.x_slider.setMaximum(450)
        self.x_slider.setSingleStep(1)
        
        self.layoutGrid.addWidget(self.x_slider, 0, 1)

    def create_y_slider(self):
        self.y_slider = QSlider(Qt.Vertical)
        self.y_slider.setMinimum(-450)
        self.y_slider.setMaximum(450)
        self.y_slider.setSingleStep(1)
        
        self.layoutGrid.addWidget(self.y_slider, 1, 0)

    def createSampleSlider(self):
        self.sampleSlider = QSlider(Qt.Horizontal)
        self.sampleSlider.setMinimum(0)
        self.sampleSlider.setMaximum(self.regViewer.status.sliceNum-1)
        self.sampleSlider.setSingleStep(1)
        
        self.layoutGrid.addWidget(self.sampleSlider,2,3)
    
    def createImageTitle(self):
        self.imageTitle = QLabel()
        self.imageTitle.setText(str(self.regViewer.status.currentSliceNumber) +
                                '---'+self.regViewer.status.imgFileName[self.regViewer.status.currentSliceNumber])
        font = self.imageTitle.font()
        # adapt title fontscale
        font.setPointSize(int(self.regViewer.atlasModel.fontscale*20))
        self.imageTitle.setFont(font)
        self.imageTitle.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.layoutGrid.addWidget(self.imageTitle,0,3)
    
    def createTransformToggle(self):
        self.toggle = ModeToggle()
        self.layoutGrid.addWidget(self.toggle,1,2)
        # link click to buttonstate
        
    
    def addDots(self):
        # get clicked coordinates
        x_clicked, y_clicked = self.regViewer.status.pressPos.x(),self.regViewer.status.pressPos.y()
        # create DotObject inside itemGroup
        dotLeft = DotObject(x_clicked, y_clicked, 
                            self.regViewer.dotRR)
        
        # predict dot at sample based on previous transformation
        if len(self.viewerLeft.itemGroup) >5 :
            x_predict,y_predict = predictPointSample(x_clicked,y_clicked,self.regViewer.atlasModel.rtransform)
            # if projection on the right is outside of viewer, use clicked position
            if (
                x_predict < 0)|(
                x_predict > self.viewerRight.scene.width() - 1)|(
                y_predict < 0)|(
                y_predict > self.viewerRight.scene.height() - 1):
                dotRight = DotObject(x_clicked, y_clicked, self.regViewer.dotRR)
            else: # do prediction
                dotRight = DotObject(x_predict, y_predict, self.regViewer.dotRR)
        else:
            dotRight = DotObject(x_clicked, y_clicked, self.regViewer.dotRR)

        dotLeft.linkPairedDot(dotRight)
        dotRight.linkPairedDot(dotLeft)
        # add dots to scene
        self.viewerLeft.scene.addItem(dotLeft)
        self.viewerRight.scene.addItem(dotRight)
        # store dot to itemGroup
        self.viewerLeft.itemGroup.append(dotLeft) # add dot to leftViewer
        self.viewerRight.itemGroup.append(dotRight) # add dot to rightViewer

    
    def removeRecentDot(self):
        itemGroupL = self.viewerLeft.itemGroup

        if len(itemGroupL) == 0:
            print("There's no point on the screen!")
        else:
            # remove dots from scene
            self.viewerLeft.scene.removeItem(self.viewerLeft.itemGroup[-1])
            self.viewerRight.scene.removeItem(self.viewerRight.itemGroup[-1])
            # remove dots from itemGroup storage
            self.viewerLeft.itemGroup = self.viewerLeft.itemGroup[:-1]
            self.viewerRight.itemGroup = self.viewerRight.itemGroup[:-1]
        




        



    



        
    
