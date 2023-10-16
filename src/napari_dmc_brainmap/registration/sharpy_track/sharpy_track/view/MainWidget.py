from PyQt5.QtWidgets import QSlider,QWidget,QGridLayout,QLabel,QGraphicsItemGroup
from PyQt5.QtCore import Qt
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.GraphicViewers import ViewerLeft,ViewerRight
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.ModeToggle import ModeToggle
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.DotObject import DotObject
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import predictPointSample

class MainWidget():
    def __init__(self, regViewer, regi_dict):
        self.widget = QWidget()
        self.layoutGrid = QGridLayout()
        self.widget.setLayout(self.layoutGrid)

        # add left viewer
        self.viewerLeft = ViewerLeft(regViewer)
        self.layoutGrid.addWidget(self.viewerLeft.view,1,1)
        self.viewerLeft.loadSlice(regViewer)
        # add right viewer
        self.viewerRight = ViewerRight(regViewer)
        self.layoutGrid.addWidget(self.viewerRight.view,1,3)
        self.viewerRight.loadSample(regViewer)
        # create volume slider

        self.create_z_slider(regViewer, regi_dict['xyz_dict']['z'][1])
        self.create_x_slider(regViewer, regi_dict['xyz_dict']['x'][1])
        self.create_y_slider(regViewer, regi_dict['xyz_dict']['y'][1])

    def create_z_slider(self, regViewer, z_size):
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(z_size - 1)
        self.z_slider.setSingleStep(1)
        self.z_slider.setSliderPosition(int(round(z_size / 2)))
        self.z_slider.valueChanged.connect(lambda: regViewer.status.z_changed(regViewer))
        self.layoutGrid.addWidget(self.z_slider, 2, 1)

    def create_x_slider(self, regViewer, x_size):
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(int(-x_size / 2))
        self.x_slider.setMaximum(int(x_size / 2))
        self.x_slider.setSingleStep(1)
        self.x_slider.valueChanged.connect(lambda: regViewer.status.x_changed(regViewer))
        self.layoutGrid.addWidget(self.x_slider, 0, 1)

    def create_y_slider(self, regViewer, y_size):
        self.y_slider = QSlider(Qt.Vertical)
        self.y_slider.setMinimum(int(-y_size / 2))
        self.y_slider.setMaximum(int(y_size / 2))
        self.y_slider.setSingleStep(1)
        self.y_slider.valueChanged.connect(lambda: regViewer.status.y_changed(regViewer))
        self.layoutGrid.addWidget(self.y_slider, 1, 0)

    def createSampleSlider(self,regViewer):
        self.sampleSlider = QSlider(Qt.Horizontal)
        self.sampleSlider.setMinimum(0)
        self.sampleSlider.setMaximum(regViewer.status.sliceNum-1)
        self.sampleSlider.setSingleStep(1)
        self.sampleSlider.valueChanged.connect(lambda: regViewer.status.sampleChanged(regViewer))
        self.layoutGrid.addWidget(self.sampleSlider,2,3)
    
    def createImageTitle(self,regViewer):
        self.imageTitle = QLabel()
        self.imageTitle.setText(str(regViewer.status.currentSliceNumber) +
                                '---'+regViewer.status.imgFileName[regViewer.status.currentSliceNumber])
        font = self.imageTitle.font()
        # adapt title fontscale
        font.setPointSize(int(regViewer.atlasModel.fontscale*20))
        self.imageTitle.setFont(font)
        self.imageTitle.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.layoutGrid.addWidget(self.imageTitle,0,3)
    
    def createTransformToggle(self,regViewer):
        self.toggle = ModeToggle()
        self.layoutGrid.addWidget(self.toggle,1,2)
        # link click to buttonstate
        self.toggle.clicked.connect(lambda: regViewer.status.toggleChanged(regViewer))
    
    def addDots(self,regViewer):
        # get clicked coordinates
        x_clicked, y_clicked = regViewer.status.pressPos.x(),regViewer.status.pressPos.y()
        # create DotObject inside itemGroup
        dotLeft = DotObject(x_clicked, y_clicked, int(10 * regViewer.status.scaleFactor))
        # predict dot at sample based on previous transformation
        if len(self.viewerLeft.itemGroup) >5 :
            x_predict,y_predict = predictPointSample(x_clicked,y_clicked,regViewer.atlasModel.rtransform)
            dotRight = DotObject(x_predict, y_predict, int(10 * regViewer.status.scaleFactor))
        else:
            dotRight = DotObject(x_clicked, y_clicked, int(10 * regViewer.status.scaleFactor))

        dotLeft.linkPairedDot(dotRight)
        dotRight.linkPairedDot(dotLeft)
        # add dots to scene
        self.viewerLeft.scene.addItem(dotLeft)
        self.viewerRight.scene.addItem(dotRight)
        # store dot to itemGroup
        self.viewerLeft.itemGroup.append(dotLeft) # add dot to leftViewer
        self.viewerRight.itemGroup.append(dotRight) # add dot to rightViewer
        # # check number of dots, if more than 5, do transformation
        # numDots = len(self.viewerLeft.itemGroup)
        # if numDots >= 5:
        #     regViewer.atlasModel.updateTransform(regViewer)
        # else:
        #     pass

    
    def removeRecentDot(self):
        itemGroupL = self.viewerLeft.itemGroup
        itemGroupR = self.viewerRight.itemGroup

        if len(itemGroupL) == 0:
            print("There's no point on the screen!")

        else:
            # remove dots from scene
            self.viewerLeft.scene.removeItem(self.viewerLeft.itemGroup[-1])
            self.viewerRight.scene.removeItem(self.viewerRight.itemGroup[-1])
            # remove dots from itemGroup storage
            self.viewerLeft.itemGroup = self.viewerLeft.itemGroup[:-1]
            self.viewerRight.itemGroup = self.viewerRight.itemGroup[:-1]
        




        



    



        
    
