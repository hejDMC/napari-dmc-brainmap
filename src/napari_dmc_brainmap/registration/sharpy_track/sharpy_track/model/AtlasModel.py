import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.DotObject import DotObject
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans
from napari_dmc_brainmap.preprocessing.preprocessing_tools import adjust_contrast, do_8bit
from napari_dmc_brainmap.utils import get_bregma, xyz_atlas_transform, coord_mm_transform, sort_ap_dv_ml


from pathlib import Path
from pkg_resources import resource_filename
from bg_atlasapi import BrainGlobeAtlas


class AtlasModel():
    def __init__(self, regi_dict) -> None:
        self.regi_dict = regi_dict
        self.sharpy_dir = Path(resource_filename("napari_dmc_brainmap", 'registration'))
        self.imgStack = None
        print("loading reference atlas...")
        self.atlas = BrainGlobeAtlas(self.regi_dict['atlas'])
        self.xyz_dict = regi_dict['xyz_dict']
        self.z_idx = self.atlas.space.axes_description.index(self.xyz_dict['z'][0])
        self.calculateImageGrid()
        self.loadTemplate()
        self.loadAnnot()
        self.loadStructureTree()


    def loadTemplate(self):
        print('loading template volume...')
        self.template = self.atlas.reference
        self.template = adjust_contrast(self.template, (0, self.template.max()))
        self.template = do_8bit(self.template)


    def loadAnnot(self):
        self.annot = self.atlas.annotation


    def loadStructureTree(self):
        self.sTree = self.atlas.structures
        self.bregma = get_bregma(self.regi_dict['atlas'])

    def calculateImageGrid(self):
        y = np.arange(self.xyz_dict['y'][1])
        x = np.arange(self.xyz_dict['x'][1])
        grid_x,grid_y = np.meshgrid(x, y)
        self.r_grid_x = grid_x.ravel()
        self.r_grid_y = grid_y.ravel()
        self.grid = np.stack([grid_y, grid_x], axis=2)
    
    def getContourIndex(self,regViewer):
        # check simple slice or angled slice
        # slice annotation volume, convert to int32 for contour detection
        if (regViewer.status.x_angle == 0) and (regViewer.status.y_angle == 0):
            z_coord = coord_mm_transform(regViewer.status.current_z, self.bregma[self.z_idx],
                                      self.atlas.space.resolution[self.z_idx], mm_to_coord=True)
            self.sliceAnnot = self.annot[z_coord, :, :].copy().astype(np.int32)
        else:
            self.sliceAnnot = self.annot[self.z_flat, self.r_grid_y, self.r_grid_x].reshape(self.xyz_dict['y'][1], self.xyz_dict['x'][1]).astype(np.int32)
        # get contours
        contours,_ = cv2.findContours(self.sliceAnnot, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # create canvas
        empty = np.zeros((self.xyz_dict['y'][1], self.xyz_dict['x'][1]),dtype=np.uint8)
        # draw contours on canvas
        self.outline = cv2.drawContours(empty,contours,-1,color=255) # grayscale, 8bit
        self.outline= cv2.cvtColor(self.outline, cv2.COLOR_GRAY2RGBA) # convert to RGBA
        self.outline[:,:,3][np.where(self.outline[:,:,0]==0)] = 0 # set black background transparent
    
    def displayContour(self,regViewer):
        regViewer.status.contour = 1 # set status contour active
        self.getContourIndex(regViewer)
        regViewer.widget.viewerLeft.showContourLabel(regViewer)


    def hideContour(self,regViewer):
        regViewer.status.contour = 0 # set status contour inactive
        regViewer.widget.viewerLeft.hideContourLabel()
    
    def treeFindArea(self,regViewer):
        y = regViewer.status.hoverY
        x = regViewer.status.hoverX
        z = int(self.z_mat[y, x])
        # get coordinates in mm
        tripled_coord = xyz_atlas_transform([x, y, z], self.regi_dict, self.atlas.space.axes_description)
        tripled_mm = coord_mm_transform(tripled_coord, self.bregma, self.atlas.space.resolution)

        tripled_mm_sorted = sort_ap_dv_ml(tripled_mm, self.atlas.space.axes_description)
        # from cursor position get annotation index
        structure_id = self.atlas.structure_from_coords(tripled_coord)
        if structure_id > 0:
            # get highlight area index
            activeArea = np.where(self.sliceAnnot == structure_id)
            # find name in sTree
            structureName = self.sTree.data[structure_id]['name']
            regViewer.widget.viewerLeft.highlightArea(regViewer,tripled_mm_sorted,activeArea,structureName)


    def getSlice(self,regViewer):
        if (regViewer.status.x_angle == 0) and (regViewer.status.y_angle == 0):
            self.simpleSlice(regViewer) # update simple slice
        else:
            self.angleSlice(regViewer) # update angled slice
        name_dict = {
            'ap': 'AP',
            'si': 'DV',
            'rl': 'ML'
        }
        z_str = name_dict[self.xyz_dict['z'][0]]
        x_str = name_dict[self.xyz_dict['x'][0]]
        y_str = name_dict[self.xyz_dict['y'][0]]
        cv2.putText(self.slice, z_str + " : " + str(regViewer.status.current_z), (950, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)  # todo positions need to be fixed
        cv2.putText(self.slice, x_str + " Angle: " + str(regViewer.status.x_angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
        cv2.putText(self.slice, y_str + " Angle: " + str(regViewer.status.y_angle), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)

        self.slice = cv2.resize(self.slice,(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1])) # resize to single window size
        self.sliceQimg = QImage(self.slice.data, self.slice.shape[1],self.slice.shape[0],self.slice.strides[0],QImage.Format_Grayscale8)

    def getSample(self,regViewer):
        if regViewer.status.sliceNum == 0:
            # self.sampleQimg = QImage(str(self.sharpy_dir.joinpath('sharpy_track','sharpy_track','images','empty.png')))
            self.sample = cv2.imread(str(self.sharpy_dir.joinpath('sharpy_track','sharpy_track','images','empty.png')),cv2.IMREAD_COLOR)
            self.sample = cv2.resize(self.sample,(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1]))
            self.sampleQimg = QImage(self.sample.data, self.sample.shape[1],self.sample.shape[0],self.sample.strides[0],QImage.Format_BGR888)
        else:
            self.sample = cv2.resize(self.imgStack[regViewer.status.currentSliceNumber],(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1]))
            self.sampleQimg = QImage(self.sample.data, self.sample.shape[1],self.sample.shape[0],self.sample.strides[0],QImage.Format_Grayscale8)

    def simpleSlice(self,regViewer):
        z_coord = coord_mm_transform(regViewer.status.current_z, self.bregma[self.z_idx],
                                  self.atlas.space.resolution[self.z_idx], mm_to_coord=True)
        self.slice = self.template[z_coord, :, :].copy()
        self.z_mat = np.full((self.xyz_dict['y'][1], self.xyz_dict['x'][1]), z_coord)
    
    def angleSlice(self,regViewer):
        # calculate from ML and DV angle, the plane of current slice
        x_shift = int(np.tan(np.deg2rad(regViewer.status.x_angle)) * (self.xyz_dict['x'][1] / 2))
        y_shift = int(np.tan(np.deg2rad(regViewer.status.y_angle)) * (self.xyz_dict['y'][1] / 2))
        z_coord = coord_mm_transform(regViewer.status.current_z, self.bregma[self.z_idx],
                                  self.atlas.space.resolution[self.z_idx], mm_to_coord=True)

        center = np.array([z_coord, (self.xyz_dict['y'][1] / 2), (self.xyz_dict['x'][1] / 2)])
        c_right = np.array([z_coord + x_shift, (self.xyz_dict['y'][1] / 2), (self.xyz_dict['x'][1] - 1)])
        c_top = np.array([z_coord - y_shift, 0, (self.xyz_dict['x'][1] / 2)])
        # calculate plane vector
        vec_1 = c_right-center
        vec_2 = c_top-center
        vec_n = np.cross(vec_1,vec_2)
        # calculate AP matrix
        z_mat = (-vec_n[1]*(self.grid[:,:,0]-center[1])-vec_n[2]*(self.grid[:,:,1]-center[2]))/vec_n[0] + center[0]
        z_flat = z_mat.astype(int).ravel() # flatten AP matrix
        # within-volume check
        outside_vol = np.argwhere((z_flat<0)|(z_flat>(self.xyz_dict['z'][1]-1))) # outside of volume index
        if outside_vol.size == 0: # if outside empty, inside of volume
            # index volume with z_mat and grid
            self.z_mat = z_mat # save AP plane for indexing structure information
            self.z_flat = z_flat # save current AP list to AtlasModel for getContourIndex
            self.slice = self.template[z_flat, self.r_grid_y, self.r_grid_x].reshape(self.xyz_dict['y'][1], self.xyz_dict['x'][1])
        else: # if not empty, show black image with warning
            self.slice = np.zeros((self.xyz_dict['y'][1], self.xyz_dict['x'][1]),dtype=np.uint8)
            cv2.putText(self.slice, "Slice out of volume!", (400,400), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
    
    def getStack(self,regViewer):
        self.imgStack = np.full((regViewer.status.sliceNum,self.xyz_dict['y'][1],self.xyz_dict['x'][1]),-1,dtype=np.uint8)
        # copy slices to stack
        for i in range(regViewer.status.sliceNum):
            full_path = Path(regViewer.status.folderPath).joinpath(regViewer.status.imgFileName[i])
            img_data = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
            self.imgStack[i, :, :] = img_data
        print(regViewer.status.sliceNum, "Slice(s) loaded")


    def updateDotPosition(self,regViewer):
        # ignore if less than 5 pairs of dots
        if len(regViewer.widget.viewerLeft.itemGroup) < 5:
            # check if has saved coodinates
            pass
        else: # refresh dot coodinate
            atlas_pts = [] 
            for dot in regViewer.widget.viewerLeft.itemGroup: # itemGroup to list
                atlas_pts.append([int(dot.pos().x() / regViewer.status.scaleFactor), int(dot.pos().y() / regViewer.status.scaleFactor)]) # scale coordinates
            sample_pts = []
            for dot in regViewer.widget.viewerRight.itemGroup: # itemGroup to list
                sample_pts.append([int(dot.pos().x() / regViewer.status.scaleFactor), int(dot.pos().y() / regViewer.status.scaleFactor)]) # scale coordinates
            # update dot record in dictionary
            regViewer.status.atlasDots[regViewer.status.currentSliceNumber] = atlas_pts
            regViewer.status.sampleDots[regViewer.status.currentSliceNumber] = sample_pts
            regViewer.status.saveRegistration()
            # apply transformation
            self.updateTransform(regViewer, np.array(atlas_pts) * regViewer.status.scaleFactor, np.array(sample_pts) * regViewer.status.scaleFactor) # scale coordinates
        
    def checkSaved(self,regViewer):
        # load exist dots if there is any
        regViewer.status.blendMode[regViewer.status.currentSliceNumber] = 1
        if not(regViewer.status.currentSliceNumber in regViewer.status.atlasLocation):
            pass
        elif not(regViewer.status.currentSliceNumber in regViewer.status.atlasDots):
            regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.sliceQimg))
        elif len(regViewer.status.atlasDots[regViewer.status.currentSliceNumber]) == 0:
            regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.sliceQimg))
        else:
            atlas_pts = regViewer.status.atlasDots[regViewer.status.currentSliceNumber] # read dictionary, create dot object
            sample_pts = regViewer.status.sampleDots[regViewer.status.currentSliceNumber]

            regViewer.status.x_angle = regViewer.status.atlasLocation[regViewer.status.currentSliceNumber][0] # read atlasLocation
            regViewer.status.y_angle = regViewer.status.atlasLocation[regViewer.status.currentSliceNumber][1]
            regViewer.status.current_z = regViewer.status.atlasLocation[regViewer.status.currentSliceNumber][2]
            regViewer.widget.viewerLeft.loadSlice(regViewer) # slice atlas

            for xyAtlas, xySample in zip(atlas_pts,sample_pts):
                dotLeft = DotObject(int(xyAtlas[0] * regViewer.status.scaleFactor), int(xyAtlas[1] * regViewer.status.scaleFactor), int(10 * regViewer.status.scaleFactor)) # list to itemGroup
                dotRight = DotObject(int(xySample[0] * regViewer.status.scaleFactor), int(xySample[1] * regViewer.status.scaleFactor), int(10 * regViewer.status.scaleFactor)) # list to itemGroup
                dotLeft.linkPairedDot(dotRight)
                dotRight.linkPairedDot(dotLeft)
                # add dots to scene
                regViewer.widget.viewerLeft.scene.addItem(dotLeft)
                regViewer.widget.viewerRight.scene.addItem(dotRight)
                # store dot to itemGroup
                regViewer.widget.viewerLeft.itemGroup.append(dotLeft) # add dot to leftViewer
                regViewer.widget.viewerRight.itemGroup.append(dotRight) # add dot to rightViewer

    def updateTransform(self,regViewer,atlas_pts,sample_pts):
        transform = fitGeoTrans(sample_pts,atlas_pts) # save transform for prediction
        self.rtransform = fitGeoTrans(atlas_pts,sample_pts)
        self.sampleWarp = cv2.warpPerspective(self.sample,transform,(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1]))
        self.sampleBlend = cv2.addWeighted(self.slice, 0.5, self.sampleWarp, 0.5, 0)

        self.qWarp = QImage(self.sampleWarp.data,self.sampleWarp.shape[1],self.sampleWarp.shape[0],self.sampleWarp.strides[0],QImage.Format_Grayscale8)
        self.qBlend = QImage(self.sampleBlend.data, self.sampleBlend.shape[1],self.sampleBlend.shape[0],self.sampleBlend.strides[0],QImage.Format_Grayscale8)
        if not(regViewer.status.currentSliceNumber in regViewer.status.blendMode):
            regViewer.status.blendMode[regViewer.status.currentSliceNumber] = 1 # overlay
        else:
            pass
        if regViewer.status.blendMode[regViewer.status.currentSliceNumber] == 0: # all atlas
            regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.sliceQimg))
        elif regViewer.status.blendMode[regViewer.status.currentSliceNumber] == 1: # overlay
            regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.qBlend))
        else:
            regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.qWarp)) # all sample

    
