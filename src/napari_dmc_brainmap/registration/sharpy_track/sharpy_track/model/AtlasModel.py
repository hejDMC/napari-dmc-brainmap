import cv2
import numpy as np
from qtpy.QtGui import QImage, QPixmap
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.DotObject import DotObject
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.coordinates import (
    homography_to_display_space,
)
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.prediction import (
    RegistrationTransformPredictor,
    homography_to_registration_points,
)
# from napari_dmc_brainmap.preprocessing.preprocessing_tools import adjust_contrast, do_8bit
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, xyz_atlas_transform, coord_mm_transform, sort_ap_dv_ml
from napari_dmc_brainmap.utils.atlas_cache import load_reference_8bit

from importlib.resources import files
from pathlib import Path
from bg_atlasapi import BrainGlobeAtlas


class AtlasModel():
    def __init__(self, regViewer, predictor=None) -> None:
        self.regViewer = regViewer
        self.regi_dict = regViewer.regi_dict
        self.sharpy_dir = Path(files("napari_dmc_brainmap").joinpath("registration"))
        self.imgStack = None
        print("loading reference atlas...")
        self.atlas = BrainGlobeAtlas(self.regi_dict['atlas'])
        self.xyz_dict = self.regi_dict['xyz_dict']
        # adaptive fontsize
        self.fontscale = np.round(np.min([self.xyz_dict['x'][1],self.xyz_dict['y'][1]]) / 800 * self.regViewer.scaleFactor,1)
        self.fontthickness = np.rint(np.min([self.xyz_dict['x'][1],self.xyz_dict['y'][1]]) / 256 * self.regViewer.scaleFactor).astype(int)
        self.z_idx = self.atlas.space.axes_description.index(self.xyz_dict['z'][0])
        self.calculateImageGrid()
        self.loadTemplate()
        self.loadAnnot()
        self.loadStructureTree()
        self.atlas_pts = []
        self.sample_pts = []
        self.predictor = predictor
        self.slice_in_volume = True
        self.predictedSliceNumber = None
        self.predictedTransformNative = None
        self.predictedTransformDisplay = None


    def loadTemplate(self):
        print('checking template volume...')
        self.template = load_reference_8bit(self.atlas, notify=print)
        if not np.issubdtype(self.template.dtype, np.uint8):
            print(f"Data type for reference volume: {self.template.dtype}")
            print("8-bit / 16-bit grayscale volume is required.")
            print("Reference volume cannot be correctly loaded to RegistrationViewer!")
        
        ori_trans_dict = {"coronal": (0,1,2),
                          "horizontal": (1,0,2),
                          "sagittal": (2,0,1)}
        self.template = np.transpose(self.template,
                                     axes=ori_trans_dict[self.regi_dict["orientation"]])



    def loadAnnot(self):
        ori_trans_dict = {"coronal": (0,1,2),
                          "horizontal": (1,0,2),
                          "sagittal": (2,0,1)}
        self.annot = np.transpose(self.atlas.annotation,
                                  axes=ori_trans_dict[self.regi_dict["orientation"]])


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
    
    def getContourIndex(self):
        # check simple slice or angled slice
        # slice annotation volume, convert to int32 for contour detection
        if (self.regViewer.status.x_angle == 0) and (self.regViewer.status.y_angle == 0):
            z_coord = coord_mm_transform([self.regViewer.status.current_z], [self.bregma[self.z_idx]],
                                      [self.atlas.space.resolution[self.z_idx]], mm_to_coord=True)
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
    
    def displayContour(self):
        self.regViewer.status.contour = 1 # set status contour active
        self.getContourIndex()
        self.regViewer.widget.viewerLeft.showContourLabel()

    def hideContour(self):
        self.regViewer.status.contour = 0 # set status contour inactive
        self.regViewer.widget.viewerLeft.hideContourLabel()
    
    def treeFindArea(self):
        y = self.regViewer.status.hoverY
        x = self.regViewer.status.hoverX
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
            self.regViewer.widget.viewerLeft.highlightArea(tripled_mm_sorted,activeArea,structureName)


    def getSlice(self):
        if (self.regViewer.status.x_angle == 0) and (self.regViewer.status.y_angle == 0):
            self.slice_in_volume = True
            self.simpleSlice() # update simple slice
        else:
            self.angleSlice() # update angled slice
        self.reference_native = self.slice.copy()
        display_slice = self.reference_native.copy()
        name_dict = {
            'ap': 'AP',
            'si': 'DV',
            'rl': 'ML'
        }
        z_str = name_dict[self.xyz_dict['z'][0]]
        x_str = name_dict[self.xyz_dict['x'][0]]
        y_str = name_dict[self.xyz_dict['y'][0]]
        # get textbox size and calculate textbox coordinates
        offset =  int(self.fontscale * 10) # integer

        text_w, text_h = cv2.getTextSize(z_str + ": " + str(self.regViewer.status.current_z)+"mm", cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, self.fontthickness)[0]
        ap_text_location = [display_slice.shape[1]-offset-text_w,text_h+offset]

        text_w, text_h = cv2.getTextSize(x_str + " Angle: " + str(self.regViewer.status.x_angle), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, self.fontthickness)[0]
        xangle_text_location = [offset,text_h+offset]

        text_w, text_h = cv2.getTextSize(y_str + " Angle: " + str(self.regViewer.status.y_angle), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, self.fontthickness)[0]
        yangle_text_location = [offset,offset+text_h+offset+text_h]
        # put text
        cv2.putText(display_slice, z_str + ": " + str(self.regViewer.status.current_z)+"mm", (ap_text_location[0], ap_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, 255, self.fontthickness, cv2.LINE_AA)
        cv2.putText(display_slice, x_str + " Angle: " + str(self.regViewer.status.x_angle), (xangle_text_location[0], xangle_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, 255, self.fontthickness, cv2.LINE_AA)
        cv2.putText(display_slice, y_str + " Angle: " + str(self.regViewer.status.y_angle), (yangle_text_location[0],yangle_text_location[1]), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, 255, self.fontthickness, cv2.LINE_AA)

        self.slice = cv2.resize(display_slice,(self.regViewer.singleWindowSize[0],self.regViewer.singleWindowSize[1])) # resize to single window size
        if self.regViewer.status.imageRGB is False:
            self.sliceQimg = QImage(self.slice.data, self.slice.shape[1],self.slice.shape[0],self.slice.strides[0],QImage.Format_Grayscale8)
        else:
            self.slice = cv2.cvtColor(self.slice, cv2.COLOR_GRAY2BGR)
            self.sliceQimg = QImage(self.slice.data, self.slice.shape[1],self.slice.shape[0],self.slice.strides[0],QImage.Format_BGR888)

    def getSample(self):
        if self.regViewer.status.sliceNum == 0:
            # self.sampleQimg = QImage(str(self.sharpy_dir.joinpath('sharpy_track','sharpy_track','images','empty.png')))
            self.sample_native = cv2.imread(str(self.sharpy_dir.joinpath('sharpy_track','sharpy_track','images','empty.png')),cv2.IMREAD_COLOR)
            self.sample = cv2.resize(self.sample_native,(self.regViewer.singleWindowSize[0],self.regViewer.singleWindowSize[1]))
            self.sampleQimg = QImage(self.sample.data, self.sample.shape[1],self.sample.shape[0],self.sample.strides[0],QImage.Format_BGR888)
        else:
            self.sample_native = self.imgStack[
                self.regViewer.status.currentSliceNumber
            ]
            self.sample = cv2.resize(self.sample_native,(self.regViewer.singleWindowSize[0],self.regViewer.singleWindowSize[1]))
            if self.regViewer.status.imageRGB is False:
                self.sampleQimg = QImage(self.sample.data, self.sample.shape[1],self.sample.shape[0],self.sample.strides[0],QImage.Format_Grayscale8) # if grayscale sample
            else:
                self.sampleQimg = QImage(self.sample.data, self.sample.shape[1],self.sample.shape[0],self.sample.strides[0],QImage.Format_BGR888) # if RGB sample


    def simpleSlice(self):
        z_coord = coord_mm_transform([self.regViewer.status.current_z], [self.bregma[self.z_idx]],
                                  [self.atlas.space.resolution[self.z_idx]], mm_to_coord=True)
        self.slice = self.template[z_coord, :, :].copy()
        self.z_mat = np.full((self.xyz_dict['y'][1], self.xyz_dict['x'][1]), z_coord)
    
    def angleSlice(self):
        # calculate from ML and DV angle, the plane of current slice
        x_shift = int(np.tan(np.deg2rad(self.regViewer.status.x_angle)) * (self.xyz_dict['x'][1] / 2))
        y_shift = int(np.tan(np.deg2rad(self.regViewer.status.y_angle)) * (self.xyz_dict['y'][1] / 2))
        z_coord = coord_mm_transform([self.regViewer.status.current_z], [self.bregma[self.z_idx]],
                                  [self.atlas.space.resolution[self.z_idx]], mm_to_coord=True)

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
            self.slice_in_volume = True
            # index volume with z_mat and grid
            self.z_mat = z_mat # save AP plane for indexing structure information
            self.z_flat = z_flat # save current AP list to AtlasModel for getContourIndex
            self.slice = self.template[z_flat, self.r_grid_y, self.r_grid_x].reshape(self.xyz_dict['y'][1], self.xyz_dict['x'][1])
        else: # if not empty, show black image with warning
            self.slice_in_volume = False
            self.slice = np.zeros((self.xyz_dict['y'][1], self.xyz_dict['x'][1]),dtype=np.uint8)
            cv2.putText(self.slice, "Slice out of volume!", (int(self.xyz_dict['x'][1]/3),int(self.xyz_dict['y'][1]/2)), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, 255, self.fontthickness, cv2.LINE_AA)
    
    def getStack(self):
        # check image grayscale or RGB, only check first image, assume all grayscale/all RGB
        folder_path = Path(self.regViewer.status.folderPath)
        image_0_path = folder_path / self.regViewer.status.imgFileName[0]
        image_0 = cv2.imread(str(image_0_path), cv2.IMREAD_UNCHANGED)
        if len(image_0.shape) == 2: # gray scale [0-255]
            self.imgStack = np.empty((self.regViewer.status.sliceNum,self.regi_dict['xyz_dict']['y'][1],self.regi_dict['xyz_dict']['x'][1]),dtype=np.uint8) # adaptive imgStack dimension
            # copy slices to stack
            for i in range(self.regViewer.status.sliceNum):
                full_path = folder_path / self.regViewer.status.imgFileName[i]
                img_data = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
                self.imgStack[i,:,:] = img_data

        else: # 3 channel RGB/BGR or 4 channel RGBA
            self.regViewer.status.imageRGB = True
            self.imgStack = np.empty((self.regViewer.status.sliceNum,self.regi_dict['xyz_dict']['y'][1],self.regi_dict['xyz_dict']['x'][1],3),dtype=np.uint8) # for RGBA also just keep RGB channels in stack
            # copy slices to stack
            for i in range(self.regViewer.status.sliceNum):
                full_path = folder_path / self.regViewer.status.imgFileName[i]
                img_data = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
                self.imgStack[i,:,:,:] = img_data[:,:,:3] # load first 3 channels of sample image to stack

        print(self.regViewer.status.sliceNum, "Slice(s) loaded")


    def updateDotPosition(self,mode='default'):
        # ignore if less than 5 pairs of dots
        if len(self.regViewer.widget.viewerLeft.itemGroup) == 0:
            return

        atlas_pts = self._dot_items_to_native(
            self.regViewer.widget.viewerLeft.itemGroup
        )
        sample_pts = self._dot_items_to_native(
            self.regViewer.widget.viewerRight.itemGroup
        )
        if (
            atlas_pts == self.atlas_pts
            and sample_pts == self.sample_pts
            and mode == 'default'
        ):
            return

        self.atlas_pts = atlas_pts
        self.sample_pts = sample_pts
        self.regViewer.status.atlasDots[
            self.regViewer.status.currentSliceNumber
        ] = atlas_pts
        self.regViewer.status.sampleDots[
            self.regViewer.status.currentSliceNumber
        ] = sample_pts
        self.regViewer.status.saveRegistration()

        if len(atlas_pts) >= 5:
            self.updateTransform(
                self._native_points_to_display(atlas_pts),
                self._native_points_to_display(sample_pts),
            )

    def _dot_items_to_native(self, dots):
        points = []
        for dot in dots:
            center = [
                dot.pos().x() + dot.size / 2.0,
                dot.pos().y() + dot.size / 2.0,
            ]
            points.append(self.regViewer.display_to_native_point(center))
        return points

    def _native_points_to_display(self, points):
        return np.asarray(
            [self.regViewer.native_to_display_point(point) for point in points],
            dtype=np.float32,
        )
        
    def checkSaved(self):
        # load exist dots if there is any
        self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] = 1
        # try loading ML,DV,AP location
        try:
            self.regViewer.status.x_angle = self.regViewer.status.atlasLocation[self.regViewer.status.currentSliceNumber][0] # read atlasLocation
            self.regViewer.status.y_angle = self.regViewer.status.atlasLocation[self.regViewer.status.currentSliceNumber][1]
            self.regViewer.status.current_z = self.regViewer.status.atlasLocation[self.regViewer.status.currentSliceNumber][2]
            # update sliders
            self.regViewer.widget.x_slider.setSliderPosition(int(self.regViewer.status.x_angle * 10))
            self.regViewer.widget.y_slider.setSliderPosition(int(self.regViewer.status.y_angle * 10))
            self.regViewer.widget.z_slider.setSliderPosition(coord_mm_transform([self.regViewer.status.current_z], [self.bregma[self.z_idx]],
                                      [self.xyz_dict['z'][2]], mm_to_coord=True))
            self.regViewer.widget.viewerLeft.loadSlice() # slice atlas

        except KeyError: # no record at atlasLocation dictionary
            pass

        if not(self.regViewer.status.currentSliceNumber in self.regViewer.status.atlasLocation):
            pass
        elif not(self.regViewer.status.currentSliceNumber in self.regViewer.status.atlasDots):
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.sliceQimg))
        elif len(self.regViewer.status.atlasDots[self.regViewer.status.currentSliceNumber]) == 0:
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.sliceQimg))
        else:
            atlas_pts = self.regViewer.status.atlasDots[self.regViewer.status.currentSliceNumber] # read dictionary, create dot object
            sample_pts = self.regViewer.status.sampleDots[self.regViewer.status.currentSliceNumber]
            self._validate_dot_pairs(atlas_pts, sample_pts)
            self.atlas_pts = atlas_pts
            self.sample_pts = sample_pts
            self._display_dot_pairs(atlas_pts, sample_pts)
            if len(atlas_pts) >= 5:
                self.updateTransform(
                    self._native_points_to_display(atlas_pts),
                    self._native_points_to_display(sample_pts),
                )

    def _validate_dot_pairs(self, atlas_pts, sample_pts):
        for xyAtlas, xySample in zip(atlas_pts,sample_pts):
            # check if dot coordinates are within boundary
            if (xyAtlas[0] >=0) and (xyAtlas[0] < self.regViewer.atlas_resolution[0]) and (
                xyAtlas[1] >=0) and (xyAtlas[1] < self.regViewer.atlas_resolution[1]) and (
                xySample[0] >=0) and (xySample[0] < self.regViewer.atlas_resolution[0]) and (
                xySample[1] >=0) and (xySample[1] < self.regViewer.atlas_resolution[1]):
                pass
            else:
                raise IndexError("Registration coordinates out of boundary! \n"
                                 "Check slide {} : atlasDots {}, sampleDots{}. \n"
                                 "Must fulfill: [0=<i<{},0<=j<{}]".format(self.regViewer.status.currentSliceNumber,
                                                                          xyAtlas,xySample,
                                                                          self.regViewer.atlas_resolution[0],self.regViewer.atlas_resolution[1]))

    def _clear_visible_dot_pairs(self):
        left_scene = self.regViewer.widget.viewerLeft.scene
        right_scene = self.regViewer.widget.viewerRight.scene
        previous_left_block = left_scene.blockSignals(True)
        previous_right_block = right_scene.blockSignals(True)
        try:
            for dot in self.regViewer.widget.viewerLeft.itemGroup:
                left_scene.removeItem(dot)
            for dot in self.regViewer.widget.viewerRight.itemGroup:
                right_scene.removeItem(dot)
            self.regViewer.widget.viewerLeft.itemGroup = []
            self.regViewer.widget.viewerRight.itemGroup = []
        finally:
            left_scene.blockSignals(previous_left_block)
            right_scene.blockSignals(previous_right_block)

    def _display_dot_pairs(self, atlas_pts, sample_pts):
        self._clear_visible_dot_pairs()
        left_scene = self.regViewer.widget.viewerLeft.scene
        right_scene = self.regViewer.widget.viewerRight.scene
        previous_left_block = left_scene.blockSignals(True)
        previous_right_block = right_scene.blockSignals(True)
        try:
            for xyAtlas, xySample in zip(atlas_pts,sample_pts):
                display_atlas = self.regViewer.native_to_display_point(xyAtlas)
                display_sample = self.regViewer.native_to_display_point(xySample)
                dotLeft = DotObject(
                    display_atlas[0],
                    display_atlas[1],
                    self.regViewer.dotRR,
                )
                dotRight = DotObject(
                    display_sample[0],
                    display_sample[1],
                    self.regViewer.dotRR,
                )
                
                dotLeft.linkPairedDot(dotRight)
                dotRight.linkPairedDot(dotLeft)
                # add dots to scene
                left_scene.addItem(dotLeft)
                right_scene.addItem(dotRight)
                # store dot to itemGroup
                self.regViewer.widget.viewerLeft.itemGroup.append(dotLeft) # add dot to leftViewer
                self.regViewer.widget.viewerRight.itemGroup.append(dotRight) # add dot to rightViewer
        finally:
            left_scene.blockSignals(previous_left_block)
            right_scene.blockSignals(previous_right_block)

    def updateTransform(self,atlas_pts,sample_pts):
        self.clearPredictedPreview()
        transform = fitGeoTrans(sample_pts,atlas_pts) # save transform for prediction
        self.rtransform = fitGeoTrans(atlas_pts,sample_pts)
        self.sampleWarp,self.sampleBlend,self.qWarp,self.qBlend = self._render_transform(transform)

        if not(self.regViewer.status.currentSliceNumber in self.regViewer.status.blendMode):
            self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] = 1 # overlay
        else:
            pass
        if self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] == 0: # all atlas
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.sliceQimg))
        elif self.regViewer.status.blendMode[self.regViewer.status.currentSliceNumber] == 1: # overlay
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.qBlend))
        else:
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(QPixmap.fromImage(self.qWarp)) # all sample

    def applyPredictedTransform(self):
        if not getattr(self, "slice_in_volume", True):
            raise ValueError(
                "Prediction is unavailable because the atlas slice extends "
                "outside the atlas volume. Adjust the slice position or angles."
            )
        if self.predictor is None:
            self.predictor = RegistrationTransformPredictor(
                self.regViewer.regi_dict["model_path"]
            )
        native_transform = self.predictor.predict_transform(
            self.sample_native,
            self.reference_native,
        )
        display_transform = homography_to_display_space(
            native_transform,
            self._image_size(self.sample_native),
            self._image_size(self.sample),
            self._image_size(self.reference_native),
            self._image_size(self.slice),
        )
        self.clearPredictedPreview()
        self.predictedTransformNative = native_transform
        self.predictedTransformDisplay = display_transform
        self.predictedSliceNumber = self.regViewer.status.currentSliceNumber
        (
            self.predictedWarp,
            self.predictedBlend,
            self.qPredictedWarp,
            self.qPredictedBlend,
        ) = self._render_transform(display_transform)
        self.regViewer.status.blendMode[self.predictedSliceNumber] = 1
        self.showPredictedPreview()

    def materializePredictedDots(self):
        if not self.hasPredictedPreview():
            return False

        current_slice = self.regViewer.status.currentSliceNumber
        sample_pts, atlas_pts = homography_to_registration_points(
            self.predictedTransformNative,
            self.sample_native.shape,
            self.regViewer.atlas_resolution,
        )
        self._validate_dot_pairs(atlas_pts, sample_pts)
        self.regViewer.status.atlasLocation[current_slice] = [
            self.regViewer.status.x_angle,
            self.regViewer.status.y_angle,
            self.regViewer.status.current_z,
        ]
        self.regViewer.status.atlasDots[current_slice] = atlas_pts
        self.regViewer.status.sampleDots[current_slice] = sample_pts
        self.atlas_pts = atlas_pts
        self.sample_pts = sample_pts
        self._display_dot_pairs(atlas_pts, sample_pts)
        self.regViewer.status.blendMode[current_slice] = 1
        self.regViewer.status.saveRegistration()
        self.updateTransform(
            self._native_points_to_display(atlas_pts),
            self._native_points_to_display(sample_pts),
        )
        self.regViewer.widget.updatePredictButton()
        return True

    def clearPredictedPreview(self, reset_view=False):
        active_preview = (
            self.predictedSliceNumber == self.regViewer.status.currentSliceNumber
        )
        self.predictedSliceNumber = None
        self.predictedTransformNative = None
        self.predictedTransformDisplay = None
        for attr in [
            "predictedWarp",
            "predictedBlend",
            "qPredictedWarp",
            "qPredictedBlend",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)
        if reset_view and active_preview:
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(
                QPixmap.fromImage(self.sliceQimg)
            )

    def hasPredictedPreview(self):
        return (
            self.predictedSliceNumber == self.regViewer.status.currentSliceNumber
            and hasattr(self, "qPredictedBlend")
            and hasattr(self, "qPredictedWarp")
        )

    def showPredictedPreview(self):
        if not self.hasPredictedPreview():
            return False

        blend_mode = self.regViewer.status.blendMode.get(
            self.regViewer.status.currentSliceNumber,
            1,
        )
        if blend_mode == 0:
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(
                QPixmap.fromImage(self.sliceQimg)
            )
        elif blend_mode == 1:
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(
                QPixmap.fromImage(self.qPredictedBlend)
            )
        else:
            self.regViewer.widget.viewerLeft.labelImg.setPixmap(
                QPixmap.fromImage(self.qPredictedWarp)
            )
        return True

    def showCurrentTransformPreview(self):
        if self.showPredictedPreview():
            return
        self.updateDotPosition(mode='force')

    @staticmethod
    def _image_size(image):
        return [image.shape[1], image.shape[0]]

    def _render_transform(self, transform):
        sample_warp = cv2.warpPerspective(
            self.sample,
            transform,
            (self.regViewer.singleWindowSize[0], self.regViewer.singleWindowSize[1]),
        )
        sample_blend = cv2.addWeighted(self.slice, 0.5, sample_warp, 0.5, 0)

        if self.regViewer.status.imageRGB is False:
            q_warp = QImage(
                sample_warp.data,
                sample_warp.shape[1],
                sample_warp.shape[0],
                sample_warp.strides[0],
                QImage.Format_Grayscale8,
            )
            q_blend = QImage(
                sample_blend.data,
                sample_blend.shape[1],
                sample_blend.shape[0],
                sample_blend.strides[0],
                QImage.Format_Grayscale8,
            )
        else:
            q_warp = QImage(
                sample_warp.data,
                sample_warp.shape[1],
                sample_warp.shape[0],
                sample_warp.strides[0],
                QImage.Format_BGR888,
            )
            q_blend = QImage(
                sample_blend.data,
                sample_blend.shape[1],
                sample_blend.shape[0],
                sample_blend.strides[0],
                QImage.Format_BGR888,
            )
        return sample_warp, sample_blend, q_warp, q_blend
