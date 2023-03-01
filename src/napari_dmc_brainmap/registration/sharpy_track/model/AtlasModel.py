import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
from sharpy_track.view.DotObject import DotObject
from sharpy_track.model.calculation import *
import os

class AtlasModel():
    def __init__(self) -> None:
        self.loadVolume()
        self.loadAnnot()
        self.calculateImageGrid()
        self.imgStack = None
    

    def loadVolume(self):
        self.vol = np.load(os.path.join('sharpy_track','sharpy_track','atlas','template_volume_8bit.npy')) # load 8bit volume
    
    def loadAnnot(self):
        self.annot = np.load(os.path.join('sharpy_track','sharpy_track','atlas','annotation_volume_10um_by_index.npy'))

    def calculateImageGrid(self):
        dv = np.arange(800)
        ml = np.arange(1140)
        grid_x,grid_y = np.meshgrid(ml,dv)
        self.r_grid_x = grid_x.ravel()
        self.r_grid_y = grid_y.ravel()
        self.grid = np.stack([grid_y,grid_x],axis=2)
    
    def getContourIndex(self,regViewer):
        # check simple slice or angled slice
        # slice annotation volume, convert to int32 for contour detection
        if (regViewer.status.MLangle == 0) and (regViewer.status.DVangle == 0):
            self.sliceAnnot = self.annot[get_ap(regViewer.status.currentAP),:,:].copy().astype(np.int32)
        else:
            self.sliceAnnot = self.annot[self.ap_flat,self.r_grid_y,self.r_grid_x].reshape(800,1140).astype(np.int32)
        # get contours
        contours,_ = cv2.findContours(self.sliceAnnot, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # create canvas
        empty = np.zeros((800,1140),dtype=np.uint8)
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

        

    def getSlice(self,regViewer):
        if (regViewer.status.MLangle == 0) and (regViewer.status.DVangle == 0):
            self.simpleSlice(regViewer) # update simple slice
        else:
            self.angleSlice(regViewer) # update angled slice

        cv2.putText(self.slice, "AP: "+str(regViewer.status.currentAP), (950,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
        cv2.putText(self.slice, "ML Angle: "+str(regViewer.status.MLangle), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
        cv2.putText(self.slice, "DV Angle: "+str(regViewer.status.DVangle), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)

        self.slice = cv2.resize(self.slice,(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1])) # resize to single window size
        self.sliceQimg = QImage(self.slice.data, self.slice.shape[1],self.slice.shape[0],self.slice.strides[0],QImage.Format_Grayscale8)

    def getSample(self,regViewer):
        if regViewer.status.sliceNum == 0:
            self.sampleQimg = QImage(os.path.join('sharpy_track','sharpy_track','images','empty.png'))
        else:
            self.sample = cv2.resize(self.imgStack[regViewer.status.currentSliceNumber],(regViewer.status.singleWindowSize[0],regViewer.status.singleWindowSize[1]))
            self.sampleQimg = QImage(self.sample.data, self.sample.shape[1],self.sample.shape[0],self.sample.strides[0],QImage.Format_Grayscale8)
            # regViewer.atlasModel.checkSaved(regViewer)

    def simpleSlice(self,regViewer):
        self.slice = self.vol[get_ap(regViewer.status.currentAP),:,:].copy()
    
    def angleSlice(self,regViewer):
        # calculate from ML and DV angle, the plane of current slice
        ml_shift = int(np.tan(np.deg2rad(regViewer.status.MLangle)) * 570)
        dv_shift = int(np.tan(np.deg2rad(regViewer.status.DVangle)) * 400)

        center = np.array([get_ap(regViewer.status.currentAP),400,570])
        c_right = np.array([get_ap(regViewer.status.currentAP)+ml_shift,400,1139])
        c_top = np.array([get_ap(regViewer.status.currentAP)-dv_shift,0,570]) 
        # calculate plane vector
        vec_1 = c_right-center
        vec_2 = c_top-center
        vec_n = np.cross(vec_1,vec_2)
        # calculate AP matrix
        ap_mat = (-vec_n[1]*(self.grid[:,:,0]-center[1])-vec_n[2]*(self.grid[:,:,1]-center[2]))/vec_n[0] + center[0]
        ap_flat = ap_mat.astype(int).ravel() # flatten AP matrix
        # within-volume check
        outside_vol = np.argwhere((ap_flat<0)|(ap_flat>1319)) # outside of volume index
        if outside_vol.size == 0: # if outside empty, inside of volume
            # index volume with ap_mat and grid
            self.ap_flat = ap_flat # add current AP list to AtlasModel for getContourIndex
            self.slice = self.vol[ap_flat,self.r_grid_y,self.r_grid_x].reshape(800,1140)
        else: # if not empty, show black image with warning
            self.slice = np.zeros((800,1140),dtype=np.uint8)
            cv2.putText(self.slice, "Slice out of volume!", (400,400), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
    
    def getStack(self,regViewer):
        self.imgStack = np.full((regViewer.status.sliceNum,800,1140),-1,dtype=np.uint8)
        # copy slices to stack
        for i in range(regViewer.status.sliceNum):
            full_path = os.path.join(regViewer.status.folderPath,regViewer.status.imgFileName[i])
            img_data = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
            self.imgStack[i,:,:] = img_data
        print(regViewer.status.sliceNum,"Slice(s) loaded")
    

    def updateDotPosition(self,regViewer):
        # ignore if less than 5 pairs of dots
        if len(regViewer.widget.viewerLeft.itemGroup) < 5:
            # check if has saved coodinates
            pass
        else: # refresh dot coodinate
            atlas_pts = [] 
            for dot in regViewer.widget.viewerLeft.itemGroup: # itemGroup to list
                atlas_pts.append([dot.pos().x(),dot.pos().y()])    
            sample_pts = []
            for dot in regViewer.widget.viewerRight.itemGroup: # itemGroup to list
                sample_pts.append([dot.pos().x(),dot.pos().y()])
            # update dot record in dictionary
            regViewer.status.atlasDots[regViewer.status.currentSliceNumber] = atlas_pts
            regViewer.status.sampleDots[regViewer.status.currentSliceNumber] = sample_pts
            regViewer.status.saveRegistration()
            # apply transformation
            self.updateTransform(regViewer,atlas_pts,sample_pts)
        
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

            regViewer.status.MLangle = regViewer.status.atlasLocation[regViewer.status.currentSliceNumber][0] # read atlasLocation
            regViewer.status.DVangle = regViewer.status.atlasLocation[regViewer.status.currentSliceNumber][1]
            regViewer.status.currentAP = regViewer.status.atlasLocation[regViewer.status.currentSliceNumber][2]
            regViewer.widget.viewerLeft.loadSlice(regViewer) # slice atlas

            for xyAtlas, xySample in zip(atlas_pts,sample_pts):
                dotLeft = DotObject(xyAtlas[0], xyAtlas[1], 10) # list to itemGroup
                dotRight = DotObject(xySample[0], xySample[1], 10) # list to itemGroup
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
        self.sampleWarp = cv2.warpPerspective(self.sample,transform,(1140,800))
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

    
