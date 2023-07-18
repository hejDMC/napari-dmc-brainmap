
import pandas as pd
from pathlib import Path
from pkg_resources import resource_filename
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import *
from napari_dmc_brainmap.utils import get_bregma
import numpy as np
import json
from bg_atlasapi import BrainGlobeAtlas

class sliceHandle():
    def __init__(self, registration=False) -> None:
        if registration:
            self.jsonPath = registration
            self.parseJSON()
            self.getTransform()
            self.calculateImageGrid()
            print("loading reference atlas...")
            self.atlas = BrainGlobeAtlas("allen_mouse_10um")
            self.loadAnnot()
            self.loadStructureTree()
            self.currentSlice = None
            self.ImgFolder = None
        else:
            print("no registration data found!")

    def loadAnnot(self):
        self.annot = self.atlas.annotation

    def loadStructureTree(self):
        self.sTree = self.atlas.structures
        self.bregma = get_bregma()

    def setSlice(self, slice_n):
        self.currentSlice = slice_n
        self.loadImg()

    def setImgFolder(self, ImgFolder):
        self.ImgFolder = ImgFolder

    def loadImg(self):
        files_all = natsorted(os.listdir(self.ImgFolder))
        self.sampleImgFiles = []
        for f in files_all:
            if not (f.startswith('.')) and (f.endswith('.tif') or f.endswith('.tiff')):
                self.sampleImgFiles.append(f)
            else:
                pass
        self.currentSampleImg = tifffile.imread(os.path.join(self.ImgFolder, self.sampleImgFiles[self.currentSlice]))
        print('Working on: ', self.sampleImgFiles[self.currentSlice])

    def parseJSON(self):
        with open(self.jsonPath, "r") as f:
            self.regData = json.load(f)

    def getTransform(self):
        self.tforms = {}
        for k in self.regData['atlasDots'].keys():
            if len(self.regData['atlasDots'][k]) < 5:
                pass
            else:  # valid pairs of points, calculate transformation
                self.tforms[k] = fitGeoTrans(self.regData['sampleDots'][k], self.regData['atlasDots'][k])

    def getVolumeIndex(self, slice_n, sample_coords):
        ap_plane = self.getPlaneAP(slice_n)
        volIndex_list = []
        for s_coord in sample_coords:
            x_pre, y_pre = s_coord
            x_post, y_post = mapPointTransform(x_pre, y_pre, self.tforms[slice_n])
            dv = int(y_post)
            ml = int(x_post)
            ap = int(ap_plane[dv, ml])
            volIndex_list.append([ap, dv, ml])
        return volIndex_list

    def getStructureID(self, volIndex):
        return self.annot[volIndex[0], volIndex[1], volIndex[2]]

    def getPlaneAP(self, slice_n):
        MLangle = self.regData['atlasLocation'][slice_n][0]
        DVangle = self.regData['atlasLocation'][slice_n][1]

        ap_c = get_ap(self.regData['atlasLocation'][slice_n][2])

        if (MLangle == 0) and (DVangle == 0):  # flat plane
            ap_plane = np.full((800, 1140), ap_c, dtype=np.uint16)

        else:  # angled plane
            ml_shift = int(np.tan(np.deg2rad(MLangle)) * 570)
            dv_shift = int(np.tan(np.deg2rad(DVangle)) * 400)
            center = np.array([ap_c, 400, 570])
            c_right = np.array([ap_c + ml_shift, 400, 1139])
            c_top = np.array([ap_c - dv_shift, 0, 570])
            vec_1 = c_right - center
            vec_2 = c_top - center
            vec_n = np.cross(vec_1, vec_2)
            ap_plane = (-vec_n[1] * (self.grid[:, :, 0] - center[1]) - vec_n[2] * (self.grid[:, :, 1] - center[2])) / \
                       vec_n[0] + center[0]

        return ap_plane

    def getCoordMM(self, vol_index):
        vol_ap, vol_dv, vol_ml = vol_index
        ap_mm = np.round((self.bregma[0] - vol_ap) * 0.01, 2)
        dv_mm = np.round((self.bregma[1] - vol_dv) * 0.01, 2)
        ml_mm = np.round((self.bregma[2] - vol_ml) * 0.01, 2)
        return [ap_mm, dv_mm, ml_mm]

    def calculateImageGrid(self):  # one time calculation
        dv = np.arange(800)
        ml = np.arange(1140)
        grid_x, grid_y = np.meshgrid(ml, dv)
        self.grid = np.stack([grid_y, grid_x], axis=2)

    def getBrainArea(self, inputCoordinates, section_name):

        inputCoordinates = np.array(inputCoordinates)

        vol_index = self.getVolumeIndex(str(self.currentSlice), inputCoordinates)  # [ap,dv,ml]
        id_list = []
        name_list = []
        acronym_list = []
        vol_mm_list = []
        for i in vol_index:
            structure_id = self.getStructureID(i)
            id_list.append(structure_id)
            if structure_id > 0 :
                name_list.append(self.sTree.data[structure_id]['name'])
                acronym_list.append(self.sTree.data[structure_id]['acronym'])
            else:
                name_list.append('root')
                acronym_list.append('root')
            # calculate Allen coordinates in mm unit
            vol_mm = self.getCoordMM(i)
            vol_mm_list.append(vol_mm)
        ap_coord, dv_coord, ml_coord = map(list, zip(*vol_index))
        ap_mm, dv_mm, ml_mm = map(list, zip(*vol_mm_list))

        section_data = pd.DataFrame(list(zip(name_list, acronym_list, ap_mm, dv_mm, ml_mm,
                                             id_list, ap_coord, dv_coord, ml_coord)),
                                    columns=['name', 'acronym', 'ap_mm', 'dv_mm', 'ml_mm',
                                             'structure_id', 'zpixel', 'ypixel', 'xpixel'])
        section_data['section_name'] = [section_name] * len(section_data)
        return section_data






