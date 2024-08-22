import json
import numpy as np
import pandas as pd

import cv2
from sklearn.preprocessing import minmax_scale
from matplotlib import path
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from napari_dmc_brainmap.utils import get_animal_id, get_bregma, xyz_atlas_transform, coord_mm_transform
from bg_atlasapi import BrainGlobeAtlas


class sliceHandle():
    def __init__(self, regi_dict=False) -> None:
        if regi_dict:
            self.regi_dict = regi_dict
            self.jsonPath = self.regi_dict['regi_dir'].joinpath('registration.json')
            self.parseJSON()
            self.getTransform()
            self.calculateImageGrid()
            print("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(self.regi_dict['atlas'])
            self.z_idx = self.atlas.space.axes_description.index(self.regi_dict['xyz_dict']['z'][0])
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
        self.bregma = get_bregma(self.regi_dict['atlas'])

    def setSlice(self, slice_n):
        if type(slice_n) is int: # if slice number identifier is integer
            self.currentSlice = slice_n
        elif type(slice_n) is str: # if slice number identifier is string
            try:
                self.currentSlice = int(slice_n) # convert string number to integer number
            except ValueError:
                for k in self.regData['imgName'].keys():
                    if self.regData['imgName'][k] == slice_n:
                        self.currentSlice = int(k)
                        print("Slice Index Found")
                        break
                    else:
                        pass
        else:
            print('Unknown Identifier for Slice Number!')
            print('Slice Number not updated!')
        self.loadImg()

    def setImgFolder(self, ImgFolder):
        self.ImgFolder = ImgFolder

    def loadImg(self):
        # self.sampleImgFiles = natsorted([f.parts[-1] for f in self.ImgFolder.glob('*.tif')])
        # self.currentSampleImg = tifffile.imread((self.ImgFolder.joinpath(self.sampleImgFiles[self.currentSlice])))
        print('Working on: ', self.regData['imgName'][str(self.currentSlice)])

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
        z_plane = self.get_z_plane(slice_n)
        volIndex_list = []
        for s_coord in sample_coords:
            x_pre, y_pre = s_coord
            x_post, y_post = mapPointTransform(x_pre, y_pre, self.tforms[slice_n])
            y = int(y_post)
            x = int(x_post)
            try:
                z = int(z_plane[y, x])
                # if any of x,y,z coordinate is negative, abort append
                if (x<0)|(y<0)|(z<0):
                    print(s_coord," mapping out of bound, skipping!")
                else:
                    volIndex_list.append([x, y, z])
            except IndexError:
                print(s_coord," mapping out of bound, skipping!")
        return volIndex_list

    def get_z_plane(self, slice_n):
        x_angle = self.regData['atlasLocation'][slice_n][0]
        y_angle = self.regData['atlasLocation'][slice_n][1]
        x_max = self.regi_dict['xyz_dict']['x'][1]
        y_max = self.regi_dict['xyz_dict']['y'][1]
        z_coord = coord_mm_transform([self.regData['atlasLocation'][slice_n][2]], [self.bregma[self.z_idx]],
                                  [self.atlas.space.resolution[self.z_idx]], mm_to_coord=True)

        if (x_angle == 0) and (y_angle == 0):  # flat plane
            z_plane = np.full((y_max, x_max), z_coord, dtype=np.uint16)

        else:  # angled plane
            x_shift = int(np.tan(np.deg2rad(x_angle)) * x_max/2)
            y_shift = int(np.tan(np.deg2rad(y_angle)) * y_max/2)
            center = np.array([z_coord, y_max/2, x_max/2])
            c_right = np.array([z_coord + x_shift, y_max/2, x_max-1])
            c_top = np.array([z_coord - y_shift, 0, x_max/2])
            vec_1 = c_right - center
            vec_2 = c_top - center
            vec_n = np.cross(vec_1, vec_2)
            z_plane = (-vec_n[1] * (self.grid[:, :, 0] - center[1]) - vec_n[2] * (self.grid[:, :, 1] - center[2])) / \
                       vec_n[0] + center[0]

        return z_plane

    def calculateImageGrid(self):  # one time calculation
        x_max = self.regi_dict['xyz_dict']['x'][1]
        y_max = self.regi_dict['xyz_dict']['y'][1]
        y = np.arange(y_max)
        x = np.arange(x_max)
        grid_x, grid_y = np.meshgrid(x, y)
        self.grid = np.stack([grid_y, grid_x], axis=2)

    def getBrainArea(self, inputCoordinates, section_name):

        inputCoordinates = np.array(inputCoordinates)  # inputCoordinates in [x, y]

        volIndex_list = self.getVolumeIndex(str(self.currentSlice), inputCoordinates)  # [[x, y, z], [x, y, z]...] in 'dmc-brainmap space'
        # if return list is empty: return None
        if len(volIndex_list) == 0:
            return None
        else:
            # transfer xyz coordinates to convention used by atlas (bg_atlasapi)
            volIndex_list = [xyz_atlas_transform(v, self.regi_dict, self.atlas.space.axes_description) for v in volIndex_list]
            id_list = []
            name_list = []
            acronym_list = []
            vol_mm_list = []
            for triplet in volIndex_list:
                structure_id = self.atlas.structure_from_coords(triplet)
                id_list.append(structure_id)
                if structure_id > 0 :
                    name_list.append(self.sTree.data[structure_id]['name'])
                    acronym_list.append(self.sTree.data[structure_id]['acronym'])
                else:
                    name_list.append('root')
                    acronym_list.append('root')
                # calculate Allen coordinates in mm unit
                vol_mm = coord_mm_transform(triplet, self.bregma, self.atlas.space.resolution)
                vol_mm_list.append(vol_mm)
            name_dict = {
                'ap': 'ap',
                'si': 'dv',
                'rl': 'ml'
            }

            a_coord, b_coord, c_coord = map(list, zip(*volIndex_list))
            a_mm, b_mm, c_mm = map(list, zip(*vol_mm_list))
            col_names = ['name', 'acronym', 'structure_id']
            col_names.extend([name_dict[n] + '_mm' for n in self.atlas.space.axes_description])
            col_names.extend([name_dict[n] + '_coords' for n in self.atlas.space.axes_description])
            section_data = pd.DataFrame(list(zip(name_list, acronym_list, id_list, a_mm, b_mm, c_mm,
                                                a_coord, b_coord, c_coord)),
                                        columns=col_names)
            section_data['section_name'] = [section_name] * len(section_data)
            return section_data


def regi_points_polygon(x_scaled, y_scaled):

    poly_points = [(x_scaled[i], y_scaled[i]) for i in range(0, len(x_scaled))]
    polygon = path.Path(poly_points)
    x_min, x_max = x_scaled.min(), x_scaled.max()
    y_min, y_max = y_scaled.min(), y_scaled.max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, (x_max - x_min) + 1),
                         np.linspace(y_min, y_max, (y_max - y_min) + 1))
    canvas_points = [(np.ndarray.flatten(xx)[i], np.ndarray.flatten(yy)[i]) for i in
                     range(0, len(np.ndarray.flatten(xx)))]
    idx_in_polygon = polygon.contains_points(canvas_points)
    points_in_polygon = [c for c, i in zip(canvas_points, idx_in_polygon) if i]
    x_poly = [p[0] for p in points_in_polygon]
    y_poly = [p[1] for p in points_in_polygon]
    coords = np.stack([x_poly, y_poly], axis=1)
    return coords



def transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir, seg_im_suffix, regi_data, regi_dir, regi_suffix):
    # todo input differently?
    curr_im = im[:-len(segment_suffix)]
    img = cv2.imread(str(seg_im_dir.joinpath(curr_im + seg_im_suffix)))
    y_im, x_im, z_im = img.shape  # original resolution of image
    # correct for 0 indices
    y_im -= 1
    x_im -= 1
    img_regi = cv2.imread(str(regi_dir.joinpath(curr_im + regi_suffix)))
    y_low, x_low, z_low = img_regi.shape  # original resolution of image
    # correct for 0 indices
    y_low -= 1
    x_low -= 1

    segment_data = pd.read_csv(segment_dir.joinpath(im))
    y_pos = list(segment_data['Position Y'])
    x_pos = list(segment_data['Position X'])
    # append mix max values for rescaling
    y_pos.append(0)
    y_pos.append(y_im)
    x_pos.append(0)
    x_pos.append(x_im)
    y_scaled = np.ceil(minmax_scale(y_pos, feature_range=(0, y_low)))[:-2].astype(int)
    x_scaled = np.ceil(minmax_scale(x_pos, feature_range=(0, x_low)))[:-2].astype(int)
    if seg_type == 'injection_site':
        for n in segment_data['idx_shape'].unique():
            n_idx = segment_data.index[segment_data['idx_shape'] == n].tolist()
            curr_x = np.array([x_scaled[i] for i in n_idx])
            curr_y = np.array([y_scaled[i] for i in n_idx])
            curr_coords = regi_points_polygon(curr_x, curr_y)
            if n == 0:
                coords = curr_coords
            else:
                coords = np.concatenate((coords, curr_coords), axis=0)

    else:
        coords = np.stack([x_scaled, y_scaled], axis=1)

    # slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
    s.setImgFolder(regi_dir)
    # set which slice in there
    s.setSlice(curr_im + regi_suffix)
    section_data = s.getBrainArea(coords, (curr_im + regi_suffix))
    if seg_type == "genes":
        section_data['cluster_id'] = segment_data['cluster_id']
        section_data['spot_id'] = segment_data['spot_id']
    return section_data