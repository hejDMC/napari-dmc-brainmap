import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from pathlib import Path

from napari.utils.notifications import show_info
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, xyz_atlas_transform, coord_mm_transform
from bg_atlasapi import BrainGlobeAtlas

class SliceHandle():
    """
    Class for handling slice data, including registration, atlas loading, transformations,
    and accessing brain region information.
    """
    def __init__(self, regi_dict: Union[Dict, bool] = False) -> None:
        """
        Initialize the SliceHandle instance.

        Parameters:
            regi_dict (Union[Dict, bool]): Registration dictionary. Defaults to False if not available.
        """
        if regi_dict:
            self.regi_dict = regi_dict
            self.jsonPath = self.regi_dict['regi_dir'].joinpath('registration.json')
            self.parseJSON()
            self.getTransform()
            self.calculateImageGrid()
            show_info("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(self.regi_dict['atlas'])
            self.z_idx = self.atlas.space.axes_description.index(self.regi_dict['xyz_dict']['z'][0])
            self.loadAnnot()
            self.loadStructureTree()
            self.currentSlice = None
            self.ImgFolder = None
        else:
            show_info("no registration data found!")

    def loadAnnot(self) -> None:
        """
        Load annotation data from the atlas.
        """
        self.annot = self.atlas.annotation

    def loadStructureTree(self) -> None:
        """
        Load the structure tree from the atlas and initialize bregma coordinates.
        """
        self.sTree = self.atlas.structures
        self.bregma = get_bregma(self.regi_dict['atlas'])

    def setSlice(self, slice_n: Union[int, str]) -> None:
        """
        Set the current slice by index or name.

        Parameters:
            slice_n (Union[int, str]): Slice identifier (integer or string).
        """
        if type(slice_n) is int: # if slice number identifier is integer
            self.currentSlice = slice_n
        elif type(slice_n) is str: # if slice number identifier is string
            try:
                self.currentSlice = int(slice_n) # convert string number to integer number
            except ValueError:
                for k in self.regData['imgName'].keys():
                    if self.regData['imgName'][k] == slice_n:
                        self.currentSlice = int(k)
                        show_info("Slice Index Found")
                        break
                    else:
                        pass
        else:
            show_info('Unknown Identifier for Slice Number!')
            show_info('Slice Number not updated!')
        self.loadImg()

    def setImgFolder(self, ImgFolder: Path) -> None:
        """
        Set the folder containing sample images.

        Parameters:
            ImgFolder (Path): Path to the image folder.
        """
        self.ImgFolder = ImgFolder

    def loadImg(self) -> None:
        """
        Load the current sample image.
        """
        # self.sampleImgFiles = natsorted([f.parts[-1] for f in self.ImgFolder.glob('*.tif')])
        # self.currentSampleImg = tifffile.imread((self.ImgFolder.joinpath(self.sampleImgFiles[self.currentSlice])))
        show_info(f"Working on: {self.regData['imgName'][str(self.currentSlice)]}")

    def parseJSON(self) -> None:
        """
        Parse the registration JSON file to load registration data.
        """
        with open(self.jsonPath, "r") as f:
            self.regData = json.load(f)


    def getTransform(self) -> None:
        """
        Calculate geometric transformations for registration.
        """
        self.tforms = {}
        for k in self.regData['atlasDots'].keys():
            if len(self.regData['atlasDots'][k]) < 5:
                pass
            else:  # valid pairs of points, calculate transformation
                self.tforms[k] = fitGeoTrans(self.regData['sampleDots'][k], self.regData['atlasDots'][k])

    def getVolumeIndex(self, slice_n: str, sample_coords: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Map sample coordinates to volume indices in the atlas.

        Parameters:
            slice_n (str): Slice identifier.
            sample_coords (List[Tuple[int, int]]): List of sample coordinates.

        Returns:
            List[List[int]]: List of volume indices in [x, y, z] format.
        """
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
                if not (x<0)|(y<0)|(z<0):
                    volIndex_list.append([x, y, z])
            except IndexError:
                pass
        return volIndex_list

    def get_z_plane(self, slice_n: str) -> np.ndarray:
        """
        Calculate the z-plane for the given slice.

        Parameters:
            slice_n (str): Slice identifier.

        Returns:
            np.ndarray: Z-plane as a NumPy array.
        """
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

    def calculateImageGrid(self) -> None:
        """
        Precompute the 2D image grid for transformations.
        """
        x_max = self.regi_dict['xyz_dict']['x'][1]
        y_max = self.regi_dict['xyz_dict']['y'][1]
        y = np.arange(y_max)
        x = np.arange(x_max)
        grid_x, grid_y = np.meshgrid(x, y)
        self.grid = np.stack([grid_y, grid_x], axis=2)

    def getBrainArea(self, inputCoordinates: List[Tuple[int, int]], section_name: str) -> Union[None, pd.DataFrame]:
        """
        Retrieve brain area information for input coordinates.

        Parameters:
            inputCoordinates (List[Tuple[int, int]]): List of coordinates in [x, y] format.
            section_name (str): Name of the section being analyzed.

        Returns:
            Union[None, pd.DataFrame]: DataFrame containing brain area details, or None if no data is found.
        """

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