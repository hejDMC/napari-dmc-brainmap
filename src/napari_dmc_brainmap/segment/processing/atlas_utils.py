import numpy as np
from typing import Dict, List, Tuple, Union
from bg_atlasapi import config, BrainGlobeAtlas
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.atlas_utils import coord_mm_transform


def calculateImageGrid(x_res: int, y_res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a 2D image grid based on x and y resolutions.

    Parameters:
        x_res (int): The resolution along the x-axis.
        y_res (int): The resolution along the y-axis.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the grid,
        raveled x-coordinates, and raveled y-coordinates.
    """
    y = np.arange(y_res)
    x = np.arange(x_res)
    grid_x, grid_y = np.meshgrid(x, y)
    r_grid_x = grid_x.ravel()
    r_grid_y = grid_y.ravel()
    grid = np.stack([grid_y, grid_x], axis=2)
    return grid, r_grid_x, r_grid_y

def loadAnnotBool(atlas: str) -> np.ndarray:
    """
    Load or create a binary annotation volume for a given atlas.

    Parameters:
        atlas (str): The name of the atlas.

    Returns:
        np.ndarray: A binary annotation volume where 0 indicates outside the brain
        and 255 indicates inside the brain.
    """
    brainglobe_dir = config.get_brainglobe_dir()
    atlas_name_general = f"{atlas}_v*"
    atlas_names_local = list(brainglobe_dir.glob(atlas_name_general))[
        0]  # glob returns generator object, need to exhaust it in list, then take out
    annot_bool_dir = brainglobe_dir.joinpath(atlas_names_local, 'annot_bool.npy')
    # for any atlas else, in this case test with zebrafish atlas
    show_info('checking for annot_bool volume...')
    if annot_bool_dir.exists():  # when directory has 8-bit template volume, load it
        show_info('loading annot_bool volume...')
        annot_bool = np.load(annot_bool_dir)

    else:  # when saved template not found
        # check if template volume from brainglobe is already 8-bit
        show_info('... local version not found, loading annotation volume...')
        annot = BrainGlobeAtlas(atlas).annotation

        show_info('... creating annot_bool version...')

        annot_bool = np.where(annot>0, 255, 0)  # 0, outside brain, 255 inside brain
        np.save(annot_bool_dir, annot_bool)

    return annot_bool


def angleSlice(
    x_angle: float,
    y_angle: float,
    z: float,
    annot_bool: np.ndarray,
    z_idx: int,
    z_res: float,
    bregma: List[Union[int, float]],
    xyz_dict: Dict[str, Tuple[float, int]]
) -> np.ndarray:
    """
    Generate a slice of the brain annotation volume at a specific angle and z-coordinate.

    Parameters:
        x_angle (float): Angle along the x-axis in degrees.
        y_angle (float): Angle along the y-axis in degrees.
        z (float): Z-coordinate for the slice in mm.
        annot_bool (np.ndarray): Binary annotation volume.
        z_idx (int): Index of the z-coordinate.
        z_res (float): Resolution along the z-axis in mm.
        bregma (List[Union[int, float]]): Dictionary containing bregma coordinates.
        xyz_dict (Dict[str, Tuple[float, int]]): Dictionary with axis resolutions
        and dimensions (e.g., {'x': (res, dim), 'y': (res, dim), 'z': (res, dim)}).

    Returns:
        np.ndarray: A 2D slice of the annotation volume at the specified angle and z-coordinate.
    """
    # calculate from ml and dv angle, the plane of current slice
    x_shift = int(np.tan(np.deg2rad(x_angle)) * (xyz_dict['x'][1] / 2))
    y_shift = int(np.tan(np.deg2rad(y_angle)) * (xyz_dict['y'][1] / 2))
    # pick up slice
    z_coord = coord_mm_transform([z], [bregma[z_idx]],
                                 [z_res], mm_to_coord=True)

    center = np.array([z_coord, (xyz_dict['y'][1] / 2), (xyz_dict['x'][1] / 2)])
    c_right = np.array([z_coord+x_shift, (xyz_dict['y'][1] / 2), (xyz_dict['x'][1] - 1)])
    c_top = np.array([z_coord-y_shift, 0, (xyz_dict['x'][1] / 2)])
    # calculate plane normal vector
    vec_1 = c_right-center
    vec_2 = c_top-center
    vec_n = np.cross(vec_1,vec_2)
    # calculate ap matrix
    grid,r_grid_x,r_grid_y = calculateImageGrid(xyz_dict['x'][1], xyz_dict['y'][1])
    ap_mat = (-vec_n[1]*(grid[:,:,0]-center[1])-vec_n[2]*(grid[:,:,1]-center[2]))/vec_n[0] + center[0]
    ap_flat = ap_mat.astype(int).ravel()
    # within volume check
    outside_vol = np.argwhere((ap_flat<0)|(ap_flat>(xyz_dict['z'][1]-1))) # outside of volume index
    if outside_vol.size == 0: # if outside empty, inside of volume
        # index volume with ap_mat and grid
        slice = annot_bool[ap_mat.astype(int).ravel(),r_grid_y,r_grid_x].reshape(xyz_dict['y'][1],xyz_dict['x'][1])
    else: # if not empty, show black image
        slice = np.zeros((xyz_dict['y'][1], xyz_dict['x'][1]),dtype=np.uint8)
    return slice