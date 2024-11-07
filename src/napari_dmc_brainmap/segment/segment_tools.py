import numpy as np
from bg_atlasapi import config, BrainGlobeAtlas
from napari_dmc_brainmap.utils import coord_mm_transform

def calculateImageGrid(x_res, y_res): # one time calculation
    y = np.arange(y_res)
    x = np.arange(x_res)
    grid_x, grid_y = np.meshgrid(x, y)
    r_grid_x = grid_x.ravel()
    r_grid_y = grid_y.ravel()
    grid = np.stack([grid_y, grid_x], axis=2)
    return grid, r_grid_x, r_grid_y

def loadAnnotBool(atlas):
    brainglobe_dir = config.get_brainglobe_dir()
    atlas_name_general = f"{atlas}_v*"
    atlas_names_local = list(brainglobe_dir.glob(atlas_name_general))[
        0]  # glob returns generator object, need to exhaust it in list, then take out
    annot_bool_dir = brainglobe_dir.joinpath(atlas_names_local, 'annot_bool.npy')
    # for any atlas else, in this case test with zebrafish atlas
    print('checking for annot_bool volume...')
    if annot_bool_dir.exists():  # when directory has 8-bit template volume, load it
        print('loading annot_bool volume...')
        annot_bool = np.load(annot_bool_dir)

    else:  # when saved template not found
        # check if template volume from brainglobe is already 8-bit
        print('... local version not found, loading annotation volume...')
        annot = BrainGlobeAtlas(atlas).annotation

        print('... creating annot_bool version...')

        annot_bool = np.where(annot>0, 255, 0)  # 0, outside brain, 255 inside brain
        np.save(annot_bool_dir, annot_bool)

    return annot_bool


def angleSlice(x_angle, y_angle, z, annot_bool, z_idx, z_res, bregma, xyz_dict):
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


def get_cmap(name):
    cmaps = {
        'cells': {
            'dapi': 'yellow',
            'green': 'magenta',
            'n3': 'gray',
            'cy3': 'cyan',
            'cy5': 'lightblue'
        },
        'npx': {
            '0': 'deepskyblue',
            '1': 'orange',
            '2': 'springgreen',
            '3': 'darkgray',
            '4': 'fuchsia',
            '5': 'royalblue',
            '6': 'gold',
            '7': 'powderblue',
            '8': 'lightsalmon',
            '9': 'olive'
        },
        'injection': {
            'dapi': 'gold',
            'green': 'purple',
            'n3': 'navy',
            'cy3': 'darkorange',
            'cy5': 'cornflowerblue'
        },
        'display': {
            'dapi': 'blue',
            'green': 'green',
            'n3': 'orange',
            'cy3': 'red',
            'cy5': 'pink'
        }
    }
    return cmaps[name]