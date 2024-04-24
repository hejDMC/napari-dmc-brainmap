import json
from natsort import natsorted
import pandas as pd
from mergedeep import merge
from enum import Enum
from bg_atlasapi.list_atlases import descriptors, utils
import numpy as np


def get_animal_id(input_path):
    animal_id = input_path.parts[-1]
    return animal_id


def get_info(input_path, folder_id, channel=False, seg_type=False, create_dir=False, only_dir=False):
    if not seg_type:
        if channel:
            data_dir = input_path.joinpath(folder_id, channel)
        else:
            data_dir = input_path.joinpath(folder_id)
        if data_dir.exists():
            data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.tif')])
        else:
            data_list = []
    else:
        if channel:
            data_dir = input_path.joinpath(folder_id, seg_type, channel)
        else:
            data_dir = input_path.joinpath(folder_id, seg_type)
        if data_dir.exists():
            data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.csv')])
        else:
            data_list = []
    if create_dir:
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            print('creating folder under: ' + str(data_dir))
    if only_dir:
        return data_dir
    else:
        if len(data_list) > 0:
            data_suffix = find_common_suffix(data_list, input_path=input_path, folder=folder_id, im_list_present=True)
        else:
            data_suffix = ''
        return data_dir, data_list, data_suffix


def find_common_suffix(image_list, input_path=False, folder='unknown', im_list_present=False):
    # if image list is present, load and get suffix
    if im_list_present:
        im0 = image_list[0]
        im_list = get_im_list(input_path)
        im1 = [i for i in im_list if im0.startswith(i)]
        if len(im1) == 1:
            im1 = im1[0]
            common_suffix = im0[len(im1):]
        elif len(im1) == 2:
            im1 = im1[1]
            common_suffix = im0[len(im1):]
        else:
            common_suffix = input("please, manually enter suffix: ")
    else:
        if len(image_list) > 1:
            for i in range(len(image_list[0])):
                if i > 0:
                    if image_list[0][-i] == image_list[1][-i]:
                        continue
                    else:
                        break
            common_suffix = image_list[0][-i + 1:]
            # print("estimated common_suffix for " + folder + " folder: " + common_suffix)
        elif len(image_list) == 1:
            print('only one file in ' + folder + ' folder: ' + image_list[0])
            print(
                'in DMC-BrainMap an image name has a base_string and a suffix. For an image named *animal1_obj1_1_stitched.tif* '
                '\n the base_string is animal1_obj1_1 and the suffix is _stitched.tif')
            common_suffix = input("please, manually enter suffix: ")
        else:
            common_suffix = []
    return common_suffix


def get_im_list(input_path, folder_id='stitched', file_id='*.tif'):
    im_list_name = input_path.joinpath('image_names.csv')
    if im_list_name.exists():
        image_list = pd.read_csv(im_list_name)
        image_list = image_list['0'].to_list()

    else:
        data_dir = get_info(input_path, folder_id, only_dir=True)
        if folder_id == 'confocal':
            image_list = natsorted([f.parts[-1] for f in data_dir.glob(file_id)])
        else:
            filter_dir = [f for f in data_dir.glob('**/*') if f.is_dir()][0]  # just take the first folder
            image_list = natsorted([f.parts[-1] for f in filter_dir.glob(file_id)])
        common_suffix = find_common_suffix(image_list, folder=folder_id)
        image_list = [image[:-len(common_suffix)] for image in image_list]  # delete the common_suffix

        # store data as .csv file
        image_list_store = pd.DataFrame(image_list)
        image_list_store.to_csv(im_list_name)

    return image_list


def load_params(input_path):
    params_fn = input_path.joinpath('params.json')
    if params_fn.exists():
        with open(params_fn) as fn:
            params_dict = json.load(fn)
        return params_dict
    else:
        raise FileNotFoundError(" ['Params.json'] file missing for " + get_animal_id(input_path) + " \n"
                                "Check Data Integrity at folder: {} \n"
                                "and try again!".format(input_path))
    


def clean_params_dict(params_dict, key):
    # remove empty keys and processes that have not run
    del_list = []
    for k in params_dict[key]:
        if not params_dict[key][k]:
            del_list.append(k)
    for d in del_list:
        del params_dict[key][d]
        try:
            del params_dict[d + "_params"]
        except KeyError:
            pass
    return params_dict


def update_params_dict(input_path, params_dict, create=False):
    params_fn = input_path.joinpath('params.json')
    if params_fn.exists():
        print("params.json exists -- overriding existing values")
        with open(params_fn) as fn:
            params_dict_old = json.load(fn)
        params_dict_new = merge(params_dict_old, params_dict)

        with open(params_fn, 'w') as fn:
            json.dump(params_dict_new, fn, indent=4)

        return params_dict_new
    elif create:
        with open(params_fn, 'w') as fn:
            json.dump(params_dict, fn, indent=4)
        return params_dict
    else:
        raise FileNotFoundError(" ['Params.json'] file missing for " + get_animal_id(input_path) + " \n"
                                "Check Data Integrity at folder: {} \n"
                                "and try again!".format(input_path))


def split_strings_layers(s):
    # from: https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number
    if s.startswith('TEa'):  # special case due to small 'a', will otherwise split 'TE' + 'a1', not 'TEa' + '1'
        head = s[:3]
        tail = s[3:]
    elif s.startswith('CA'):
        head = s
        tail = []
    else:
        head = s.rstrip('0123456789/ab')
        tail = s[len(head):]
    return head, tail


def get_parent(a, st):
    # dummy function to get parent id for quick and dirty quantification of injection side
    level_dict = {
        7: ['Isocortex', 'HPF', 'TH'],
        6: ['OLF', 'HY'],
        5: ['CTXsp', 'CNU', 'MB', 'CB'],
        4: ['HB']
    }
    a_parent = a
    for k, v in level_dict.items():
        for i in v:
            if st[st['acronym'] == i]['structure_id_path'].iloc[0] in a:
                if len(a.split('/')) > (k + 3):
                    a_parent = st[st['structure_id_path'] == '/'.join(a.split('/')[:k + 2]) + '/']['acronym'].iloc[0]
                break

    return a_parent


def clean_results_df(df, atlas):
    list_delete = ['root']
    for item in ['fiber tracts', 'VS']:
        list_delete.append(atlas.get_structure_descendants(item))
    list_delete = [l for sublist in list_delete for l in sublist]  # flatten list
    df = df.drop(df[df['acronym'].isin(list_delete)].index)
    df = df.reset_index(drop=True)

    return df


def split_to_list(input_str, out_format='str'):
    if not input_str:
        output_list = False
    else:
        if input_str.startswith('c:'):
            return input_str[2:]
        else:
            if out_format == 'str':
                output_list = [i for i in input_str.split(',')]
            elif out_format == 'float':
                output_list = [float(i) for i in input_str.split(',')]
            elif out_format == 'int':
                output_list = [int(i) for i in input_str.split(',')]
            else:
                output_list = [i for i in input_str.split(',')]

    return output_list


def load_group_dict(input_path, animal_list, group_id='genotype'):
    dict = {}
    for animal_id in animal_list:
        data_dir = input_path.joinpath(animal_id)
        params_fn = data_dir.joinpath('params.json')
        if params_fn.exists():
            with open(params_fn) as fn:
                params_dict = json.load(fn)
            try:
                g_id = params_dict['general'][group_id]
                if g_id in dict.keys():
                    dict[g_id].append(animal_id)
                else:
                    dict[g_id] = [animal_id]
            except KeyError:
                print("no group_id value (*" + group_id + "*) specified for " + animal_id)
                print("    --> skipping " + animal_id)
                pass

        else:
            print("No params.json file under " + str(params_fn))
            print("    --> skipping " + animal_id)

    return dict


def get_bregma(atlas_id):
    bregma_dict = {
        "allen_mouse_10um": [540, 0, 570],
        # Ref: https://github.com/cortex-lab/allenCCF/blob/master/Browsing%20Functions/allenCCFbregma.m
        "whs_sd_rat_39um": [371, 72, 266],
        # Ref: Papp EA. Neuroimage. 2014 Aug 15;97:374-86. doi: 10.1016/j.neuroimage.2014.04.001.
        "azba_zfish_4um": [360, 0, 335]
        # Ref: cannot really find one, by looking at the saggital section, it seems around 350 to 400 AP voxel
    }
    if atlas_id in bregma_dict.keys():
        bregma = bregma_dict[atlas_id]
    else:
        print('no bregma coordinates specified for ' + atlas_id + '\n'
                                                                  ' returning 0/0/0 coordinates as *bregma*')
        bregma = [0, 0, 0]  # FIXME half of x and z-axis size

    return bregma


def create_regi_dict(input_path, regi_chan):
    regi_dir = get_info(input_path, 'sharpy_track', channel=regi_chan, only_dir=True)
    params_dict = load_params(input_path)

    regi_dict = {
        'input_path': input_path,
        'regi_dir': regi_dir,
        'atlas': params_dict['atlas_info']['atlas'],
        'orientation': params_dict['atlas_info']['orientation'],
        'xyz_dict': params_dict['atlas_info']['xyz_dict']
    }

    return regi_dict


def xyz_atlas_transform(triplet, regi_dict, atlas_tuple):
    # change indices of xyz triplet tuple to match atlas
    # list with [x,y,z] triplet
    xyz_tuple = tuple([regi_dict['xyz_dict']['x'][0], regi_dict['xyz_dict']['y'][0], regi_dict['xyz_dict']['z'][0]])
    index_match = [xyz_tuple.index(e) for e in atlas_tuple]

    triplet_new = [triplet[i] for i in index_match]

    return triplet_new


def get_decimal(res_tup):  # res_tup is a list of arbitrary number of resolution(um) values
    decimal_list = []
    for r in res_tup:
        step_float = r / 1000
        # by default keep 2 decimals
        decimal = 2
        while np.abs(np.round(step_float, decimal) - step_float) >= 0.01 * step_float:
            decimal += 1
        decimal_list.append(decimal)
    return decimal_list


def consider_decimals(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        if len(args) == 4:
            pass
        else:
            if len(kwargs) == 0:
                args.append(False)
            else:
                args.append(kwargs['mm_to_coord'])
        triplet, bregma, resolution_tuple, mm_to_coord = args  # take out arguments

        ## get z_step decimals
        decimal_list = get_decimal(resolution_tuple)

        if mm_to_coord:
            triplet_new = [round(- coord / (res / 1000)) + br_coord for coord, br_coord, res in
                           zip(triplet, bregma, resolution_tuple)]
        else:
            triplet_new = []
            for coord, br_coord, res, decimal in zip(triplet, bregma, resolution_tuple, decimal_list):
                triplet_new.append(round((br_coord - coord) * (res / 1000), decimal))

        if len(triplet_new) == 1:
            return triplet_new[0]
        else:
            return triplet_new

    return wrapper


@consider_decimals
def coord_mm_transform(triplet, bregma, resolution_tuple, mm_to_coord=False):
    if mm_to_coord:
        triplet_new = [round(- coord / (res / 1000)) + br_coord for coord, br_coord, res in
                       zip(triplet, bregma, resolution_tuple)]
    else:
        triplet_new = [round((br_coord - coord) * (res / 1000), 2) for coord, br_coord, res in
                       zip(triplet, bregma, resolution_tuple)]
    if len(triplet_new) == 1:
        return triplet_new[0]
    else:
        return triplet_new


def sort_ap_dv_ml(triplet, atlas_tuple):
    # sort input triplet in respective atlas convention to new tipled in [ap, dv, ml] order
    tgt_tuple = ('ap', 'si', 'rl')  # bg naming convention
    index_match = [atlas_tuple.index(e) for e in tgt_tuple]
    triplet_new = [triplet[i] for i in index_match]
    return triplet_new


def get_xyz(atlas, section_orient):
    # resolution tuple (width, height)
    orient_dict = {
        'coronal': 'frontal',
        'horizontal': 'horizontal',
        'sagittal': 'sagittal'
    }

    orient_idx = atlas.space.sections.index(orient_dict[section_orient])
    resolution_idx = atlas.space.index_pairs[orient_idx]
    xyz_dict = {
        'x': [atlas.space.axes_description[resolution_idx[1]], atlas.space.shape[resolution_idx[1]],
              atlas.space.resolution[resolution_idx[1]]],
        'y': [atlas.space.axes_description[resolution_idx[0]], atlas.space.shape[resolution_idx[0]],
              atlas.space.resolution[resolution_idx[0]]],
        'z': [atlas.space.axes_description[orient_idx], atlas.space.shape[orient_idx],
              atlas.space.resolution[orient_idx]]
    }

    return xyz_dict


def get_available_atlases():
    """
    from: https://github.com/brainglobe/brainreg-segment  -- July 2023
    Get the available brainglobe atlases
    :return: Dict of available atlases (["name":version])
    """
    available_atlases = utils.conf_from_url(
        descriptors.remote_url_base.format("last_versions.conf")
    )
    available_atlases = dict(available_atlases["atlases"])
    # move "example_mouse_100um" to the back of the list
    available_atlases = {k: available_atlases[k] for k in available_atlases if k != 'example_mouse_100um'} \
                        | {k: available_atlases[k] for k in ['example_mouse_100um'] if k in available_atlases}
    return available_atlases


def get_atlas_dropdown():
    atlas_dict = {}
    for i, k in enumerate(get_available_atlases().keys()):
        atlas_dict.setdefault(k, k)
    atlas_keys = Enum("atlas_key", atlas_dict)
    return atlas_keys