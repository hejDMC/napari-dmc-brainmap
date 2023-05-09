import json
from natsort import natsorted
import pandas as pd
from mergedeep import merge

def get_animal_id(input_path):
    animal_id = input_path.parts[-1]
    return animal_id

def get_info(input_path, folder_id, channel=False, seg_type=False, create_dir=False, only_dir=False):

    if not seg_type:
        if channel:
            data_dir = input_path.joinpath(folder_id, channel)
        else:
            data_dir = input_path.joinpath(folder_id)
        data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.tif')])
    else:
        if channel:
            data_dir = input_path.joinpath(folder_id, seg_type, channel)
        else:
            data_dir = input_path.joinpath(folder_id, seg_type)
        data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.csv')])
    if create_dir:
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            print('creating folder under: ' + str(data_dir))
    if only_dir:
        return data_dir
    else:
        data_suffix = find_common_suffix(data_list, folder=folder_id)
        return data_dir, data_list, data_suffix


def find_common_suffix(image_list, folder='unknown'):
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
        print('only one image in ' + folder + ' folder: ' + image_list[0])
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
        common_suffix = find_common_suffix(image_list)
        image_list = [image[:-len(common_suffix)] for image in image_list]  # delete the common_suffix

        # store data as .csv file
        image_list_store = pd.DataFrame(image_list)
        image_list_store.to_csv(im_list_name)

    return image_list


def clean_params_dict(params_dict, key):
    # remove empty keys and processes that have not run
    del_list = []
    for k in params_dict[key]:
        if not params_dict[key][k]:
            del_list.append(k)
    for d in del_list:
        del params_dict[key][d]
        try:
            del params_dict[d+"_params"]
        except KeyError:
            pass
    return params_dict


def update_params_dict(input_path, params_dict):
    params_fn = input_path.joinpath('params.json')
    if params_fn.exists():
        print("params.json exists -- overriding existing values")
        with open(params_fn) as fn:
            params_dict_old = json.load(fn)
        params_dict_new = merge(params_dict_old, params_dict)
        return params_dict_new
    else:
        return params_dict


def split_strings_layers(s):
    # from: https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number
    if s.startswith('TEa'):  # special case due to small 'a', will otherwise split 'TE' + 'a1', not 'TEa' + '1'
        head = s[:3]
        tail = s[3:]
    else:
        head = s.rstrip('0123456789/ab')
        tail = s[len(head):]
    return head, tail

def clean_results_df(df, st):  # todo this somewhere seperate
    path_list = st['structure_id_path'][df['sphinx_id']]
    path_list = path_list.to_list()
    df['path_list'] = path_list
    df = df.reset_index()

    # clean data - not cells in root, fiber tracts and ventricles
    # drop "root"
    df = df.drop(df[df['name'] == 'root'].index)

    # fiber tracks and all children of it
    fiber_path = st[st['name'] == 'fiber tracts']['structure_id_path'].iloc[0]
    df = df.drop(
        df[df['path_list'].str.contains(fiber_path)].index)

    ventricle_path = st[st['name'] == 'ventricular systems']['structure_id_path'].iloc[0]
    df = df.drop(
        df[df['path_list'].str.contains(ventricle_path)].index)
    df = df.reset_index(drop=True)
    return df

def split_to_list(input_str):
    if input_str.startswith('c:'):
        return input_str[2:]
    else:
        output_str = [i for i in input_str.split(',')]
        return output_str


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

