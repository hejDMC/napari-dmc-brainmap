from natsort import natsorted
import pandas as pd

def get_animal_id(input_path):
    animal_id = input_path.parts[-1]
    return animal_id

def get_info(input_path, folder_id, seg_type=False, create_dir=False, only_dir=False):

    if not seg_type:
        data_dir = input_path.joinpath(folder_id)
        data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.tif')])
    else:
        data_dir = input_path.joinpath(folder_id, seg_type)
        data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.csv')])
    if create_dir:
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            print('creating stats folder under: ' + str(data_dir))
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
        common_suffix = input("only one image in folder, manually enter suffix for " + folder + " folder")
    else:
        common_suffix = []
    return common_suffix

def get_im_list(input_path):

    im_list_name = input_path.joinpath('image_names.csv')
    if im_list_name.exists():
        image_list = pd.read_csv(im_list_name)
    else:
        stitched_dir = get_info(input_path, 'stitched', only_dir=True)
        filter_dir = [f for f in stitched_dir.glob('**/*') if f.is_dir()][0]  # just take the first folder
        image_list = natsorted([f.parts[-1] for f in filter_dir.glob('*.tif')])
        common_suffix = find_common_suffix(image_list)
        image_list = [image[:-len(common_suffix)] for image in image_list]  # delete the common_suffix

        # store data as .csv file
        image_list = pd.DataFrame(image_list)
        image_list.to_csv(im_list_name)
    image_list = image_list['0'].to_list()
    return image_list