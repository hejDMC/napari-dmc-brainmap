import sys
sys.path.append("./src")

from napari_dmc_brainmap.utils.atlas_utils import xyz_atlas_transform
from napari_dmc_brainmap.utils.atlas_utils import get_decimal
from napari_dmc_brainmap.utils.atlas_utils import coord_mm_transform
from napari_dmc_brainmap.utils.atlas_utils import sort_ap_dv_ml
from napari_dmc_brainmap.utils.atlas_utils import get_xyz
from napari_dmc_brainmap.utils.atlas_utils import get_bregma

from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.general_utils import split_to_list
from napari_dmc_brainmap.utils.general_utils import create_regi_dict
from napari_dmc_brainmap.utils.general_utils import find_key_by_value

from napari_dmc_brainmap.utils.params_utils import load_params
from napari_dmc_brainmap.utils.params_utils import clean_params_dict
from napari_dmc_brainmap.utils.params_utils import update_params_dict

import pathlib
import pytest
from unittest.mock import patch
import json



def test_get_animal_id():
    windows_path = pathlib.PureWindowsPath(r'C:\Users\username\histology\animal_id')
    assert get_animal_id(windows_path) == 'animal_id'
    posix_path = pathlib.PurePosixPath('/home/username/histology/animal_ID')
    assert get_animal_id(posix_path) == 'animal_ID'




def test_load_params(tmp_path): # tmp_path is a pytest fixture for creating a temporary directory
    # temporary directory
    (tmp_path / "animal_id").mkdir()
    input_path = tmp_path / "animal_id"
    # case 1: json file is missing
    with pytest.raises(FileNotFoundError) as excinfo:
        load_params(input_path)
    assert "['Params.json'] file missing" in str(excinfo.value)

    # case 2: json file is present
    with open(input_path / "params.json", "w") as f:
        json.dump({"key": "value"}, f)
    assert load_params(input_path) == {"key": "value"}


def test_clean_params_dict():
        params_dict = {
            'processes': {
                'proc1': True,
                'proc2': False,
                'proc3': None,
                'proc4': 'value',
                'proc5': ''
            },
            'proc2_params': {'param1': 'value1'},
            'proc3_params': {'param2': 'value2'}
        }
        key = 'processes'

        expected_output = {
            'processes': {
                'proc1': True,
                'proc4': 'value'
            }
        }
        assert clean_params_dict(params_dict, key) == expected_output


def test_update_params_dict(tmp_path):
    # Create a temporary directory
    input_path = tmp_path / "animal_id"
    input_path.mkdir()

    # Case 1: params.json file exists
    existing_params = {"existing_key": "existing_value"}
    with open(input_path / "params.json", "w") as f:
        json.dump(existing_params, f)

    new_params = {"new_key": "new_value"}
    expected_output = {"existing_key": "existing_value", "new_key": "new_value"}
    assert update_params_dict(input_path, new_params) == expected_output

    # Case 2: params.json file does not exist, create=False
    input_path_no_file = tmp_path / "animal_id_no_file"
    input_path_no_file.mkdir()
    with pytest.raises(FileNotFoundError):
        update_params_dict(input_path_no_file, new_params)

    # Case 3: params.json file does not exist, create=True
    assert update_params_dict(input_path_no_file, new_params, create=True) == new_params
    with open(input_path_no_file / "params.json") as f:
        assert json.load(f) == new_params




def test_split_to_list():
    # Test case 1: input_str is None
    assert split_to_list(None) == False

    # Test case 2: input_str is 'auto'
    assert split_to_list('auto') == 'auto'

    # Test case 3: input_str starts with 'c:'
    assert split_to_list('c:example_string') == 'example_string'

    # Test case 4: input_str with default out_format (str)
    input_str = 'a,b,c,d'
    expected_output = ['a', 'b', 'c', 'd']
    assert split_to_list(input_str) == expected_output

    # Test case 5: input_str with out_format as float
    input_str = '1.1,2.2,3.3'
    expected_output = [1.1, 2.2, 3.3]
    assert split_to_list(input_str, out_format='float') == expected_output

    # Test case 6: input_str with out_format as int
    input_str = '1,2,3,4'
    expected_output = [1, 2, 3, 4]
    assert split_to_list(input_str, out_format='int') == expected_output

    # Test case 7: input_str with other out_format
    input_str = 'a,b,c,d'
    expected_output = ['a', 'b', 'c', 'd']
    assert split_to_list(input_str, out_format='other') == expected_output



def test_get_bregma():
    # Test case 1: Known atlas ID
    atlas_id = "allen_mouse_10um"
    expected_bregma = [540, 0, 570]
    assert get_bregma(atlas_id) == expected_bregma

    # Test case 2: Known atlas ID
    atlas_id = "whs_sd_rat_39um"
    expected_bregma = [371, 72, 266]
    assert get_bregma(atlas_id) == expected_bregma

    # Test case 3: Known atlas ID
    atlas_id = "azba_zfish_4um"
    expected_bregma = [360, 0, 335]
    assert get_bregma(atlas_id) == expected_bregma

    # Test case 4: Unknown atlas ID
    atlas_id = "unknown_atlas"
    with pytest.raises(NotImplementedError):
        get_bregma(atlas_id)


def test_create_regi_dict(tmp_path):
    # Create a temporary directory and necessary subdirectories
    input_path = tmp_path / "animal_id"
    input_path.mkdir()
    sharpy_track_dir = input_path / "sharpy_track" / "regi_chan"
    sharpy_track_dir.mkdir(parents=True)

    # Create a params.json file with necessary data
    params_data = {
        "atlas_info": {
            "atlas": "allen_mouse_10um",
            "orientation": "coronal",
            "xyz_dict": {
                "x": ["ap", 1320, 10],
                "y": ["dv", 800, 10],
                "z": ["ml", 1140, 10]
            }
        }
    }
    with open(input_path / "params.json", "w") as f:
        json.dump(params_data, f)

    # Expected output
    expected_output = {
        'input_path': input_path,
        'regi_dir': sharpy_track_dir,
        'atlas': "allen_mouse_10um",
        'orientation': "coronal",
        'xyz_dict': {
            "x": ["ap", 1320, 10],
            "y": ["dv", 800, 10],
            "z": ["ml", 1140, 10]
        }
    }

    # Call the function and assert the result
    assert create_regi_dict(input_path, sharpy_track_dir) == expected_output


def test_xyz_atlas_transform():
    regi_dict = {
        'xyz_dict': {
            'x': ['rl', 1320, 10],
            'y': ['si', 800, 10],
            'z': ['ap', 1140, 10]
        }
    }

    # Test case 1: atlas_tuple matches regi_dict order
    triplet = [100, 200, 300]
    atlas_tuple = ['rl', 'si', 'ap']
    expected_output = [100, 200, 300]
    assert xyz_atlas_transform(triplet, regi_dict, atlas_tuple) == expected_output

    # Test case 2: atlas_tuple is in different order
    triplet = [100, 200, 300]
    atlas_tuple = ['si', 'rl', 'ap']
    expected_output = [200, 100, 300]
    assert xyz_atlas_transform(triplet, regi_dict, atlas_tuple) == expected_output


def test_get_decimal():
    # Test case 1: allen_mouse_10um atlas resolution 10.0um
    res_tup = [10.0]
    expected_output = [2]
    assert get_decimal(res_tup) == expected_output

    # Test case 2: whs_sd_rat_39um atlas resolution 39.0um
    res_tup = [39.0]
    expected_output = [3]
    assert get_decimal(res_tup) == expected_output

    # Test case 3: azba_zfish_4um atlas resolution 4.0um
    res_tup = [4.0]
    expected_output = [3]
    assert get_decimal(res_tup) == expected_output

    # Test case 4: multiple values in resolution tuple
    res_tup = [10.0, 10.0, 10.0]
    expected_output = [2, 2, 2]
    assert get_decimal(res_tup) == expected_output

def test_coord_mm_transform():
    # Test case 1: coord_to_mm allen_mouse_10um atlas
    triplet = [541, 1, 571]
    bregma = [540, 0, 570]
    resolution_tuple = [10.0, 10.0, 10.0]
    mm_to_coord = False
    expected_output = [-0.01, -0.01, -0.01]
    assert coord_mm_transform(triplet, bregma, resolution_tuple, mm_to_coord) == expected_output

    # Test case 2: coord_to_mm whs_sd_rat_39um atlas
    triplet = [372, 73, 267]
    bregma = [371, 72, 266]
    resolution_tuple = [39.0, 39.0, 39.0]
    mm_to_coord = False
    expected_output = [-0.039, -0.039, -0.039]
    assert coord_mm_transform(triplet, bregma, resolution_tuple, mm_to_coord) == expected_output

    # Test case 3: mm_to_coord azba_zfish_4um atlas
    triplet = [-0.004, -0.004, -0.004]
    bregma = [360, 0, 335]
    resolution_tuple = [4.0, 4.0, 4.0]
    mm_to_coord = True
    expected_output = [361, 1, 336]
    assert coord_mm_transform(triplet, bregma, resolution_tuple, mm_to_coord) == expected_output


def test_sort_ap_dv_ml():
    # Test case 1: atlas_tuple matches target tuple order
    triplet = [1.1, 2.2, 3.3]
    atlas_tuple = ['ap', 'si', 'rl']
    expected_output = [1.1, 2.2, 3.3]
    assert sort_ap_dv_ml(triplet, atlas_tuple) == expected_output

    # Test case 2: atlas_tuple is in different order
    triplet = [1.1, 2.2, 3.3]
    atlas_tuple = ['si', 'ap', 'rl']
    expected_output = [2.2, 1.1, 3.3]
    assert sort_ap_dv_ml(triplet, atlas_tuple) == expected_output


def test_get_xyz():
    # Mock the BrainGlobeAtlas object
    atlas_mock = patch('napari_dmc_brainmap.utils.atlas_utils.BrainGlobeAtlas').start()
    atlas_instance = atlas_mock.return_value
    atlas_instance.space.sections = ['frontal', 'horizontal', 'sagittal']
    atlas_instance.space.index_pairs = [(1, 2), (0, 2), (0, 1)]
    atlas_instance.space.axes_description = ['ap', 'dv', 'ml']
    atlas_instance.space.shape = [1320, 800, 1140]
    atlas_instance.space.resolution = [10.0, 10.0, 10.0]

    # Test case 1: coronal section
    section_orient = 'coronal'
    expected_output = {
        'x': ['ml', 1140, 10.0],
        'y': ['dv', 800, 10.0],
        'z': ['ap', 1320, 10.0]
    }
    assert get_xyz(atlas_instance, section_orient) == expected_output

    # Test case 2: horizontal section
    section_orient = 'horizontal'
    expected_output = {
        'x': ['ml', 1140, 10.0],
        'y': ['ap', 1320, 10.0],
        'z': ['dv', 800, 10.0]
    }
    assert get_xyz(atlas_instance, section_orient) == expected_output

    # Test case 3: sagittal section
    section_orient = 'sagittal'
    expected_output = {
        'x': ['dv', 800, 10.0],
        'y': ['ap', 1320, 10.0],
        'z': ['ml', 1140, 10.0]
    }
    assert get_xyz(atlas_instance, section_orient) == expected_output
    # Stop the patch
    patch.stopall()




def test_find_key_by_value():
    # Test case 1: key exists
    input_dict = {
        'key1': 'value1',
        'key2': 'value2',
        'key3': 'value3'
    }
    value = 'value2'
    expected_output = 'key2'
    assert find_key_by_value(input_dict, value) == expected_output

    # Test case 2: key does not exist
    value = 'value4'
    expected_output = None
    assert find_key_by_value(input_dict, value) == expected_output




