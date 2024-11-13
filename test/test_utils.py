from napari_dmc_brainmap.utils import get_animal_id
from napari_dmc_brainmap.utils import get_info
from napari_dmc_brainmap.utils import find_common_suffix
from napari_dmc_brainmap.utils import get_image_list
from napari_dmc_brainmap.utils import chunk_list
from napari_dmc_brainmap.utils import load_params
from napari_dmc_brainmap.utils import clean_params_dict
import pathlib
import pytest
from unittest.mock import patch
import json




def test_get_animal_id():
    windows_path = pathlib.PureWindowsPath(r'C:\Users\username\histology\animal_id')
    assert get_animal_id(windows_path) == 'animal_id'
    posix_path = pathlib.PurePosixPath('/home/username/histology/animal_ID')
    assert get_animal_id(posix_path) == 'animal_ID'


def test_get_info():
    # skip this for now
    assert True is True


def test_find_common_suffix():
    # skip this for now
    assert True is True


def test_get_image_list():
    # skip this for now
    assert True is True


def test_chunk_list():
    input_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    chunk_size = 4
    expected_output = [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j"]]
    assert chunk_list(input_list, chunk_size) == expected_output
    chunk_size = 1
    expected_output = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"], ["i"], ["j"]]
    assert chunk_list(input_list, chunk_size) == expected_output
    chunk_size = 12
    expected_output = [["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]]
    assert chunk_list(input_list, chunk_size) == expected_output



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
    # continue on this
    assert True is True