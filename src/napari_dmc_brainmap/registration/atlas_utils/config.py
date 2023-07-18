"""
Utilities to manage atlas download -- similar to bg_atlasapi approach

"""

import configparser
from pathlib import Path
from pkg_resources import resource_filename

CONFIG_FILENAME = "atlas_config.conf"
CONFIG_PATH = Path(resource_filename("napari_dmc_brainmap", CONFIG_FILENAME))

# set path to atlas storage location
DEFAULT_PATH = Path.home() / ".dmc-brainmap_atlas"
CONF_DICT = {
    "default_dirs": {
        "atlas_dir": DEFAULT_PATH
    }
}


def write_config(path=CONFIG_PATH, template=CONF_DICT):

    conf = configparser.ConfigParser()
    for k, val in template.items():
        conf[k] = val

    with open(path, "w") as f:
        conf.write(f)


def read_config(path=CONFIG_PATH):

    # If no config file exists yet, write the default one:
    if not path.exists():
        write_config()

    conf = configparser.ConfigParser()
    conf.read(path)
    return conf