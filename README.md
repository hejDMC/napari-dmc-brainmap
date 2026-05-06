
# napari-dmc-brainmap
*DMC-BrainMap is an end-to-end tool for multi-feature brain mapping across species.*  
This [napari](https://napari.org/stable/) plugin was generated with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) using napari's [cookiecutter-napari-plugin](https://github.com/napari/cookiecutter-napari-plugin) template.

[![License BSD-3](https://img.shields.io/pypi/l/napari-dmc-brainmap.svg?color=green)](https://github.com/hejDMC/napari-dmc-brainmap/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-dmc-brainmap.svg?color=green)](https://pypi.org/project/napari-dmc-brainmap)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-dmc-brainmap.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-dmc-brainmap)](https://napari-hub.org/plugins/napari-dmc-brainmap)


## Quick start
A detailed guide and tutorial can be found on the [Wiki pages of this repo](https://github.com/hejDMC/napari-dmc-brainmap/wiki).

### Installation

DMC-BrainMap is a plugin for [napari](https://napari.org/stable/). There are two recommended installation paths, depending on whether you want to use the released plugin or develop the code.

#### Users

For reproducible regular use, first install napari by following the [napari installation guide](https://napari.org/dev/getting_started/installation.html). Then install DMC-BrainMap from napari's graphical plugin manager:

```text
Plugins > Install/Uninstall Plugins...
```

Search for `napari-dmc-brainmap`, then install it from the plugin manager. Napari handles the plugin installation graphically.

You can also install the released plugin with pip:

```bash
pip install napari-dmc-brainmap
```

After installation, open DMC-BrainMap from the napari plugin menu:

```text
Plugins > dmc_brainmap
```

#### Developers

For troubleshooting and contributing to repo development, install the repository as an editable project with [uv](https://docs.astral.sh/uv/). In editable mode, code changes in this repository are picked up the next time napari is started with `uv run napari`.

First install uv if needed, following the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository:

```bash
git clone https://github.com/hejDMC/napari-dmc-brainmap.git
cd napari-dmc-brainmap
```

Sync the environment:

```bash
uv sync
```

This creates the project environment, installs Python 3.10 as required by the project, installs napari and all dependencies, and installs `napari-dmc-brainmap` from the local checkout.

Start napari:

```bash
uv run napari
```

Then find DMC-BrainMap from the napari plugin menu:

```text
Plugins > dmc_brainmap
```

### Usage

Please refer to the Wiki pages for detailed instructions and a short tutorial on how to use DMC-BrainMap. When working with DMC-BrainMap on your own data, please keep the following points in mind:
- DMC-BrainMap requires single-channel 16-bit .tif/.tiff images to work (in principle 8-bit also work)
- DMC-BrainMap requires that your data is organized by animals in separate folders (you can pool data later down the lane)
- DMC-BrainMap uses 5 channel labels (`dapi`, `green`, `n3`, `cy3`, `cy5`) corresponding to blue, green, orange, red and far red channels. *However, these are only labels, you can assign them as you please. Hence, you can use DMC-BrainMap also for non-fluorescence data given you converted your images to single-channel 16-bit .tif/.tiff images*. Please contact us if you need to use more than 5 channels.
- It is essential that you structure your data in the following way (hierarchical organization, same name for images in different channels, channel labels are selected by you), **otherwise DMC-BrainMap won't work**:
```
animal_id-001
│
└───stitched
│   │
│   └───dapi
│   |    │   animal_id-001_001.tiff
│   |    │   animal_id-001_002.tiff
|   │    |   animal_id-001_003.tiff
│   |    │   animal_id-001_004.tiff
│   |    │   ...
│   │   
│   └───green
│       │   animal_id-001_001.tiff
│       │   animal_id-001_002.tiff
│       │   animal_id-001_003.tiff
│       │   animal_id-001_004.tiff
│       │   ...
│   
animal_id-2
│   ...
```

## Documentation
Documentation on DMC-BrainMap's source code can be found on the project's [Read the Docs page](https://napari-dmc-brainmap.readthedocs.io/en/latest/#).

## Seeking help or contributing

DMC-BrainMap is an open-source project, and we welcome contributions of all kinds. If you have any questions, feedback, or suggestions, please feel free to open an issue on this repository. 

## License

Distributed under the terms of the [BSD-3](https://github.com/teamdigitale/licenses/blob/master/BSD-3-Clause) license,
"napari-dmc-brainmap" is free and open source software

## Citing DMC-BrainMap

If you use DMC-BrainMap in your scientific work, please cite:
```
Jung, F., Cao, X., Heymans, L., Carlén, M. (2026) "DMC-BrainMap is an open-source, end-to-end tool for multi-feature brain mapping across species", Cell Reports Methods, https://doi.org/10.1016/j.crmeth.2026.101302
```

BibTeX:  
``` bibtex
@article{Jung2026a,
title = {DMC-BrainMap is an open-source, end-to-end tool for multi-feature brain mapping in different species},
journal = {Cell Reports Methods},
volume = {6},
number = {2},
pages = {101302},
year = {2026},
issn = {2667-2375},
doi = {https://doi.org/10.1016/j.crmeth.2026.101302},
url = {https://www.sciencedirect.com/science/article/pii/S2667237526000020},
author = {Felix Jung and Xiao Cao and Loran Heymans and Marie Carlén}
}
```