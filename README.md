
# napari-dmc-brainmap
*DMC-BrainMap is an end-to-end tool for multi-feature brain mapping across species.*  
This [napari](https://napari.org/stable/) plugin was generated with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) using napari's [cookiecutter-napari-plugin](https://github.com/napari/cookiecutter-napari-plugin) template.

[![Tests](https://github.com/hejDMC/napari-dmc-brainmap/actions/workflows/test_and_deploy.yml/badge.svg?branch=main)](https://github.com/hejDMC/napari-dmc-brainmap/actions/workflows/test_and_deploy.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/hejDMC/napari-dmc-brainmap/branch/main/graph/badge.svg)](https://app.codecov.io/gh/hejDMC/napari-dmc-brainmap)
[![PyPI](https://img.shields.io/pypi/v/napari-dmc-brainmap.svg?color=green)](https://pypi.org/project/napari-dmc-brainmap)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-dmc-brainmap)](https://napari-hub.org/plugins/napari-dmc-brainmap.html)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://napari-dmc-brainmap.readthedocs.io/en/latest/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](https://github.com/hejDMC/napari-dmc-brainmap/blob/main/LICENSE.txt)


## Quick start
User guides and tutorials are available in the
[project wiki](https://github.com/hejDMC/napari-dmc-brainmap/wiki). The
generated technical documentation is published on
[Read the Docs](https://napari-dmc-brainmap.readthedocs.io/en/latest/).

### Installation

DMC-BrainMap is a plugin for [napari](https://napari.org/stable/). There are two recommended installation paths, depending on whether you want to use the released plugin or develop the code.

DMC-BrainMap supports Python 3.11 through 3.13. On macOS, Apple Silicon is required; Intel Macs are not supported.

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

##### Optional registration prediction

PyTorch is optional and is not installed with the standard plugin. To use the
registration prediction feature in a pip or napari environment, install the
CPU-only build separately:

```bash
python -m pip install "torch>=2.12.0" --index-url https://download.pytorch.org/whl/cpu
```

See the
[prediction-assisted registration guide](https://napari-dmc-brainmap.readthedocs.io/en/latest/registration_prediction.html)
for model download, setup, usage, and quality-control guidance.

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

This creates the project environment with a compatible Python version, installs
napari and all required dependencies, and installs
`napari-dmc-brainmap` from the local checkout. Optional prediction support is
installed only when the `prediction` extra is selected as described below.

##### Optional registration prediction

Enable the `prediction` extra when synchronizing the development environment:

```bash
uv sync --extra prediction
```

To include both prediction and test dependencies, use:

```bash
uv sync --extra prediction --group test
```

`uv sync --all-extras` also installs prediction support. A plain
`uv sync` intentionally excludes optional extras and may remove a previously
installed PyTorch package during exact synchronization. PyTorch remains
recorded in `uv.lock` so the selected extra stays reproducible.

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

- [Technical documentation](https://napari-dmc-brainmap.readthedocs.io/en/latest/)
- [Prediction-assisted registration](https://napari-dmc-brainmap.readthedocs.io/en/latest/registration_prediction.html)
- [User guides and tutorials](https://github.com/hejDMC/napari-dmc-brainmap/wiki)

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
