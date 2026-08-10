from inspect import Parameter, signature
from pathlib import Path

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.core import Atlas
from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions
from brainglobe_atlasapi.structure_class import StructuresDict
from brainglobe_space import AnatomicalSpace


STRUCTURES = [
    {
        "acronym": "root",
        "id": 997,
        "name": "root",
        "structure_id_path": [997],
        "rgb_triplet": [255, 255, 255],
    },
    {
        "acronym": "CHILD",
        "id": 1,
        "name": "Child region",
        "structure_id_path": [997, 1],
        "rgb_triplet": [10, 20, 30],
    },
    {
        "acronym": "LEAF",
        "id": 2,
        "name": "Leaf region",
        "structure_id_path": [997, 1, 2],
        "rgb_triplet": [40, 50, 60],
    },
]


def _offline_atlas() -> Atlas:
    atlas = object.__new__(Atlas)
    atlas.metadata = {
        "name": "contract_mouse",
        "shape": [2, 2, 2],
        "resolution": [100.0, 100.0, 100.0],
    }
    atlas.structures = StructuresDict(STRUCTURES)
    atlas._template = np.arange(8, dtype=np.uint16).reshape((2, 2, 2))
    atlas._annotation = np.asarray(
        [
            [[997, 1], [2, 0]],
            [[1, 2], [0, 997]],
        ],
        dtype=np.uint32,
    )
    return atlas


def test_v3_entry_points_match_the_plugin_contract() -> None:
    constructor_parameters = signature(BrainGlobeAtlas).parameters
    listing_parameters = signature(get_all_atlases_lastversions).parameters
    required_listing_parameters = [
        parameter
        for parameter in listing_parameters.values()
        if parameter.default is Parameter.empty
        and parameter.kind
        not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    ]

    assert "atlas_name" in constructor_parameters
    assert not required_listing_parameters
    for attribute in (
        "annotation",
        "get_structure_ancestors",
        "get_structure_descendants",
        "hierarchy",
        "resolution",
        "shape",
        "structure_from_coords",
        "template",
    ):
        assert hasattr(Atlas, attribute)


def test_v3_atlas_identity_and_version_contract(tmp_path: Path) -> None:
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = "contract_mouse_100um"
    atlas.metadata = {"name": "contract_mouse", "version": "3.0"}
    atlas.root_dir = tmp_path / "brainglobe-atlasapi"
    atlas._local_version = None

    assert atlas.atlas_name == "contract_mouse_100um"
    assert atlas.metadata["name"] == "contract_mouse"
    assert atlas.local_version == (3, 0)
    assert isinstance(atlas.root_dir, Path)


def test_v3_anatomical_space_contract() -> None:
    space = AnatomicalSpace(
        origin="asr",
        shape=(132, 80, 114),
        resolution=(100.0, 100.0, 100.0),
    )

    assert space.axes_description == ("ap", "si", "rl")
    assert space.sections == ("frontal", "horizontal", "sagittal")
    assert tuple(tuple(pair) for pair in space.index_pairs) == (
        (1, 2),
        (0, 2),
        (0, 1),
    )
    assert space.shape == (132, 80, 114)
    assert space.resolution == (100.0, 100.0, 100.0)


def test_v3_structure_mapping_and_hierarchy_contract() -> None:
    atlas = _offline_atlas()

    assert atlas.structures["CHILD"]["id"] == 1
    assert atlas.structures[np.uint32(1)]["acronym"] == "CHILD"
    assert atlas.structures.acronym_to_id_map["LEAF"] == 2
    assert atlas.structures.data[2]["structure_id_path"] == [997, 1, 2]
    assert atlas.hierarchy.is_branch(997) == [1]
    assert atlas.get_structure_ancestors(2) == ["root", "CHILD"]
    assert atlas.get_structure_descendants(997) == ["CHILD", "LEAF"]


def test_v3_volume_and_coordinate_lookup_contract() -> None:
    atlas = _offline_atlas()

    assert isinstance(atlas.template, np.ndarray)
    assert atlas.template.dtype == np.dtype(np.uint16)
    assert isinstance(atlas.annotation, np.ndarray)
    assert atlas.annotation.dtype == np.dtype(np.uint32)
    assert atlas.shape == (2, 2, 2)
    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.structure_from_coords((0, 1, 0)) == 2
    assert atlas.structure_from_coords((0, 1, 0), as_acronym=True) == "LEAF"
