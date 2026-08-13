from collections import Counter

import numpy as np
from probeinterface import read_spikeglx
from probeinterface.neuropixels_tools import build_neuropixels_probe


def test_spikeglx_reader_is_available_from_public_api() -> None:
    assert callable(read_spikeglx)


def test_np2013_catalogue_geometry_matches_four_shank_contract() -> None:
    probe = build_neuropixels_probe(probe_part_number="NP2013")

    assert probe.get_contact_count() == 5120
    assert probe.si_units == "um"
    assert probe.model_name == "NP2013"
    assert probe.annotations["num_readout_channels"] == 384
    assert Counter(probe.shank_ids) == {
        "0": 1280,
        "1": 1280,
        "2": 1280,
        "3": 1280,
    }

    shank_left_edges = [
        np.min(probe.contact_positions[probe.shank_ids == shank_id, 0])
        for shank_id in ("0", "1", "2", "3")
    ]
    assert shank_left_edges == [0.0, 250.0, 500.0, 750.0]

    vertical_positions = np.unique(probe.contact_positions[:, 1])
    assert np.all(np.diff(vertical_positions) == 15.0)

    columns = set(probe.to_dataframe(complete=True).columns)
    assert {"x", "y", "shank_ids", "contact_ids", "si_units"} <= columns
