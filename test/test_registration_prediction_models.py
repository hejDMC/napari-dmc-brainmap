"""Real-PyTorch checks for state-dict prediction checkpoints.

Skipped when the optional ``prediction`` extra (PyTorch) is not installed.
"""
import hashlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.append("./src")

from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.prediction import (  # noqa: E402
    RegistrationTransformPredictor,
    _PyramidOffsetModel,
    _SpatialOffsetModel,
)


TINY_PYRAMID_CONFIG = {
    "model_name": "pyramid",
    "encoder_base_channels": 8,
    "pyramid_channels": 16,
    "pyramid_head_channels": 8,
    "spatial_pool_size": [2, 3],
    "spatial_hidden_channels": 32,
    "max_corner_offset_px": 512.0,
}
TINY_SPATIAL_CONFIG = {
    "model_name": "spatial",
    "base_channels": 4,
    "max_corner_offset_px": 512.0,
    "spatial_pool_size": [2, 3],
    "spatial_hidden_channels": 16,
}


def _build(config):
    if config["model_name"] == "pyramid":
        return _PyramidOffsetModel(
            torch,
            encoder_base_channels=config["encoder_base_channels"],
            pyramid_channels=config["pyramid_channels"],
            pyramid_head_channels=config["pyramid_head_channels"],
            spatial_pool_size=tuple(config["spatial_pool_size"]),
            spatial_hidden_channels=config["spatial_hidden_channels"],
            max_corner_offset_px=config["max_corner_offset_px"],
        )
    return _SpatialOffsetModel(
        torch,
        base_channels=config["base_channels"],
        max_corner_offset_px=config["max_corner_offset_px"],
        spatial_pool_size=tuple(config["spatial_pool_size"]),
        spatial_hidden_channels=config["spatial_hidden_channels"],
    )


def _save_release(tmp_path, config):
    torch.manual_seed(0)
    model = _build(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    path = tmp_path / f"{config['model_name']}.pt"
    torch.save(
        {
            "format_version": 1,
            "model_version": "0.0.0",
            "model_config": config,
            "model_state_dict": model.state_dict(),
            "training_provenance": {"run_name": "test"},
        },
        path,
    )
    return path, model.eval()


def _pair():
    rng = np.random.default_rng(0)
    gray = np.zeros((800, 1140), dtype=np.uint8)
    gray[60:740, 90:1050] = rng.integers(1, 256, (680, 960), dtype=np.uint8)
    reference = np.zeros((800, 1140), dtype=np.uint8)
    reference[40:760, 70:1080] = rng.integers(1, 256, (720, 1010), dtype=np.uint8)
    return gray, reference


def test_released_pyramid_architecture_matches_training_layout():
    model = _PyramidOffsetModel(
        torch,
        encoder_base_channels=64,
        pyramid_channels=128,
        pyramid_head_channels=64,
        spatial_pool_size=(5, 8),
        spatial_hidden_channels=256,
        max_corner_offset_px=512.0,
    )

    # sharpy-model PairPyramidHomographyNet(matching=False) defaults. The digest
    # covers ordered state-dict "key:shape" lines from the training model.
    layout = "\n".join(
        f"{key}:{tuple(value.shape)}" for key, value in model.state_dict().items()
    )
    assert sum(p.numel() for p in model.parameters()) == 13_976_840
    assert len(model.state_dict()) == 102
    assert hashlib.sha256(layout.encode()).hexdigest() == (
        "949952f3de143fc1408f20409eb384ccd366c54c4457ec6cb214e3002100009d"
    )
    assert model.requires_grayscale_inputs


@pytest.mark.parametrize("config", [TINY_PYRAMID_CONFIG, TINY_SPATIAL_CONFIG])
def test_predictor_loads_state_dict_checkpoint_and_reproduces_model(tmp_path, config):
    path, model = _save_release(tmp_path, config)
    gray, reference = _pair()
    predictor = RegistrationTransformPredictor(path)
    predictor.load()

    offsets = predictor.predict_offsets(gray, reference)

    def to_tensor(image):
        return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]

    with torch.no_grad():
        expected = model(to_tensor(gray), to_tensor(reference)).numpy().reshape(4, 2)
    assert offsets.shape == (4, 2)
    assert np.isfinite(offsets).all()
    np.testing.assert_array_equal(offsets, expected)

    # Native colour samples are converted to the grayscale input both models use.
    color = np.repeat(gray[:, :, None], 3, axis=2)
    np.testing.assert_array_equal(predictor.predict_offsets(color, reference), offsets)


def test_pyramid_checkpoint_allows_unbounded_offsets(tmp_path):
    config = dict(TINY_PYRAMID_CONFIG, max_corner_offset_px=None)
    path, _ = _save_release(tmp_path, config)
    predictor = RegistrationTransformPredictor(path)
    predictor.load()

    assert predictor.model.max_corner_offset_px is None
    assert np.isfinite(predictor.predict_offsets(*_pair())).all()
