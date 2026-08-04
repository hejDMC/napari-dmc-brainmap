import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.append("./src")

from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.prediction import (
    RegistrationTransformPredictor,
    homography_to_registration_points,
    offsets_to_homography,
)
from napari_dmc_brainmap.registration import registration as registration_module


def _map_points(transform, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, transform).reshape(-1, 2)


def test_offsets_to_homography_identity():
    transform = offsets_to_homography(np.zeros((4, 2)), (10, 20))
    expected = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(transform, expected, atol=1e-6)


def test_offsets_to_homography_maps_corners():
    offsets = np.array(
        [
            [1, 2],
            [-1, 3],
            [2, -2],
            [0, -1],
        ],
        dtype=np.float32,
    )
    transform = offsets_to_homography(offsets, (10, 20))

    source_corners = np.array(
        [
            [0, 0],
            [19, 0],
            [19, 9],
            [0, 9],
        ],
        dtype=np.float32,
    )
    mapped_corners = _map_points(transform, source_corners)
    np.testing.assert_allclose(mapped_corners, source_corners + offsets, atol=1e-4)


def test_predictor_adapter_uses_offsets_for_transform():
    class FakePredictor(RegistrationTransformPredictor):
        def __init__(self):
            pass

        def predict_offsets(self, sample_image, reference_image):
            return np.array(
                [
                    [2, 0],
                    [2, 0],
                    [2, 0],
                    [2, 0],
                ],
                dtype=np.float32,
            )

    sample = np.zeros((8, 12), dtype=np.uint8)
    reference = np.zeros((8, 12), dtype=np.uint8)
    transform = FakePredictor().predict_transform(sample, reference)

    mapped = _map_points(transform, [[0, 0], [11, 7]])
    np.testing.assert_allclose(mapped, [[2, 0], [13, 7]], atol=1e-4)


def test_predictor_loads_model_on_cpu(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self):
            self.device = None
            self.is_evaluating = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_evaluating = True
            return self

    model = FakeModel()
    load_calls = []

    def load_model(path, map_location):
        load_calls.append((path, map_location))
        return model

    fake_torch = SimpleNamespace(
        device=lambda name: name,
        jit=SimpleNamespace(load=load_model),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model_path = tmp_path / "model.pt"
    predictor = RegistrationTransformPredictor(model_path)
    predictor.load()

    assert predictor.device == "cpu"
    assert load_calls == [(str(model_path), "cpu")]
    assert model.device == "cpu"
    assert model.is_evaluating


def test_registration_start_preloads_selected_model(monkeypatch, tmp_path):
    events = []

    class FakePredictor:
        def __init__(self, model_path):
            self.model_path = model_path
            events.append("predictor-created")

        def load(self):
            events.append("model-loaded")

    class FakeRegistrationViewer:
        def __init__(self, widget, regi_dict, predictor=None):
            self.regi_dict = regi_dict
            self.predictor = predictor
            events.append("viewer-created")

        def show(self):
            events.append("viewer-shown")

    model_path = tmp_path / "model.pt"
    widget = SimpleNamespace(
        header=SimpleNamespace(
            input_path=SimpleNamespace(value=tmp_path),
            regi_chan=SimpleNamespace(value="green"),
        ),
        model_path=model_path,
    )
    monkeypatch.setattr(registration_module, "check_input_path", lambda path: True)
    monkeypatch.setattr(
        registration_module,
        "get_info",
        lambda *args, **kwargs: tmp_path / "registration",
    )
    monkeypatch.setattr(
        registration_module,
        "create_regi_dict",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        registration_module,
        "RegistrationTransformPredictor",
        FakePredictor,
    )
    monkeypatch.setattr(
        registration_module,
        "RegistrationViewer",
        FakeRegistrationViewer,
    )

    registration_module.RegistrationWidget._start_sharpy_track(widget)

    assert events == [
        "predictor-created",
        "model-loaded",
        "viewer-created",
        "viewer-shown",
    ]
    assert widget.reg_viewer.predictor.model_path == model_path
    assert widget.reg_viewer.regi_dict["model_path"] == model_path


def test_registration_start_stops_when_model_loading_fails(
    monkeypatch,
    tmp_path,
):
    class BrokenPredictor:
        def __init__(self, model_path):
            pass

        def load(self):
            raise ValueError("invalid checkpoint")

    errors = []
    widget = SimpleNamespace(
        header=SimpleNamespace(
            input_path=SimpleNamespace(value=tmp_path),
            regi_chan=SimpleNamespace(value="green"),
        ),
        model_path=tmp_path / "broken.pt",
    )
    monkeypatch.setattr(registration_module, "check_input_path", lambda path: True)
    monkeypatch.setattr(
        registration_module,
        "RegistrationTransformPredictor",
        BrokenPredictor,
    )
    monkeypatch.setattr(
        registration_module.QMessageBox,
        "critical",
        lambda *args: errors.append(args),
    )

    def fail_if_viewer_created(*args, **kwargs):
        raise AssertionError("viewer should not be created")

    monkeypatch.setattr(
        registration_module,
        "RegistrationViewer",
        fail_if_viewer_created,
    )

    registration_module.RegistrationWidget._start_sharpy_track(widget)

    assert not hasattr(widget, "reg_viewer")
    assert len(errors) == 1
    assert errors[0][1] == "Prediction model failed to load"
    assert "invalid checkpoint" in errors[0][2]


def test_homography_to_registration_points_identity():
    sample_points, atlas_points = homography_to_registration_points(
        np.eye(3, dtype=np.float32),
        (20, 30),
        [30, 20],
    )

    assert len(sample_points) == 5
    assert len(atlas_points) == 5
    assert sample_points == atlas_points
    assert len({tuple(point) for point in sample_points}) == 5
    for x, y in sample_points:
        assert 0 <= x < 30
        assert 0 <= y < 20


def test_homography_to_registration_points_translation():
    transform = np.array(
        [
            [1, 0, 3],
            [0, 1, 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    sample_points, atlas_points = homography_to_registration_points(
        transform,
        (20, 30),
        [30, 20],
    )

    assert len(sample_points) == 5
    for sample_point, atlas_point in zip(sample_points, atlas_points):
        assert atlas_point == [sample_point[0] + 3, sample_point[1] + 2]
        assert 0 <= atlas_point[0] < 30
        assert 0 <= atlas_point[1] < 20


def test_homography_to_registration_points_rejects_out_of_bound_transform():
    transform = np.array(
        [
            [1, 0, 100],
            [0, 1, 100],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    with np.testing.assert_raises(ValueError):
        homography_to_registration_points(transform, (20, 30), [30, 20])
