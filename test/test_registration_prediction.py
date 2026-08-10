import builtins
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
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.AtlasModel import (
    AtlasModel,
)
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.MainWidget import (
    MainWidget,
)
from napari_dmc_brainmap.registration import registration as registration_module


def _map_points(transform, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, transform).reshape(-1, 2)


def test_atlas_model_get_stack_loads_grayscale_images(tmp_path):
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image_path = tmp_path / "grayscale.tif"
    assert cv2.imwrite(str(image_path), image)

    status = SimpleNamespace(
        folderPath=tmp_path,
        imgFileName=[image_path.name],
        sliceNum=1,
        imageRGB=False,
    )
    model = AtlasModel.__new__(AtlasModel)
    model.regViewer = SimpleNamespace(status=status)
    model.regi_dict = {
        "xyz_dict": {"y": ("si", 3), "x": ("rl", 4)}
    }

    model.getStack()

    assert model.imgStack.dtype == np.uint8
    assert model.imgStack.shape == (1, 3, 4)
    np.testing.assert_array_equal(model.imgStack[0], image)
    assert status.imageRGB is False


def test_atlas_model_get_stack_loads_color_images(tmp_path):
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    image_path = tmp_path / "color.tif"
    assert cv2.imwrite(str(image_path), image)

    status = SimpleNamespace(
        folderPath=tmp_path,
        imgFileName=[image_path.name],
        sliceNum=1,
        imageRGB=False,
    )
    model = AtlasModel.__new__(AtlasModel)
    model.regViewer = SimpleNamespace(status=status)
    model.regi_dict = {
        "xyz_dict": {"y": ("si", 3), "x": ("rl", 4)}
    }

    model.getStack()

    assert model.imgStack.dtype == np.uint8
    assert model.imgStack.shape == (1, 3, 4, 3)
    np.testing.assert_array_equal(model.imgStack[0], image)
    assert status.imageRGB is True


def test_atlas_model_fits_forward_and_reverse_transforms(monkeypatch):
    atlas_points = np.array(
        [[2, 3], [14, 2], [13, 15], [1, 12]],
        dtype=np.float32,
    )
    sample_points = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        dtype=np.float32,
    )
    forward_transform = np.array(
        [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
        dtype=np.float64,
    )
    reverse_transform = np.array(
        [[1, 0, -2], [0, 1, -3], [0, 0, 1]],
        dtype=np.float64,
    )
    fit_calls = []

    def fake_fit(source, destination):
        fit_calls.append(
            (
                np.asarray(source).copy(),
                np.asarray(destination).copy(),
            )
        )
        if len(fit_calls) == 1:
            return forward_transform
        return reverse_transform

    monkeypatch.setattr(
        "napari_dmc_brainmap.registration.sharpy_track.sharpy_track."
        "model.AtlasModel.fitGeoTrans",
        fake_fit,
    )
    monkeypatch.setattr(
        "napari_dmc_brainmap.registration.sharpy_track.sharpy_track."
        "model.AtlasModel.QPixmap",
        SimpleNamespace(fromImage=lambda image: image),
    )
    rendered = []
    displayed = []
    model = AtlasModel.__new__(AtlasModel)
    model.clearPredictedPreview = lambda: None
    model._render_transform = lambda transform: (
        rendered.append(transform),
        None,
        "forward-warp",
        "forward-blend",
    )
    model.regViewer = SimpleNamespace(
        status=SimpleNamespace(currentSliceNumber=0, blendMode={0: 1}),
        widget=SimpleNamespace(
            viewerLeft=SimpleNamespace(
                labelImg=SimpleNamespace(setPixmap=displayed.append)
            )
        ),
    )

    model.updateTransform(atlas_points, sample_points)

    assert len(fit_calls) == 2
    np.testing.assert_array_equal(fit_calls[0][0], sample_points)
    np.testing.assert_array_equal(fit_calls[0][1], atlas_points)
    np.testing.assert_array_equal(fit_calls[1][0], atlas_points)
    np.testing.assert_array_equal(fit_calls[1][1], sample_points)
    np.testing.assert_array_equal(rendered[0], forward_transform)
    np.testing.assert_array_equal(model.rtransform, reverse_transform)
    assert displayed == ["forward-blend"]


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


def test_predictor_loads_checkpoint_weights_only(monkeypatch, tmp_path):
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

    def fail_to_load_torchscript(path, map_location):
        raise ValueError("not a TorchScript model")

    def load_checkpoint(path, map_location, weights_only):
        load_calls.append((path, map_location, weights_only))
        return model

    fake_torch = SimpleNamespace(
        device=lambda name: name,
        jit=SimpleNamespace(load=fail_to_load_torchscript),
        load=load_checkpoint,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model_path = tmp_path / "model.pt"
    predictor = RegistrationTransformPredictor(model_path)
    predictor.load()

    assert load_calls == [(str(model_path), "cpu", True)]
    assert model.device == "cpu"
    assert model.is_evaluating


def test_predictor_missing_torch_shows_cpu_install_command(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    predictor = RegistrationTransformPredictor(tmp_path / "model.pt")

    with np.testing.assert_raises_regex(
        ImportError,
        "download.pytorch.org/whl/cpu",
    ):
        predictor.load()


def test_predictor_validates_native_input_contract():
    sample = np.zeros((800, 1140), dtype=np.uint8)
    reference = np.zeros((800, 1140), dtype=np.uint8)

    RegistrationTransformPredictor._validate_inputs(sample, reference)


def test_predictor_rejects_monitor_resized_input():
    sample = np.zeros((536, 763), dtype=np.uint8)
    reference = np.zeros((536, 763), dtype=np.uint8)

    with np.testing.assert_raises_regex(ValueError, "1140 x 800"):
        RegistrationTransformPredictor._validate_inputs(sample, reference)


def test_predictor_rejects_non_uint8_or_color_reference():
    sample = np.zeros((800, 1140), dtype=np.uint8)
    float_reference = np.zeros((800, 1140), dtype=np.float32)
    color_reference = np.zeros((800, 1140, 3), dtype=np.uint8)

    with np.testing.assert_raises_regex(ValueError, "uint8"):
        RegistrationTransformPredictor._validate_inputs(
            sample,
            float_reference,
        )
    with np.testing.assert_raises_regex(ValueError, "grayscale"):
        RegistrationTransformPredictor._validate_inputs(
            sample,
            color_reference,
        )


def test_atlas_model_predicts_native_and_renders_display_transform():
    native_transform = np.array(
        [
            [1.0, 0.0, 100.0],
            [0.0, 1.0, -50.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    class FakePredictor:
        def __init__(self):
            self.calls = []

        def predict_transform(self, sample, reference):
            self.calls.append((sample, reference))
            return native_transform.copy()

    predictor = FakePredictor()
    status = SimpleNamespace(
        currentSliceNumber=3,
        blendMode={},
    )
    model = AtlasModel.__new__(AtlasModel)
    model.predictor = predictor
    model.sample_native = np.zeros((800, 1140), dtype=np.uint8)
    model.reference_native = np.zeros((800, 1140), dtype=np.uint8)
    model.sample = np.zeros((536, 763), dtype=np.uint8)
    model.slice = np.zeros((536, 763), dtype=np.uint8)
    model.predictedSliceNumber = None
    model.predictedTransformNative = None
    model.predictedTransformDisplay = None
    model.regViewer = SimpleNamespace(status=status)
    rendered = []
    model._render_transform = lambda transform: (
        rendered.append(transform.copy()),
        None,
        None,
        None,
    )
    model.showPredictedPreview = lambda: True

    model.applyPredictedTransform()

    assert len(predictor.calls) == 1
    assert predictor.calls[0][0] is model.sample_native
    assert predictor.calls[0][1] is model.reference_native
    np.testing.assert_allclose(
        model.predictedTransformNative,
        native_transform,
    )
    np.testing.assert_allclose(
        model.predictedTransformDisplay,
        np.array(
            [
                [1.0, 0.0, 100.0 * 762 / 1139],
                [0.0, 1.0, -50.0 * 535 / 799],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        rendered[0],
        model.predictedTransformDisplay,
    )
    assert model.predictedSliceNumber == 3


def test_atlas_model_rejects_prediction_for_out_of_volume_slice():
    class FakePredictor:
        def predict_transform(self, sample, reference):
            raise AssertionError("predictor must not receive an invalid atlas slice")

    model = AtlasModel.__new__(AtlasModel)
    model.predictor = FakePredictor()
    model.slice_in_volume = False
    model.sample_native = np.zeros((800, 1140), dtype=np.uint8)
    model.reference_native = np.zeros((800, 1140), dtype=np.uint8)

    with np.testing.assert_raises_regex(ValueError, "outside the atlas volume"):
        model.applyPredictedTransform()


def test_predict_button_is_disabled_for_out_of_volume_slice():
    class FakeButton:
        def setEnabled(self, enabled):
            self.enabled = enabled

        def setToolTip(self, tooltip):
            self.tooltip = tooltip

    widget = SimpleNamespace(
        predict_btn=FakeButton(),
        toggle=SimpleNamespace(isEnabled=lambda: True),
        viewerLeft=SimpleNamespace(itemGroup=[]),
        viewerRight=SimpleNamespace(itemGroup=[]),
    )
    widget.regViewer = SimpleNamespace(
        atlasModel=SimpleNamespace(slice_in_volume=False),
        status=SimpleNamespace(
            currentSliceNumber=0,
            atlasDots={},
            sampleDots={},
            tMode=0,
        ),
        widget=widget,
    )

    MainWidget.updatePredictButton(widget)

    assert widget.predict_btn.enabled is False
    assert "outside the volume" in widget.predict_btn.tooltip


def test_atlas_model_preserves_clean_native_reference_for_prediction():
    status = SimpleNamespace(
        x_angle=0.0,
        y_angle=0.0,
        current_z=0.0,
        imageRGB=False,
    )
    model = AtlasModel.__new__(AtlasModel)
    model.regViewer = SimpleNamespace(
        status=status,
        singleWindowSize=[763, 536],
    )
    model.xyz_dict = {
        "x": ["rl", 1140, 10],
        "y": ["si", 800, 10],
        "z": ["ap", 1320, 10],
    }
    model.fontscale = 1.0
    model.fontthickness = 1
    model.simpleSlice = lambda: setattr(
        model,
        "slice",
        np.zeros((800, 1140), dtype=np.uint8),
    )

    model.getSlice()

    assert model.reference_native.shape == (800, 1140)
    assert model.slice.shape == (536, 763)
    assert not np.shares_memory(model.reference_native, model.slice)
    assert np.count_nonzero(model.reference_native) == 0
    assert np.count_nonzero(model.slice) > 0
    assert model.slice_in_volume


def test_atlas_model_preserves_native_sample_for_prediction():
    status = SimpleNamespace(
        sliceNum=1,
        currentSliceNumber=0,
        imageRGB=False,
    )
    model = AtlasModel.__new__(AtlasModel)
    model.regViewer = SimpleNamespace(
        status=status,
        singleWindowSize=[763, 536],
    )
    model.imgStack = np.zeros((1, 800, 1140), dtype=np.uint8)

    model.getSample()

    assert np.shares_memory(model.sample_native, model.imgStack)
    assert model.sample_native.shape == (800, 1140)
    assert model.sample.shape == (536, 763)


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


def test_materialized_native_dots_do_not_depend_on_display_size():
    def materialize(display_shape):
        saved = []
        status = SimpleNamespace(
            currentSliceNumber=0,
            x_angle=0.0,
            y_angle=0.0,
            current_z=0.0,
            atlasLocation={},
            atlasDots={},
            sampleDots={},
            blendMode={},
            saveRegistration=lambda: saved.append(True),
        )
        model = AtlasModel.__new__(AtlasModel)
        model.predictedTransformNative = np.eye(3, dtype=np.float32)
        model.sample_native = np.zeros((800, 1140), dtype=np.uint8)
        model.sample = np.zeros(display_shape, dtype=np.uint8)
        model.regViewer = SimpleNamespace(
            status=status,
            atlas_resolution=[1140, 800],
            widget=SimpleNamespace(updatePredictButton=lambda: None),
        )
        model.hasPredictedPreview = lambda: True
        model._display_dot_pairs = lambda atlas, sample: None
        model._native_points_to_display = lambda points: np.asarray(
            points,
            dtype=np.float32,
        )
        model.updateTransform = lambda atlas, sample: None

        assert model.materializePredictedDots()
        assert saved == [True]
        return status.sampleDots[0], status.atlasDots[0]

    full_size = materialize((800, 1140))
    reduced_size = materialize((536, 763))

    assert full_size == reduced_size
    assert full_size[0] == full_size[1]
    assert all(
        type(value) is int
        for point_set in full_size
        for point in point_set
        for value in point
    )
