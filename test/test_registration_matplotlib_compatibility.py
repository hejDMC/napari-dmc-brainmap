from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from napari_dmc_brainmap.registration.registration import RegistrationWidget
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.HelperModel import (
    HelperModel,
    _canvas_to_rgb_array,
)
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view import (
    RegistrationViewer as registration_viewer_module,
)
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import (
    RegistrationViewer,
)


class _Action:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def isEnabled(self):
        return self.enabled

    def setEnabled(self, enabled):
        self.enabled = enabled


def test_canvas_to_rgb_array_supports_current_matplotlib_api():
    figure = Figure(figsize=(2, 1), dpi=20)
    canvas = FigureCanvasAgg(figure)
    figure.patch.set_facecolor((0.25, 0.5, 0.75, 1.0))

    image = _canvas_to_rgb_array(canvas)

    assert image.shape == (20, 40, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert image.flags.owndata
    np.testing.assert_array_equal(image[0, 0], [64, 128, 191])


def test_helper_model_renders_initial_and_updated_illustrations(monkeypatch):
    helper_model_module = __import__(
        _canvas_to_rgb_array.__module__,
        fromlist=["HelperModel"],
    )

    class _QImage:
        Format_RGB888 = object()

        def __init__(self, *args):
            self.args = args

    class _QPixmap:
        @staticmethod
        def fromImage(image):
            return image

    class _PreviewLabel:
        def __init__(self):
            self.pixmap = None

        def setPixmap(self, pixmap):
            self.pixmap = pixmap

    monkeypatch.setattr(helper_model_module, "QImage", _QImage)
    monkeypatch.setattr(helper_model_module, "QPixmap", _QPixmap)

    preview_label = _PreviewLabel()
    viewer = SimpleNamespace(
        atlasModel=SimpleNamespace(
            template=np.zeros((80, 60, 4), dtype=np.uint16),
            xyz_dict={
                "x": (0, 4, 25),
                "y": (0, 60, 25),
                "z": (0, 80, 25),
            },
            bregma=(40, 0, 0),
            z_idx=0,
        ),
        status=SimpleNamespace(sliceNum=3, decimal=2),
        interpolatePositionPage=SimpleNamespace(preview_label=preview_label),
    )

    model = HelperModel(viewer)
    model.update_illustration()

    assert model.img0.ndim == 3
    assert model.img0.shape[2] == 3
    assert model.img0.dtype == np.uint8
    assert model.img0.flags.c_contiguous
    assert model.img.shape == model.img0.shape
    assert model.img.dtype == np.uint8
    assert model.img.flags.c_contiguous
    assert preview_label.pixmap is not None


def test_interpolation_action_changes_only_after_page_opens(monkeypatch):
    class _Page:
        def __init__(self, viewer):
            self.viewer = viewer
            self.shown = False

        def show(self):
            self.shown = True

    monkeypatch.setattr(registration_viewer_module, "InterpolatePosition", _Page)
    viewer = SimpleNamespace(interpolatePositionAct=_Action())

    RegistrationViewer.interpolatePositionPageOpen(viewer)

    assert viewer.interpolatePositionPage.shown
    assert not viewer.interpolatePositionAct.isEnabled()


def test_failed_interpolation_page_construction_leaves_action_enabled(monkeypatch):
    def _raise_during_construction(viewer):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(
        registration_viewer_module,
        "InterpolatePosition",
        _raise_during_construction,
    )
    viewer = SimpleNamespace(interpolatePositionAct=_Action())

    with pytest.raises(RuntimeError, match="construction failed"):
        RegistrationViewer.interpolatePositionPageOpen(viewer)

    assert viewer.interpolatePositionAct.isEnabled()
    assert not hasattr(viewer, "interpolatePositionPage")


def test_registration_cleanup_tolerates_missing_interpolation_page():
    class _Signal:
        def disconnect(self):
            pass

    scene = SimpleNamespace(changed=_Signal())
    reg_viewer = SimpleNamespace(
        widget=SimpleNamespace(
            viewerLeft=SimpleNamespace(scene=scene),
            viewerRight=SimpleNamespace(scene=scene),
        ),
        interpolatePositionAct=_Action(enabled=False),
        measurementAct=_Action(),
        shortcutsAct=_Action(),
        regViewerWidget=object(),
        app=object(),
        regi_dict={},
        status=object(),
        atlasModel=object(),
    )
    registration_widget = SimpleNamespace(reg_viewer=reg_viewer)

    RegistrationWidget.del_regviewer_instance(registration_widget)

    assert not hasattr(registration_widget, "reg_viewer")
