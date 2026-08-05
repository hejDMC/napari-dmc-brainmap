from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALIDATED_IMAGE_SHAPE = (800, 1140)


CORNER_ORDER = np.float32(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
)


def offsets_to_homography(offsets: Any, image_shape: tuple[int, ...]) -> np.ndarray:
    """Convert four sample-to-atlas corner offsets to a 3x3 homography."""
    h, w = image_shape[:2]
    src = CORNER_ORDER.copy()
    src[:, 0] *= w - 1
    src[:, 1] *= h - 1

    offsets = np.asarray(offsets, dtype=np.float32).reshape(4, 2)
    dst = src + offsets
    return cv2.getPerspectiveTransform(src, dst)


def project_points(transform: np.ndarray, points: Any) -> np.ndarray:
    """Apply a 3x3 homography to a list of xy points."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(transform, dtype=np.float32)).reshape(-1, 2)


def homography_to_registration_points(
    transform: np.ndarray,
    image_shape: tuple[int, ...],
    atlas_resolution: tuple[int, int] | list[int] | None = None,
    scale_mapping: dict[int, int] | None = None,
    count: int = 5,
) -> tuple[list[list[int]], list[list[int]]]:
    """Convert an image-space homography into spread sample/atlas point pairs."""
    if count != 5:
        raise ValueError("Prediction materialization currently requires exactly 5 points.")

    height, width = image_shape[:2]
    if height <= 1 or width <= 1:
        raise ValueError("Cannot generate prediction dots for an empty image.")

    inset_x = max((width - 1) * 0.08, 1.0)
    inset_y = max((height - 1) * 0.08, 1.0)
    grid_x = np.linspace(inset_x, (width - 1) - inset_x, 17)
    grid_y = np.linspace(inset_y, (height - 1) - inset_y, 17)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    sample_candidates = np.column_stack([mesh_x.ravel(), mesh_y.ravel()]).astype(np.float32)
    rounded_sample = np.rint(sample_candidates).astype(int)
    sample_candidates = rounded_sample.astype(np.float32)
    atlas_candidates = project_points(transform, sample_candidates)

    rounded_atlas = np.rint(atlas_candidates).astype(int)
    valid = (
        np.isfinite(atlas_candidates).all(axis=1)
        & (rounded_sample[:, 0] >= 0)
        & (rounded_sample[:, 0] < width)
        & (rounded_sample[:, 1] >= 0)
        & (rounded_sample[:, 1] < height)
        & (rounded_atlas[:, 0] >= 0)
        & (rounded_atlas[:, 0] < width)
        & (rounded_atlas[:, 1] >= 0)
        & (rounded_atlas[:, 1] < height)
    )

    if not np.any(valid):
        raise ValueError(
            "Prediction did not produce any in-bound points for this image."
        )

    sample_candidates = sample_candidates[valid]
    rounded_sample = rounded_sample[valid]
    rounded_atlas = rounded_atlas[valid]
    anchors = np.array(
        [
            [inset_x, inset_y],
            [(width - 1) - inset_x, inset_y],
            [(width - 1) - inset_x, (height - 1) - inset_y],
            [inset_x, (height - 1) - inset_y],
            [(width - 1) / 2.0, (height - 1) / 2.0],
        ],
        dtype=np.float32,
    )

    selected_indices = []
    available_indices = set(range(len(sample_candidates)))
    for anchor in anchors:
        if not available_indices:
            break
        ordered = np.argsort(np.linalg.norm(sample_candidates - anchor, axis=1))
        for index in ordered:
            index = int(index)
            if index in available_indices:
                selected_indices.append(index)
                available_indices.remove(index)
                break

    if len(selected_indices) < count:
        raise ValueError(
            "Prediction could not generate 5 in-bound registration dots. "
            "Try adjusting the atlas slice or prediction model."
        )

    sample_points = rounded_sample[selected_indices].tolist()
    atlas_points = rounded_atlas[selected_indices].tolist()

    if scale_mapping is not None:
        sample_points = _scale_points(sample_points, scale_mapping)
        atlas_points = _scale_points(atlas_points, scale_mapping)

    if atlas_resolution is not None:
        _validate_bounds(sample_points, atlas_resolution, "sample")
        _validate_bounds(atlas_points, atlas_resolution, "atlas")

    return sample_points, atlas_points


def _scale_points(points: list[list[int]], scale_mapping: dict[int, int]) -> list[list[int]]:
    try:
        return [[int(scale_mapping[x]), int(scale_mapping[y])] for x, y in points]
    except KeyError as exc:
        raise ValueError("Prediction dot coordinate is outside the display scale map.") from exc


def _validate_bounds(
    points: list[list[int]],
    atlas_resolution: tuple[int, int] | list[int],
    name: str,
) -> None:
    width, height = atlas_resolution
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Predicted {name} dot [{x}, {y}] is outside image bounds "
                f"[0 <= x < {width}, 0 <= y < {height}]."
            )


class RegistrationTransformPredictor:
    """Lazy PyTorch adapter for preview-only registration predictions."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.torch = None
        self.device = None

    def load(self) -> None:
        """Load and prepare the prediction model for CPU inference."""
        self._ensure_model_loaded()

    def predict_transform(
        self,
        sample_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> np.ndarray:
        offsets = self.predict_offsets(sample_image, reference_image)
        return offsets_to_homography(offsets, sample_image.shape)

    def predict_offsets(
        self,
        sample_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> np.ndarray:
        self._validate_inputs(sample_image, reference_image)
        self._ensure_model_loaded()
        expected_channels = self._model_input_channels()
        force_gray = expected_channels == 2
        sample_tensor = self._image_to_tensor(sample_image, force_gray=force_gray)
        reference_tensor = self._image_to_tensor(reference_image, force_gray=force_gray)

        with self.torch.no_grad():
            output = self.model(sample_tensor, reference_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return np.asarray(output, dtype=np.float32).reshape(4, 2)

    @staticmethod
    def _validate_inputs(
        sample_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        sample = np.asarray(sample_image)
        reference = np.asarray(reference_image)
        for name, image in [("sample", sample), ("reference", reference)]:
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"Prediction {name} image must be grayscale or color."
                )
            if image.shape[:2] != VALIDATED_IMAGE_SHAPE:
                raise ValueError(
                    f"Prediction {name} image must be 1140 x 800 pixels "
                    f"(width x height); got {image.shape[1]} x {image.shape[0]}."
                )
            if image.dtype != np.uint8:
                raise ValueError(
                    f"Prediction {name} image must use uint8 pixels; "
                    f"got {image.dtype}."
                )
        if reference.ndim != 2:
            raise ValueError("Prediction reference image must be grayscale.")

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to use the registration Predict button. "
                "Install the CPU build with: python -m pip install "
                "\"torch>=2.12.0\" --index-url "
                "https://download.pytorch.org/whl/cpu"
            ) from exc

        self.torch = torch
        self.device = torch.device("cpu")
        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        except Exception:
            checkpoint = torch.load(
                str(self.model_path),
                map_location=self.device,
                weights_only=True,
            )
            if hasattr(checkpoint, "eval"):
                self.model = checkpoint
            elif isinstance(checkpoint, dict) and hasattr(checkpoint.get("model"), "eval"):
                self.model = checkpoint["model"]
            elif (
                isinstance(checkpoint, dict)
                and checkpoint.get("model_config", {}).get("model_name") == "spatial"
                and "model_state_dict" in checkpoint
            ):
                self.model = self._load_spatial_model(checkpoint)
            else:
                raise ValueError(
                    "Model file must be a TorchScript model or a checkpoint "
                    "containing a callable model object."
                )
        self.model.to(self.device)
        self.model.eval()

    def _load_spatial_model(self, checkpoint: dict):
        config = checkpoint["model_config"]
        model = _SpatialOffsetModel(
            self.torch,
            base_channels=int(config.get("base_channels", 24)),
            max_corner_offset_px=float(config.get("max_corner_offset_px", 512.0)),
            spatial_pool_size=tuple(config.get("spatial_pool_size", [5, 8])),
            spatial_hidden_channels=int(
                config.get("spatial_hidden_channels", 256)
            ),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _model_input_channels(self) -> int | None:
        try:
            return int(self.model.features[0][0].weight.shape[1])
        except (AttributeError, IndexError, TypeError):
            return None

    def _image_to_tensor(self, image: np.ndarray, force_gray: bool = False):
        image = np.asarray(image)
        if force_gray and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 2:
            image = image[:, :, None]
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return self.torch.from_numpy(image).to(self.device)


class _SpatialOffsetModel:
    """Architecture for spatial-head state-dict registration checkpoints."""

    def __new__(
        cls,
        torch_module,
        base_channels: int,
        max_corner_offset_px: float,
        spatial_pool_size: tuple[int, int],
        spatial_hidden_channels: int,
    ):
        nn = torch_module.nn

        class SpatialOffsetModel(nn.Module):
            def __init__(self):
                super().__init__()
                channels = [
                    base_channels,
                    base_channels * 2,
                    base_channels * 4,
                    base_channels * 4,
                    base_channels * 8,
                ]
                in_channels = 2
                blocks = []
                for out_channels in channels:
                    blocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                    in_channels = out_channels

                self.features = nn.Sequential(*blocks)
                self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_pool_size)
                pooled_features = channels[-1] * spatial_pool_size[0] * spatial_pool_size[1]
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(pooled_features, spatial_hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_channels, spatial_hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_channels, 8),
                )
                self.max_corner_offset_px = max_corner_offset_px

            def forward(self, sample, reference):
                pair = torch_module.cat([sample, reference], dim=1)
                features = self.spatial_pool(self.features(pair))
                offsets = self.head(features).reshape(-1, 4, 2)
                if self.max_corner_offset_px is not None:
                    offsets = torch_module.tanh(offsets) * self.max_corner_offset_px
                return offsets

        return SpatialOffsetModel()
