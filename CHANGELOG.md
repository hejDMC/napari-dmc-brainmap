# Changelog

## Unreleased

- Python 3.14 is now supported, matching napari's default Python version.
  Automated tests also run on Ubuntu with Python 3.14.
- Registration prediction now uses `torch.jit.load` only for legacy
  TorchScript model files. Recommended state-dict checkpoints (predictor
  v2.0.0) no longer trigger PyTorch's warning that TorchScript is unsupported
  on Python 3.14.

## 0.2.1 - 2026-11-04

This is the first stable release since `0.1.8`. Upgrading from `0.1.x` also
brings every change from the `0.2.0b1` and `0.2.0b2` beta releases below,
including the BrainGlobe Atlas API 3.x migration, the platform-specific atlas
cache, and the standardized hemisphere and mediolateral coordinate
conventions. Review those entries before upgrading existing projects.

- Registration prediction now supports version 2.0.0 of the DMC-BrainMap
  Registration Predictor, a residual feature-pyramid model, and the
  prediction guide recommends it. Version 1.0.0 checkpoints still load
  unchanged.
- Prediction checkpoints with an unsupported model type or a newer file format
  now fail with a specific error message naming the supported types.

### Compatibility and testing

- Inference remains CPU-only FP32. When the optional `prediction` extra is
  installed, prediction tests save and load small synthetic spatial and pyramid
  checkpoints with real PyTorch and check the released pyramid weight layout;
  they are skipped otherwise. Release checks confirmed identical
  offsets to the training implementation on 24 held-out section pairs for both
  model versions.

### Installation

As a stable release, `0.2.1` is selected by a normal installation or upgrade:

```bash
pip install --upgrade napari-dmc-brainmap
```

## 0.2.0b2 - 2026-08-17

- Upgraded BrainGlobe Atlas API to `3.0.1`.
- Added automatic RGB contrast calibration with reproducible per-channel ranges,
  selectable signal-preservation profiles, all-black-section handling, and
  progress reporting in preprocessing. Fixed disabled/manual contrast behavior
  and replaced stale RGB parameter metadata.
- Added ProbeInterface 0.3.2 as the supported SpikeGLX metadata and
  Neuropixels geometry reader for the upcoming multi-shank workflow.
- Simplified the segmentation widget by showing only controls and
  presegmentation sections relevant to the selected workflow. RGB annotation
  now offers DAPI, green, and Cy3; N3 and Cy5 are available in single-channel
  mode.
- Standardized public mediolateral coordinates as positive in the anatomical
  left hemisphere and negative in the anatomical right hemisphere.
- Corrected the brain-section hemisphere tooltip and coordinate displays, and
  added compatibility normalization for historical left-negative result CSVs.
- Documented the relationship between atlas RL indices, all section
  orientations, 3D voxel exports, and signed ML coordinates.
- Removed Allen-10-um-specific hemisphere slicing from injection-site coverage
  and corrected atlas-plane extraction for horizontal and sagittal sections.

### Compatibility and testing

- The automatic RGB calibration path supports uint8 and uint16 stitched images;
  regression coverage includes calibration, preprocessing, atlas, and platform
  compatibility contracts.
- BrainGlobe Atlas API is pinned to `3.0.1`.

### Installation

Pre-releases are not selected by a normal pip upgrade. Install this beta
explicitly:

```bash
pip install napari-dmc-brainmap==0.2.0b2
```

## 0.2.0b1 - 2026-08-11

This is a beta release of the next DMC-BrainMap feature series. It contains
substantial atlas, scientific-stack, stitching, and packaging changes. The
automated test suite covers the supported platforms and core compatibility
contracts, but manual testing across real-world datasets and workflows is still
in progress.

### Highlights

- Migrated atlas integration from BrainGlobe Atlas API 1.x to the 3.x API.
- Moved derived atlas volumes into the platform-specific application cache and
  added offline API contract coverage.
- Replaced the AICS-based presegmentation path with an internal implementation
  while preserving supported grayscale and RGB behavior through regression
  fixtures.
- Reworked rigid-overlap stitching to infer ImageJ-compatible acquisition
  layouts from embedded TIFF metadata and stream tiles into a local scratch
  canvas.
- Added lossless compressed TIFF publishing with validated destinations,
  scratch-space checks, operation-specific partial files, and safe handling of
  local and network storage.
- Modernized the NumPy, pandas, SciPy, scikit-learn, scikit-image, TIFF,
  plotting, and napari GUI dependency stacks.
- Fixed registration illustration rendering for the Matplotlib 3.10 canvas API.
- Added optional prediction-assisted registration without making PyTorch a
  required plugin dependency.

### Compatibility

- Python 3.11, 3.12, and 3.13 are supported.
- Intel macOS is no longer supported; macOS support targets Apple Silicon.
- BrainGlobe Atlas API is pinned to `3.0.0rc1` for this beta.
- Existing users must download Atlas API v3 atlas data and regenerate derived
  application caches rather than migrate legacy `reference_8bit` caches.
- The SHARPy-track Cy3 default contrast range is now `50,500`, matching direct
  stitching.

### Testing status

- Automated tests cover Ubuntu with Python 3.11 and 3.13, Windows with Python
  3.11, and Apple Silicon macOS with Python 3.11.
- The local release-preparation suite contains 219 passing tests.
- Stitching was exercised with a real 54-tile acquisition and an SMB network
  destination.
- Atlas-dependent manual testing covered the primary registration,
  preprocessing, stitching, segmentation, and visualization workflows, but the
  beta should still be validated against additional datasets before the final
  `0.2.0` release.

### Installation

Pre-releases are not selected by a normal pip upgrade. Install this beta
explicitly:

```bash
pip install napari-dmc-brainmap==0.2.0b1
```

Please report regressions with the operating system, Python version, atlas,
input image format, and a minimal traceback or reproduction procedure.
