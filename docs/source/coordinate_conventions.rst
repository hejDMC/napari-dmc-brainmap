Coordinate and hemisphere conventions
=====================================

DMC-BrainMap reports public mediolateral coordinates with a left-positive
sign convention:

* ``ml_mm > 0`` is the anatomical left hemisphere.
* ``ml_mm < 0`` is the anatomical right hemisphere.
* ``ml_mm == 0`` is the midline.

For BrainGlobe atlases whose right-to-left axis is named ``rl``, atlas indices
increase from anatomical right to anatomical left. Therefore:

* ``ml_coords > bregma_rl`` is the anatomical left hemisphere.
* ``ml_coords < bregma_rl`` is the anatomical right hemisphere.

An unflipped coronal atlas image consequently displays anatomical left on the
right side of the image and anatomical right on the left side. Image pixels
must not be assigned a hemisphere from screen left or screen right after an
image has been mirrored, rotated, or exported; use the registered atlas
coordinate instead.

Section orientations
--------------------

The hemisphere rule is independent of section orientation. For the standard
BrainGlobe ASR axis order ``(ap, si, rl)``, DMC-BrainMap maps atlas sections as
follows:

* coronal: ``(x, y, slice) = (rl, si, ap)``;
* horizontal: ``(x, y, slice) = (rl, ap, si)``;
* sagittal: ``(x, y, slice) = (si, ap, rl)``.

The RL coordinate is therefore an in-plane horizontal coordinate in standard
coronal and horizontal plots. In a sagittal view, RL selects the slice: a
positive ML value selects a slice in the anatomical left hemisphere rather
than a left-hand region within the displayed plane. A viewer may transpose or
rotate a plane, so screen direction must not be used as the hemisphere label.

Voxel coordinates and 3D rendering
----------------------------------

Atlas voxel coordinates are absolute, origin-based indices. The optional
brainrender export writes these coordinates in atlas axis order and scales
them to micrometres. It does not write bregma-relative ``ap_mm``, ``dv_mm``,
and ``ml_mm`` values. A point whose ``ml_coords`` is greater than
``bregma_rl`` therefore remains in the anatomical left hemisphere in the 3D
atlas, but a signed ``ml_mm`` value must not be passed directly as an atlas
world coordinate.

The apparent left and right sides of a rotatable 3D view also depend on camera
position and display handedness. This can make a correctly located point look
mirrored without changing its underlying anatomical hemisphere.

Result-file compatibility
-------------------------

Current result CSV files and user-facing coordinate displays use left-positive
``ml_mm`` values. Older result CSV files may contain axis-native values in
which anatomical left is negative. The visualization data loader derives
``ml_mm`` from ``ml_coords``, atlas bregma, and atlas resolution, so both old
and current result files are interpreted using the public left-positive
convention. ``coord_mm_transform`` remains a low-level index-oriented utility
for registration calculations and should not be used by itself to assign a
hemisphere.
