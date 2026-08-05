Prediction-assisted registration
================================

DMC-BrainMap can use an optional PyTorch checkpoint to propose an initial
projective alignment between a histological section and a selected Allen Mouse
Brain Atlas reference slice. The supported checkpoint and its model card are
published in the `DMC-BrainMap Registration Predictor repository
<https://huggingface.co/xiao-1011/dmc-brainmap-registration-predictor>`_.

Install PyTorch for prediction
------------------------------

PyTorch is optional and is not installed with the standard DMC-BrainMap
package. To use registration prediction, install the CPU-only build in the
same Python environment as napari:

.. code-block:: console

   python -m pip install "torch>=2.12.0" --index-url https://download.pytorch.org/whl/cpu

The ``prediction`` package extra declares the compatible PyTorch version for
package managers that are already configured to use the appropriate PyTorch
index. Installing from the CPU index explicitly avoids downloading unnecessary
GPU runtime packages.

Download the model
------------------

#. Open the model repository linked above.
#. Select **Files and versions**.
#. Download the `version 1.0.0 checkpoint
   <https://huggingface.co/xiao-1011/dmc-brainmap-registration-predictor/resolve/v1.0.0/dmc-brainmap-registration-predictor-v1.0.0.pt?download=true>`_.
   The filename is ``dmc-brainmap-registration-predictor-v1.0.0.pt``.

The model card documents the checkpoint version, supported inputs, evaluation
results, limitations, license, and SHA-256 checksum.

Select the model
----------------

#. Open the DMC-BrainMap registration widget in napari.
#. Configure the input folder, registration channel, atlas, and orientation as
   usual.
#. Click **Browse Model** and select the downloaded ``.pt`` checkpoint.
#. Confirm that the selected filename appears below the button.
#. Click **Start Registration GUI**. DMC-BrainMap validates and loads the model
   on the CPU before opening the registration window.

Use a prediction
----------------

#. Navigate to the atlas slice and angles that correspond to the current
   histological section.
#. Make sure the section does not already contain registration points.
#. Click the **Predict registration** button next to the transformation toggle.
#. Inspect the predicted blend or warp preview.
#. If the preview is useful, turn on transformation mode. The preview is
   converted into five editable registration-point pairs.
#. Move, add, or remove points as needed until the alignment is satisfactory.
#. Continue through the standard reviewed registration and saving workflow.

The prediction button is disabled when registration points already exist or
when the current viewer state cannot safely accept a preview. Changing the
atlas slice, angles, or sample clears an outstanding preview; run prediction
again after selecting the desired reference view.

Quality control
---------------

The model supplies an initialization only. It does not select the correct atlas
slice and does not replace visual review. Predictions can be inaccurate for
damaged or incomplete sections, unusual contrast, acquisition artifacts, or
data outside the conditions described in the model card. Always inspect and,
when necessary, correct the registration points before using the saved
registration for downstream analysis.
