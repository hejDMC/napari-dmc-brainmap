#%%
import pandas as pd
from napari_dmc_brainmap.probe_visualizer.probe_visualizer_tools import estimate_confidence
import tifffile


df = pd.read_csv(r"C:\Users\xiao\histology_data\NPX-000\neuropixels_probe_0_data.csv")
annot = tifffile.imread(r"C:\Users\xiao\.brainglobe\allen_mouse_10um_v1.2\annotation.tiff")

confidence_list = estimate_confidence(v_coords = df[["Voxel_AP",
                                                     "Voxel_DV",
                                                     "Voxel_ML"]],
                                                atlas_resolution_um = 10,
                                                annot = annot)



print(confidence_list)
