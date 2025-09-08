import numpy as np
from skimage.transform import ProjectiveTransform
from typing import Union, List


def fitGeoTrans(src: Union[List[int], np.ndarray], 
                dst: Union[List[int], np.ndarray], 
                mode="projective",**kwargs) -> np.ndarray:
    """
    This function is the same as matlab fitgeotrans
    https://github.com/huruifeng/MERmate
    /merfish/scripts/affine.py
    """
    src = np.float32(src)
    dst = np.float32(dst)
    if 'projective' ==mode:
        # tform = findProjectiveTransform(src, dst)
        # tform = tform.params
        tform_x = ProjectiveTransform()
        tform_x.estimate(src, dst)
        tform_x = tform_x.params
    else:
        raise Exception("Unsupported transformation")
    return tform_x

def mapPointTransform(x_sample : Union[int,float],
                      y_sample : Union[int,float],
                      tform : np.ndarray) -> tuple[float, float]:
    vec_3 = np.array([x_sample,y_sample,1])
    fit_3 = np.matmul(tform,vec_3)
    x_atlas,y_atlas = fit_3[0]/fit_3[2],fit_3[1]/fit_3[2]
    return x_atlas,y_atlas

def predictPointSample(x_atlas,y_atlas,tform):
    vec_3 = np.array([x_atlas,y_atlas,1])
    fit_3 = np.matmul(tform,vec_3)
    x_sample,y_sample = fit_3[0]/fit_3[2],fit_3[1]/fit_3[2]
    return x_sample,y_sample
