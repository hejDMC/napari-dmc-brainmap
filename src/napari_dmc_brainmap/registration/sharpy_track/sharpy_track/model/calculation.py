import numpy as np
from skimage.transform import ProjectiveTransform


## generate coordinate mapping
# generate ap-axis mapping
#def get_z(z_mm): # todo delete
#    return int(round((5.39-ap_mm) * 100))
# generate dv-axis mapping
# def get_y(dv_mm):
#     return int(round(dv_mm * 100))
# # generate ml-axis mappint
# def get_x(ml_mm):
#     return int(round((5.69+ml_mm) * 100))

def fitGeoTrans(src, dst, mode="projective",**kwargs):
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

def mapPointTransform(x_sample,y_sample,tform):
    vec_3 = np.array([x_sample,y_sample,1])
    fit_3 = np.matmul(tform,vec_3)
    x_atlas,y_atlas = fit_3[0]/fit_3[2],fit_3[1]/fit_3[2]
    return x_atlas,y_atlas

def predictPointSample(x_atlas,y_atlas,tform):
    vec_3 = np.array([x_atlas,y_atlas,1])
    fit_3 = np.matmul(tform,vec_3)
    x_sample,y_sample = fit_3[0]/fit_3[2],fit_3[1]/fit_3[2]
    return x_sample,y_sample
