import json
import numpy as np
import pandas as pd








def regi_points_polygon(x_scaled, y_scaled):

    poly_points = [(x_scaled[i], y_scaled[i]) for i in range(0, len(x_scaled))]
    polygon = path.Path(poly_points)
    x_min, x_max = x_scaled.min(), x_scaled.max()
    y_min, y_max = y_scaled.min(), y_scaled.max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, (x_max - x_min) + 1),
                         np.linspace(y_min, y_max, (y_max - y_min) + 1))
    canvas_points = [(np.ndarray.flatten(xx)[i], np.ndarray.flatten(yy)[i]) for i in
                     range(0, len(np.ndarray.flatten(xx)))]
    idx_in_polygon = polygon.contains_points(canvas_points)
    points_in_polygon = [c for c, i in zip(canvas_points, idx_in_polygon) if i]
    x_poly = [p[0] for p in points_in_polygon]
    y_poly = [p[1] for p in points_in_polygon]
    coords = np.stack([x_poly, y_poly], axis=1)
    return coords



def transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir, seg_im_suffix, regi_data, regi_dir, regi_suffix):
    # todo input differently?
    curr_im = im[:-len(segment_suffix)]
    img = cv2.imread(str(seg_im_dir.joinpath(curr_im + seg_im_suffix)))
    y_im, x_im, z_im = img.shape  # original resolution of image
    # correct for 0 indices
    y_im -= 1
    x_im -= 1
    img_regi = cv2.imread(str(regi_dir.joinpath(curr_im + regi_suffix)))
    y_low, x_low, z_low = img_regi.shape  # original resolution of image
    # correct for 0 indices
    y_low -= 1
    x_low -= 1

    segment_data = pd.read_csv(segment_dir.joinpath(im))
    y_pos = list(segment_data['Position Y'])
    x_pos = list(segment_data['Position X'])
    # append mix max values for rescaling
    y_pos.append(0)
    y_pos.append(y_im)
    x_pos.append(0)
    x_pos.append(x_im)
    y_scaled = np.ceil(minmax_scale(y_pos, feature_range=(0, y_low)))[:-2].astype(int)
    x_scaled = np.ceil(minmax_scale(x_pos, feature_range=(0, x_low)))[:-2].astype(int)
    if seg_type == 'injection_site':
        for n in segment_data['idx_shape'].unique():
            n_idx = segment_data.index[segment_data['idx_shape'] == n].tolist()
            curr_x = np.array([x_scaled[i] for i in n_idx])
            curr_y = np.array([y_scaled[i] for i in n_idx])
            curr_coords = regi_points_polygon(curr_x, curr_y)
            if n == 0:
                coords = curr_coords
            else:
                coords = np.concatenate((coords, curr_coords), axis=0)

    else:
        coords = np.stack([x_scaled, y_scaled], axis=1)

    # slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
    s.setImgFolder(regi_dir)
    # set which slice in there
    s.setSlice(curr_im + regi_suffix)
    section_data = s.getBrainArea(coords, (curr_im + regi_suffix))
    if seg_type == "genes":
        section_data['cluster_id'] = segment_data['cluster_id']
        section_data['spot_id'] = segment_data['spot_id']
    return section_data




