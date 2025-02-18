# @thread_worker(progress={'total': 100})
# def do_presegmentation(input_path, params_dict, channels, single_channel, regi_bool, regi_chan, preseg_params,
#                        start_end_im, mask_folder, output_folder, seg_type='cells'):
#     """
#     Perform pre-segmentation on the provided image data.
#
#     Parameters:
#     input_path (Path): The path to the input images.
#     params_dict (dict): Dictionary containing various parameters for the process.
#     channels (list): List of channels to segment.
#     single_channel (bool): Flag to indicate if the image is a single channel.
#     regi_bool (bool): Whether the registration was performed.
#     regi_chan (str): Registration channel.
#     preseg_params (dict): Pre-segmentation parameters.
#     start_end_im (list): Start and end indices for image range to presegment.
#     mask_folder (str): Folder name for masks.
#     output_folder (str): Folder name for output.
#     seg_type (str): Segmentation type.
#     """
#     if single_channel:
#         single_channel = 'single_channel'  # todo bit weird detour check if this stills works
#     xyz_dict = params_dict['atlas_info']['xyz_dict']
#     atlas_id = params_dict['atlas_info']['atlas']
#     regi_list = []
#     if regi_bool:
#         try:
#             regi_data, annot_bool, z_idx, z_res, bregma = load_registration_data(input_path, regi_chan, atlas_id, xyz_dict)
#             regi_list = [regi_data, annot_bool, z_idx, z_res, bregma]
#         except FileNotFoundError:
#             show_info('NO REGISTRATION DATA FOUND')
#             regi_bool = False
#
#     progress_value = 0
#     n_images = get_total_images(input_path, start_end_im, channels, single_channel)
#     progress_step = 100 / n_images
#     for chan in channels:
#         mask_dir, output_dir = prepare_segmentation_folders(input_path, mask_folder, output_folder, chan, seg_type)
#
#         seg_im_dir, seg_im_list, seg_im_suffix = load_image_list(input_path, start_end_im, chan, single_channel)
#
#         for im in seg_im_list:
#             print(f'Processing image: {im}')
#             image_path = seg_im_dir.joinpath(im)
#             chan_dict = {
#                 'cy3': 0,
#                 'green': 1,
#                 'dapi': 2
#             }
#             struct_img0 = load_segmentation_image(image_path, chan, single_channel, structure_channel=chan_dict[chan])
#             struct_img0 = np.array([struct_img0, struct_img0])  # Duplicate layer stack
#
#             structure_img_smooth = preprocess_image(struct_img0, preseg_params)
#
#             segmentation = segment_image(structure_img_smooth, preseg_params)
#             if np.mean(segmentation[0]) != 0:
#                 mask_save_fn = mask_dir.joinpath(im[:-len(seg_im_suffix)] + '_masks.tiff')
#                 save_segmentation_to_tiff(segmentation, mask_save_fn)
#
#                 csv_save_name = output_dir.joinpath(im.split('.')[0] + '_' + seg_type + '.csv')
#                 save_segmentation_to_csv(segmentation, csv_save_name, regi_bool, regi_list, xyz_dict, im, seg_im_suffix)
#             progress_value += progress_step
#             yield int(progress_value), 'preseg_cells'
#     yield 100, 'preseg_cells'
#     return 'preseg_cells'


# @thread_worker(progress={'total': 100})
# def create_projection_preseg_old(input_path, params_dict, channels, regi_bool, regi_chan, binary_folder, output_folder):
#     xyz_dict = params_dict['atlas_info']['xyz_dict']
#     atlas_id = params_dict['atlas_info']['atlas']
#     progress_value = 0
#     n_images = get_total_images(input_path, False, channels, 'binary')
#     progress_step = 100 / n_images
#
#     if regi_bool:
#         try:
#             regi_data, annot_bool, z_idx, z_res, bregma = load_registration_data(input_path, regi_chan, atlas_id,
#                                                                                   xyz_dict)
#             regi_list = [regi_data, annot_bool, z_idx, z_res, bregma]
#         except FileNotFoundError:
#             show_info('NO REGISTRATION DATA FOUND')
#             regi_bool = False
#
#     for chan in channels:
#         binary_dir, binary_images, binary_suffix = get_info(input_path, binary_folder, channel=chan)
#         output_dir = get_info(input_path, output_folder, seg_type='projections', channel=chan, create_dir=True,
#                               only_dir=True)
#         # binary_images = natsorted([im.parts[-1] for im in binary_dir.glob('*.tif')])
#         for im_name in binary_images:
#             print(f'... {im_name}')
#             path_to_im = binary_dir.joinpath(im_name)
#             image = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
#             idx = np.where(image == 255)
#             if regi_bool:  # exclude idx outside of brain
#                 dim_binary = image.shape
#                 x_res = xyz_dict['x'][1]
#                 y_res = xyz_dict['y'][1]
#                 x_binary = idx[1] / dim_binary[1] * x_res
#                 y_binary = idx[0] / dim_binary[0] * y_res
#                 for k, v in regi_data['imgName'].items():
#                     if v.startswith(im_name[:-len(binary_suffix)]):
#                         regi_index = k
#                 drop_bool = transform_segmentation(regi_index, xyz_dict, x_binary, y_binary, regi_list)
#
#
#
#             else:
#                 drop_bool = False
#             csv_to_save = pd.DataFrame({'Position Y': idx[0], 'Position X': idx[1]})
#             if regi_bool and drop_bool:
#                 csv_to_save = csv_to_save.iloc[np.where(np.array(drop_bool) == 0)[0], :].copy().reset_index(
#                     drop=True)
#             csv_save_name = output_dir.joinpath(im_name.split('.')[0] + '_projections.csv')
#             csv_to_save.to_csv(csv_save_name)
#             progress_value += progress_step
#             yield int(progress_value), 'preseg_proj'
#     yield 100, 'preseg_proj'
#     return 'preseg_proj'

# @thread_worker(progress={'total': 100})
# def get_center_coord(input_path, channels, mask_folder, output_folder, mask_type='cells'):
#     progress_value = 0
#     n_images = get_total_images_centroids(input_path, channels, mask_folder, mask_type)
#     progress_step = 100 / n_images
#
#     for chan in channels:
#         mask_dir = get_info(input_path, mask_folder, seg_type=mask_type, channel=chan, only_dir=True)
#         output_dir = get_info(input_path, output_folder, seg_type=mask_type, channel=chan, create_dir=True,
#                               only_dir=True)
#         mask_images = natsorted([im.parts[-1] for im in mask_dir.glob('*.tiff')])
#         for im_name in mask_images:
#             path_to_im = mask_dir.joinpath(im_name)
#             image = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
#             label_img = label(image)  # identify individual segmented structures
#             regions = regionprops(
#                 label_img)  # get there properties -> we want to have the centroid point as a "location" of the cell
#             cent = np.zeros((np.size(regions), 2))
#             for idx, props in enumerate(regions):
#                 cent[idx, 0] = props.centroid[0]  # y-coordinates
#                 cent[idx, 1] = props.centroid[1]  # x-coordinates
#             # create csv file in folders to match imaris output
#             csv_to_save = pd.DataFrame(cent)
#             csv_to_save = csv_to_save.rename(columns={0: "Position Y", 1: "Position X"})
#             csv_save_name = output_dir.joinpath(im_name.split('.')[0] + '_' + mask_type + '.csv')
#             csv_to_save.to_csv(csv_save_name)
#             #
#             location_binary = np.zeros((image.shape))  # make new binary with centers of segmented cells only
#             cent = (np.round(cent)).astype(int)
#             for val in cent:
#                 location_binary[val[0], val[1]] = 255
#             location_binary = location_binary.astype(int)
#             location_binary = location_binary.astype('uint8')  # convert to 8-bit
#             save_name = im_name.split('.')[0] + '_centroids.tif'  # get the name
#             cv2.imwrite(str(mask_dir.joinpath(save_name)), location_binary)
#             progress_value += progress_step
#             yield int(progress_value), 'preseg_center'
#     yield 100, 'preseg_center'
#     return 'preseg_center'


# def _do_presegmentation(self):
    #     input_path = self.segment.input_path.value
    #     # check if user provided a valid input_path
    #     if not input_path.is_dir() or str(input_path) == '.':
    #         msg_box = QMessageBox()
    #         msg_box.setIcon(QMessageBox.Critical)
    #         msg_box.setText(f"Input path is not a valid directory. Please make sure this exists: >> '{str(input_path)}' <<")
    #         msg_box.setWindowTitle("Invalid Path Error")
    #         msg_box.exec_()
    #         return
    #
    #     params_dict = load_params(input_path)
    #     channels = self.segment.channels.value
    #     single_channel = self.preseg_cells.single_channel_bool.value
    #     regi_bool = self.preseg_cells.regi_bool.value
    #     regi_chan = self.preseg_cells.regi_chan.value
    #     seg_type = self.preseg_cells.seg_type.value
    #     preseg_params = {
    #         "intensity_norm": split_to_list(self.preseg_cells.intensity_norm.value, out_format='float'),
    #         "gaussian_smoothing_sigma": int(self.preseg_cells.gaussian_smoothing_sigma.value),
    #         # "gaussian_smoothing_truncate_range": int(self.preseg_cells.gaussian_smoothing_truncate_range.value),
    #         "dot_3d_sigma": int(self.preseg_cells.dot_3d_sigma.value),
    #         "dot_3d_cutoff": float(self.preseg_cells.dot_3d_cutoff.value),
    #         "hole_min_max": split_to_list(self.preseg_cells.hole_min_max.value, out_format='int'),
    #         "minArea": int(self.preseg_cells.minArea.value)
    #     }
    #     start_end_im = split_to_list(self.preseg_cells.start_end_im.value, out_format='int')
    #     mask_folder = self.preseg_cells.mask_folder.value
    #     output_folder = self.preseg_cells.output_folder.value
    #     do_preseg_worker = do_presegmentation(input_path, params_dict, channels, single_channel, regi_bool, regi_chan,
    #                                           preseg_params, start_end_im, mask_folder, output_folder,
    #                                           seg_type=seg_type)
    #     do_preseg_worker.yielded.connect(self._update_progress)
    #     do_preseg_worker.started.connect(
    #         lambda: self.btn_cells.setText("Presegmenting images..."))  # Change button text when stitching starts
    #     do_preseg_worker.returned.connect(self._show_success_message)
    #     do_preseg_worker.start()

    # def _create_projection_preseg(self):
    #     input_path = self.segment.input_path.value
    #     params_dict = load_params(input_path)
    #     channels = self.segment.channels.value
    #     regi_bool = self.preseg_projections.regi_bool.value
    #     regi_chan = self.preseg_projections.regi_chan.value
    #     binary_folder = self.preseg_projections.binary_folder.value
    #     output_folder = self.preseg_projections.output_folder.value
    #     projection_worker = create_projection_preseg(input_path, params_dict, channels, regi_bool, regi_chan,
    #                                                  binary_folder, output_folder)
    #     projection_worker.yielded.connect(self._update_progress)
    #     projection_worker.started.connect(
    #         lambda: self.btn_projections.setText("Presegmenting images..."))  # Change button text when stitching starts
    #     projection_worker.returned.connect(self._show_success_message)
    #     projection_worker.start()

# def _get_center_coord(self):
#     input_path = self.segment.input_path.value
#     channels = self.segment.channels.value
#     mask_folder = self.center.mask_folder.value
#     mask_type = self.center.mask_type.value
#     output_folder = self.center.output_folder.value
#     center_worker = get_center_coord(input_path, channels, mask_folder, output_folder, mask_type=mask_type)
#     center_worker.yielded.connect(self._update_progress)
#     center_worker.started.connect(
#         lambda: self.btn_find_centroids.setText("Calculating centroids..."))  # Change button text when stitching starts
#     center_worker.returned.connect(self._show_success_message)
#     center_worker.start()


# def _update_progress(self, yield_tuple) -> None:
#     """
#     Update the progress bar with the given value.
#
#     Parameters:
#     - value (int): Progress value to set.
#     """
#     value, operation = yield_tuple
#     if operation == 'preseg_cells':
#         self.progress_signal_cells.emit(value)
#     elif operation == 'preseg_proj':
#         self.progress_signal_projections.emit(value)
#     else:
#         self.progress_signal_center.emit(value)