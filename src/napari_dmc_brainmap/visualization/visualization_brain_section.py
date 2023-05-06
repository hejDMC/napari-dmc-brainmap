

results_data_merged = coord_mm_transform(results_data_merged)

results_dir = '/home/felix/Academia/DMC-lab/Projects/Dopamine/Analyses/Anatomy/results/plots_orb-manuscript/'  # to store figs
section_list = [2.5, 2.0, 1.5, 1.0]  # in mm AP coordinates for coronal sections relative to bregma -- [ORB, AUD]
# target_list = ['Orbital area', 'Prelimbic area', 'Infralimbic area', 'Anterior cingulate area', 'Secondary motor area', 'Agranular insular area']
# target_colors = ['green', 'blue', 'yellow', 'pink', 'brown', 'orange']
target_list = ['Orbital area, lateral part', 'Orbital area, ventrolateral part', 'Orbital area, medial part',
               'Prelimbic area', 'Infralimbic area',
               'Anterior cingulate area, dorsal part', 'Anterior cingulate area, ventral part',
               'Agranular insular area, dorsal part', 'Agranular insular area, ventral part',
               'Secondary motor area']
target_colors = ['darkgreen', 'green', 'lightgreen',
                 'navy', 'royalblue',
                 'darkorchid', 'plum',
                 'olive', 'yellowgreen',
                 'indigo']
target_transparency = [100] * len(target_list)
for section in section_list:
    slice_idx = int(-(section / 0.01 - bregma[0]))
    annot_section = annot[slice_idx, :, 570:].copy()
    annot_section_plt = plot_brain_schematic(annot_section, structure_tree, target_list, target_colors, target_transparency)
    fig, ax = plt.subplots(figsize=(10, 6))  # len(section_list), figsize=(18,5))
    im = ax.imshow(annot_section_plt)
    ax.contour(annot[slice_idx, :, 570:], levels=np.unique(annot[slice_idx, :, 570:]), colors=['gainsboro'],
               linewidths=0.2)
    ax.axis('off')

def get_brain_section_params(brainsec_widget):
    plotting_params = {
        "section_list": split_to_list(heatmap_widget.cmap.value),
        "groups": heatmap_widget.cbar_label.value,
        "cmap_groups": split_to_list(heatmap_widget.cmap.value),
        "range": heatmap_widget.cbar_label.value,
        "cmap": split_to_list(heatmap_widget.cmap.value),

        "cmap_min_max":  [float(i) for i in heatmap_widget.cmap_min_max.value.split(',')],  # [0] is absolute value for vmin, [1] is value to multiply max_range value with
        "intervals": [float(i) for i in heatmap_widget.intervals.value.split(',')],
        "interval_labels": get_interval_labels(split_to_list(heatmap_widget.intervals.value)),
        "include_layers": heatmap_widget.include_layers.value,
        "transpose": heatmap_widget.transpose.value,
        "save_name": heatmap_widget.save_name.value,
        "save_fig": heatmap_widget.save_fig.value,
        "absolute_numbers": heatmap_widget.absolute_numbers.value
    }
    return plotting_params

def get_rows_cols(section_list):
    n_sec = len(section_list)
    if n_sec % 2 == 0:
        n_rows = n_cols = n_sec / 2
    else:
        n_rows = round(n_sec / 2)
        n_cols = round(n_sec / 2) + 1
    return int(n_rows), int(n_cols)


geno_cmap = {'rbp4': 'coral', 'cux2': 'orchid', 'tlx3': 'royalblue'}
tgt_channel = 'green'  # todo check if you want to do more
section_list = [2.5, 2.0, 1.5]  # in mm AP coordinates for coronal sections

mpl_widget = FigureCanvas(Figure(figsize=(figsize)))
static_ax = mpl_widget.figure.subplots(1, (len(tgt_list) + 1),
                                       gridspec_kw={'width_ratios': [1] * len(tgt_list) + [0.15]})
n_row = 0
for n_sec, section in enumerate(section_list):
    n_col = n_sec%4
    target_ap = [section + 0.05, section - 0.05]
    target_ap = [int(-(target / 0.01 - bregma[0])) for target in target_ap]
    slice_idx = int(-(section / 0.01 - bregma[0]))
    annot_section = annot[slice_idx, :, :].copy()
    annot_section_plt = plot_brain_schematic(annot_section, structure_tree)
    im = ax[n_row, n_col].imshow(annot_section_plt)
    ax[n_row, n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]), colors=['gainsboro'],
              linewidths=0.2)
    # for geno_col, geno in zip(geno_cmap, geno_list):
    #     df = results_data_merged[(results_data_merged['genotype'] == geno) & (results_data_merged['channel'] == tgt_channel) &
    #                              (results_data_merged['AP_location'] >= target_ap[0]) &
    #                              (results_data_merged['AP_location'] <= target_ap[1])]
    #     # df = df.reset_index()
    #     sns.scatterplot(ax=ax[n_sec], x='ML_location', y='DV_location', data=df, palette=geno_col)
    df = results_data_merged[(results_data_merged['genotype'] != 'wt') & (results_data_merged['channel'] == tgt_channel) &
                             (results_data_merged['AP_location'] >= target_ap[0]) &
                             (results_data_merged['AP_location'] <= target_ap[1])]
    # df = df.reset_index()
    sns.scatterplot(ax=ax[n_row, n_col], x='ML_location', y='DV_location', data=df, hue='genotype', palette=geno_cmap)
    ax[0].title.set_text('bregma - ' + str(-(slice_idx - bregma[0]) * 0.01) + ' mm')
    ax[0].axis('off')
