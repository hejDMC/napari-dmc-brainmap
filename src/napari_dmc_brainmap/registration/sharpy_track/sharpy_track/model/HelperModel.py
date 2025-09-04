import numpy as np 
import matplotlib.pyplot as plt 
from PyQt5.QtGui import QPixmap,QImage


class HelperModel():
    def __init__(self,regViewer):
        self.regViewer = regViewer
        saggital_mid_index = int(regViewer.atlasModel.xyz_dict['x'][1]/2)
        self.saggital_mid = regViewer.atlasModel.template[:,:,saggital_mid_index].T
        self.get_illustration_stats()
        self.anchor_dict = {}
        self.mapping_dict = {}
        self.total_num = regViewer.status.sliceNum
        self.active_anchor = []
        self.get_location_img0()


    def get_illustration_stats(self):
        self.bregma_z_vox = self.regViewer.atlasModel.bregma[self.regViewer.atlasModel.z_idx] # 540 from [540,0,570]
        self.z_vox_max = self.regViewer.atlasModel.xyz_dict['z'][1]
        
        self.z_mm_ant = np.round((self.bregma_z_vox * self.regViewer.atlasModel.xyz_dict['z'][2])/1000,  # convert um to mm
                                 self.regViewer.status.decimal)
        self.z_mm_pos = np.round(
            (
            (
            self.bregma_z_vox - self.z_vox_max + 1
            ) * self.regViewer.atlasModel.xyz_dict['z'][2]
            ) /1000,
            self.regViewer.status.decimal)

        if self.z_mm_ant > 0:
            self.z_mm_ant_str = "+{}mm".format(self.z_mm_ant)
        else:
            self.z_mm_ant_str = "{}mm".format(self.z_mm_ant)

        self.z_mm_pos_str = "{}mm".format(self.z_mm_pos)

        self.figsize = [self.saggital_mid.shape[1] * 4 / 1140,
                        self.saggital_mid.shape[0] * 3 / 800]



    
    def get_location_img0(self): # initiate bregma only
        fig,ax = plt.subplots(figsize=(self.figsize[0],self.figsize[1]),nrows=1,ncols=1)
        ax.imshow(self.saggital_mid,cmap='gray')
        ax.get_yaxis().set_visible(False)
        ax.xaxis.tick_top()
        ax.set_xlim(0,self.regViewer.atlasModel.xyz_dict['z'][1]-1)
        ax.set_xticks([0,self.bregma_z_vox,self.z_vox_max-1],
                      labels=[self.z_mm_ant_str,"0 mm",self.z_mm_pos_str],fontsize=7)
        
        # fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.img0 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
    

    def update_illustration(self):
        fig,ax = plt.subplots(figsize=(self.figsize[0],self.figsize[1]),nrows=1,ncols=1)
        ax.imshow(self.saggital_mid,cmap='gray')
        ax.get_yaxis().set_visible(False)
        ax.xaxis.tick_top()
        ax.set_xlim(0,self.regViewer.atlasModel.xyz_dict['z'][1]-1)
        ax.set_xticks([0,self.bregma_z_vox,self.z_vox_max-1],
                      labels=[self.z_mm_ant_str,"0 mm",self.z_mm_pos_str],fontsize=7)

        if len(self.anchor_dict.keys())<2:
            pass # empty mapping_dict
        else:
            for v in self.mapping_dict.values():
                ax.axvline(int(self.bregma_z_vox-(1000/self.regViewer.atlasModel.xyz_dict['z'][2])*v),color='blue',linewidth=1)
        
        if len(self.anchor_dict.keys())==0:
            pass # empty anchor_dict
        else:
            for k,v in self.anchor_dict.items():
                ax.axvline(int(self.bregma_z_vox-(1000/self.regViewer.atlasModel.xyz_dict['z'][2])*v),color='yellow',linewidth=1)

                ax.annotate(text="{}".format(k),
                    xy=(int(self.bregma_z_vox-(1000/self.regViewer.atlasModel.xyz_dict['z'][2])*v),self.regViewer.atlasModel.xyz_dict['y'][1]),
                    xytext=(15,-15),
                    xycoords="data",
                    textcoords="offset points",
                    arrowprops={"arrowstyle":"simple",
                                "facecolor":"yellow",
                                "lw": 0.5})
        # fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        # update image in QLabel
        h,w,_ = self.img.shape
        previewimg_update = QImage(self.img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.regViewer.interpolatePositionPage.preview_label.setPixmap(QPixmap.fromImage(previewimg_update))


    def add_anchor(self,anchorrow,slice_id,ap_mm):
        self.active_anchor.append(anchorrow)
        self.anchor_dict[slice_id] = ap_mm
        self._update_mapping()
    

    def update_anchor(self):
        self.anchor_dict = {}
        for a in self.active_anchor:
            self.anchor_dict[a.spinSliceIndex.value()] = np.round(a.spinAPmm.value(),self.regViewer.status.decimal)
        self._update_mapping()
        
    
    def remove_anchor(self,anchorrow):
        self.active_anchor.remove(anchorrow)
        self.update_anchor()
        self._update_mapping()


    
    def _update_mapping(self):
        if len(self.anchor_dict.keys())<2:
            self.mapping_dict = {}
        else:
            self.mapping_dict = {}
            for s in range(self.total_num):
                self.mapping_dict[s] = self.get_ap_from_id(s)
        self.update_illustration()
        self.regViewer.interpolatePositionPage.update_button_availability(status_code=1)
    
    def get_ap_from_id(self,slice_id):
        slice_id_list = list(self.anchor_dict.keys())
        if slice_id in slice_id_list: # slice id is anchor
            ap_from_id = self.anchor_dict[slice_id]
            # print("Slice {} is manully set at {}mm".format(slice_id,ap_from_id))
        else: # interpolate
            slice_id_list.sort()
            if slice_id < slice_id_list[0]:
                # print("Segment {} ~ {}".format(0,slice_id_list[0]))
                # use anchor 0,1 for interpolation
                step = (self.anchor_dict[slice_id_list[0]]-self.anchor_dict[slice_id_list[1]])/(slice_id_list[1] - slice_id_list[0])
                step_n = slice_id_list[0] - slice_id
                ap_from_id = np.round(self.anchor_dict[slice_id_list[0]] + step_n * step,self.regViewer.status.decimal)

            elif slice_id > slice_id_list[-1]:
                # print("Segment {} ~ {}".format(slice_id_list[-1],self.total_num-1))
                # use anchor -2,-1 for interpolation
                step = (self.anchor_dict[slice_id_list[-1]]-self.anchor_dict[slice_id_list[-2]])/(slice_id_list[-1] - slice_id_list[-2])
                step_n = slice_id - slice_id_list[-1]
                ap_from_id = np.round(self.anchor_dict[slice_id_list[-1]] + step_n * step,self.regViewer.status.decimal)

            else:
                for i in range(len(slice_id_list)):
                    if slice_id < slice_id_list[i]:
                        if slice_id > slice_id_list[i-1]:
                            # print("Segment {} ~ {}".format(slice_id_list[i-1],slice_id_list[i]))
                            # use anchor i-1,i for interpolation
                            step = (self.anchor_dict[slice_id_list[i-1]]-self.anchor_dict[slice_id_list[i]])/(slice_id_list[i] - slice_id_list[i-1])
                            step_n = slice_id_list[i] - slice_id
                            ap_from_id = np.round(self.anchor_dict[slice_id_list[i]] + step_n * step,self.regViewer.status.decimal)
        return ap_from_id
        