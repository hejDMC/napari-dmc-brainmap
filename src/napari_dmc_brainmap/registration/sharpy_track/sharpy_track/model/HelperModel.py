import numpy as np 
import matplotlib.pyplot as plt 
from PyQt5.QtGui import QPixmap,QImage


class HelperModel():
    def __init__(self,regViewer):
        self.regViewer = regViewer
        self.saggital_mid = regViewer.atlasModel.template[:,:,570].T
        self.get_location_img0()
        self.anchor_dict = {}
        self.mapping_dict = {}
        self.total_num = regViewer.status.sliceNum
        self.active_anchor = []

    
    def get_location_img0(self): # initiate bregma only
        fig,ax = plt.subplots(figsize=(4,3),nrows=1,ncols=1)
        ax.imshow(self.saggital_mid,cmap='gray')
        ax.get_yaxis().set_visible(False)
        ax.xaxis.tick_top()
        ax.set_xlim(0,1320)
        ax.set_xticks([0,540,1320],labels=["+5.4 mm","0 mm","-7.8 mm"])
        
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.img0 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
    

    def update_illustration(self):
        fig,ax = plt.subplots(figsize=(4,3),nrows=1,ncols=1)
        ax.imshow(self.saggital_mid,cmap='gray')
        ax.get_yaxis().set_visible(False)
        ax.xaxis.tick_top()
        ax.set_xlim(0,1320)
        ax.set_xticks([0,540,1320],labels=["+5.4 mm","0 mm","-7.8 mm"])

        if len(self.anchor_dict.keys())<2:
            pass # empty mapping_dict
        else:
            for v in self.mapping_dict.values():
                ax.axvline(int(540-100*v),color='blue',linewidth=1)
        
        if len(self.anchor_dict.keys())==0:
            pass # empty anchor_dict
        else:
            for k,v in self.anchor_dict.items():
                ax.axvline(int(540-100*v),color='yellow',linewidth=1)

                ax.annotate(text="i={}".format(k),
                    xy=(int(540-100*v),800),
                    xytext=(50,-50),
                    xycoords="data",
                    textcoords="offset points",
                    arrowprops={"arrowstyle":"simple",
                                "facecolor":"yellow",
                                "lw": 0.5})
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        # update image in QLabel
        h,w,_ = self.img.shape
        previewimg_update = QImage(self.img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.regViewer.helperPage.preview_label.setPixmap(QPixmap.fromImage(previewimg_update))


    def add_anchor(self,anchorrow,slice_id,ap_mm):
        self.active_anchor.append(anchorrow)
        self.anchor_dict[slice_id] = ap_mm
        self._update_mapping()
    

    def update_anchor(self):
        self.anchor_dict = {}
        for a in self.active_anchor:
            self.anchor_dict[a.spinSliceIndex.value()] = np.round(a.spinAPmm.value(),2)
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
        self.regViewer.helperPage.update_button_availability(status_code=1)
    
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
                ap_from_id = np.round(self.anchor_dict[slice_id_list[0]] + step_n * step,2)

            elif slice_id > slice_id_list[-1]:
                # print("Segment {} ~ {}".format(slice_id_list[-1],self.total_num-1))
                # use anchor -2,-1 for interpolation
                step = (self.anchor_dict[slice_id_list[-1]]-self.anchor_dict[slice_id_list[-2]])/(slice_id_list[-1] - slice_id_list[-2])
                step_n = slice_id - slice_id_list[-1]
                ap_from_id = np.round(self.anchor_dict[slice_id_list[-1]] + step_n * step,2)

            else:
                for i in range(len(slice_id_list)):
                    if slice_id < slice_id_list[i]:
                        if slice_id > slice_id_list[i-1]:
                            # print("Segment {} ~ {}".format(slice_id_list[i-1],slice_id_list[i]))
                            # use anchor i-1,i for interpolation
                            step = (self.anchor_dict[slice_id_list[i-1]]-self.anchor_dict[slice_id_list[i]])/(slice_id_list[i] - slice_id_list[i-1])
                            step_n = slice_id_list[i] - slice_id
                            ap_from_id = np.round(self.anchor_dict[slice_id_list[i]] + step_n * step,2)
        return ap_from_id
        