from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.tools_slice import *
import tifffile
from natsort import natsorted
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pkg_resources import resource_filename

class sliceHandle():
    def __init__(self, registration=False) -> None:
        self.sharpy_dir = Path(resource_filename("napari_dmc_brainmap", 'registration'))
        if registration:
            self.jsonPath = registration
            self.parseJSON()
        self.loadAnnot()
        self.getTransform()
        self.calculateImageGrid()
        self.currentSlice = None
        self.ImgFolder = None
        self.df_tree = pd.read_csv(self.sharpy_dir.joinpath('sharpy_track', 'sharpy_track', 'atlas', 'structure_tree_safe_2017.csv'))
        self.bregma = [540,0,570] # DV=0 for now, can be 65 (Pierre, Neuron 2021),
        # or 44 (https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858)
    
    def setSlice(self,slice_n):
        self.currentSlice = slice_n
        self.loadImg()

    def setImgFolder(self,ImgFolder):
        self.ImgFolder = ImgFolder
    
    def loadImg(self):
        files_all = natsorted(os.listdir(self.ImgFolder))
        self.sampleImgFiles = []
        for f in files_all:
            if not(f.startswith('.')) and (f.endswith('.tif') or f.endswith('.tiff')):
                self.sampleImgFiles.append(f)
            else:
                pass
        self.currentSampleImg = tifffile.imread(os.path.join(self.ImgFolder,self.sampleImgFiles[self.currentSlice]))
        print('Working on: ',self.sampleImgFiles[self.currentSlice])


    def loadAnnot(self):
        self.annot = np.load(self.sharpy_dir.joinpath('sharpy_track', 'sharpy_track', 'atlas', 'annotation_volume_10um_by_index.npy'))

    def parseJSON(self):
        with open(self.jsonPath,"r") as f:
            self.regData = json.load(f)
            
    
    def getTransform(self):
        self.tforms = {}
        for k in self.regData['atlasDots'].keys():
            if len(self.regData['atlasDots'][k]) < 5:
                pass
            else: # valid pairs of points, calculate transformation
                self.tforms[k] = fitGeoTrans(self.regData['sampleDots'][k],self.regData['atlasDots'][k])
    
    def getVolumeIndex(self,slice_n,sample_coords):
        ap_plane = self.getPlaneAP(slice_n)
        volIndex_list = []
        for s_coord in sample_coords:
            x_pre, y_pre = s_coord
            x_post,y_post = mapPointTransform(x_pre,y_pre,self.tforms[slice_n])
            dv = int(y_post)
            ml = int(x_post)
            ap = int(ap_plane[dv,ml])
            volIndex_list.append([ap,dv,ml])
        return volIndex_list
        
    def getStructureID(self,volIndex):
        return self.annot[volIndex[0],volIndex[1],volIndex[2]]
    
    def getPlaneAP(self,slice_n):
        MLangle = self.regData['atlasLocation'][slice_n][0]
        DVangle = self.regData['atlasLocation'][slice_n][1]

        ap_c = get_ap(self.regData['atlasLocation'][slice_n][2])

        if (MLangle == 0) and (DVangle == 0): # flat plane
            ap_plane = np.full((800,1140),ap_c,dtype=np.uint16)

        else: # angled plane
            ml_shift = int(np.tan(np.deg2rad(MLangle)) * 570)
            dv_shift = int(np.tan(np.deg2rad(DVangle)) * 400)
            center = np.array([ap_c,400,570])
            c_right = np.array([ap_c+ml_shift,400,1139])
            c_top = np.array([ap_c-dv_shift,0,570]) 
            vec_1 = c_right-center
            vec_2 = c_top-center
            vec_n = np.cross(vec_1,vec_2)
            ap_plane = (-vec_n[1]*(self.grid[:,:,0]-center[1])-vec_n[2]*(self.grid[:,:,1]-center[2]))/vec_n[0] + center[0]

        return ap_plane

    def renderAtlas(self,slice_n):
        return angleSlice(self.regData['atlasLocation'][slice_n][0],self.regData['atlasLocation'][slice_n][1],self.regData['atlasLocation'][slice_n][2])

    def visualizeMapping(self,inputCoordinates):
        inputCoordinates = np.array(inputCoordinates)
        x_atlas,y_atlas = inputCoordinates.T

        fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(30,12))
        atlas_img = self.renderAtlas(str(self.currentSlice))
        ax[0].imshow(atlas_img,cmap='gray')
        ax[1].imshow(self.currentSampleImg,cmap='gray')
        ax[1].scatter(x_atlas,y_atlas,c='r',s=1)
        vol_index = self.getVolumeIndex(str(self.currentSlice),inputCoordinates)
        for i in vol_index:
            sphinx_id = self.getStructureID(i)
            # calculate Allen coordinates in mm unit
            ap_mm,dv_mm,ml_mm = self.getCoordMM(i)
            print(sphinx_id,self.df_tree.iloc[sphinx_id-1,:]['safe_name'],
                  ' AP:',str(ap_mm),' ML:',str(ml_mm),' DV:',str(dv_mm))
            ax[0].scatter([i[2]],[i[1]],c='r',s=1)
        plt.show()
    
    def getCoordMM(self,vol_index):
        vol_ap,vol_dv,vol_ml = vol_index
        ap_mm = np.round((self.bregma[0] - vol_ap) * 0.01,2)
        dv_mm = np.round((self.bregma[1] - vol_dv) * 0.01,2)
        ml_mm = np.round((self.bregma[2] - vol_ml) * 0.01,2)
        return [ap_mm,dv_mm,ml_mm]
       
    def calculateImageGrid(self): # one time calculation
        dv = np.arange(800)
        ml = np.arange(1140)
        grid_x,grid_y = np.meshgrid(ml,dv)
        self.grid = np.stack([grid_y,grid_x],axis=2)
        
    
    def getBrainArea(self,inputCoordinates, section_name):

        inputCoordinates = np.array(inputCoordinates)

        vol_index = self.getVolumeIndex(str(self.currentSlice),inputCoordinates) # [ap,dv,ml]
        id_list = []
        name_list =[]
        acronym_list =[]
        vol_mm_list = []
        for i in vol_index:
            sphinx_id = self.getStructureID(i)
            id_list.append(sphinx_id)
            name_list.append(self.df_tree.iloc[sphinx_id-1, :]['safe_name'])
            acronym_list.append(self.df_tree.iloc[sphinx_id - 1, :]['acronym'])
            # calculate Allen coordinates in mm unit
            vol_mm = self.getCoordMM(i)
            vol_mm_list.append(vol_mm)
        ap_coord, dv_coord, ml_coord = map(list, zip(*vol_index))
        ap_mm, dv_mm, ml_mm = map(list, zip(*vol_mm_list))

        section_data = pd.DataFrame(list(zip(name_list, acronym_list, ap_mm, dv_mm, ml_mm,
                                    id_list, ap_coord, dv_coord, ml_coord)),
                                    columns=['name', 'acronym', 'ap_mm', 'dv_mm', 'ml_mm',
                                              'sphinx_id', 'zpixel', 'ypixel', 'xpixel'])
        section_data['section_name'] = [section_name] * len(section_data)
        return section_data






