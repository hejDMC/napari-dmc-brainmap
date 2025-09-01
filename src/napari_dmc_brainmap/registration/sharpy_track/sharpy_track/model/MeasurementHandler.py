import json
import numpy as np
from copy import deepcopy
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.TreRow import TreRow

class MeasurementHandler:
    def __init__(self, measurementPage) -> None:
        self.regViewer = measurementPage.regViewer
        self.load_json()
    
    def load_json(self):
        # check if measurement.json exist
        json_path = self.regViewer.status.folderPath.joinpath('measurement.json')
        if json_path.is_file():
            with open(json_path, 'r') as json_file:
                self.json_data = json.load(json_file)
        else:
            self.create_measurement_template()
    
    def create_measurement_template(self):
        # create measurement.json template
        self.json_data = dict(sourceDots=dict(),
                              useTransformation=dict(),
                              targetDots=dict(),
                              truthDots=dict(),
                              treScore=dict(),
                              imgName=dict())
        # save to measurement.json
        with open(self.regViewer.status.folderPath.joinpath('measurement.json'), 'w') as json_file:
            json.dump(self.json_data, json_file)

    
    def load_measurement_record(self):
        # update imgIndex
        self.regViewer.measurementPage.active_rows["imgIndex"] = self.regViewer.status.currentSliceNumber
        # check if current sample has measurement record
        if str(self.regViewer.status.currentSliceNumber) in self.json_data["imgName"]:
            self.paint_rows()
        else:
            pass

    def paint_rows(self):
        imgIndex = self.regViewer.status.currentSliceNumber
        for source_xy in self.json_data["sourceDots"][str(imgIndex)]:
            self.regViewer.measurementPage.active_rows["source_coords"].append(source_xy)
            # recalculate tranformation matrix from registration json
            tform = fitGeoTrans(self.regViewer.status.sampleDots[imgIndex], 
                                self.regViewer.status.atlasDots[imgIndex])
            self.regViewer.measurementPage.active_rows["tform_matrix"] = tform.tolist()
            # remap target coordinates
            target_xy = mapPointTransform(source_xy[0], source_xy[1], tform)
            target_xy = [int(np.round(target_xy[0])), 
                         int(np.round(target_xy[1]))]
            self.regViewer.measurementPage.active_rows["target_coords"].append(target_xy)

        for truth_xy in self.json_data["truthDots"][str(imgIndex)]:
            self.regViewer.measurementPage.active_rows["truth_coords"].append(truth_xy)
        # calculate tre score
        for tru,tar in zip(self.regViewer.measurementPage.active_rows["truth_coords"],
                           self.regViewer.measurementPage.active_rows["target_coords"]):
            TRE = np.sqrt((tru[0] - tar[0])**2 + (tru[1] - tar[1])**2)
            self.regViewer.measurementPage.active_rows["tre_score"].append(np.round(TRE, 4).astype(str))
        # create row object (remove_btn disabled)
        for row_i in range(len(self.regViewer.measurementPage.active_rows["tre_score"])):
            row_obj = TreRow(self.regViewer.measurementPage)
            row_obj.source_pos_label.setText(f"({self.regViewer.measurementPage.active_rows['source_coords'][row_i][0]}, {self.regViewer.measurementPage.active_rows['source_coords'][row_i][1]})")
            row_obj.target_pos_label.setText(f"({self.regViewer.measurementPage.active_rows['target_coords'][row_i][0]}, {self.regViewer.measurementPage.active_rows['target_coords'][row_i][1]})")
            row_obj.true_pos_label.setText(f"({self.regViewer.measurementPage.active_rows['truth_coords'][row_i][0]}, {self.regViewer.measurementPage.active_rows['truth_coords'][row_i][1]})")
            row_obj.tre_label.setText(f"{float(self.regViewer.measurementPage.active_rows['tre_score'][row_i]):.2f}")
            self.regViewer.measurementPage.active_rows["row_obj"].append(row_obj)
        # create dot objects
        for source_xy in self.regViewer.measurementPage.active_rows["source_coords"]:
            # project to scaled space
            source_xy = self.regViewer.res_down[source_xy[0]], self.regViewer.res_down[source_xy[1]]
            self.regViewer.widget.viewerRight.addSourceDot(source_xy[0], source_xy[1])
        for truth_xy in self.regViewer.measurementPage.active_rows["truth_coords"]:
            # project to scaled space
            truth_xy = self.regViewer.res_down[truth_xy[0]], self.regViewer.res_down[truth_xy[1]]
            self.regViewer.widget.viewerLeft.addTruthDot(truth_xy[0], truth_xy[1])
        # enable remove_btn
        for row_obj in self.regViewer.measurementPage.active_rows["row_obj"]:
            row_obj.connect_delete_btn()
            row_obj.remove_btn.setEnabled(True)
            row_obj.setMouseTracking(True)

        




    def save_measurement_record(self):
        # check if current sample has measurement record
        if len(self.regViewer.measurementPage.active_rows["source_coords"]) == 0:
            # check if user deleted all data for current sample
            if str(self.regViewer.measurementPage.active_rows["imgIndex"]) in self.json_data["imgName"]:
                # delete dictionary entry for current sample index
                del self.json_data["sourceDots"][str(self.regViewer.measurementPage.active_rows["imgIndex"])]
                del self.json_data["useTransformation"][str(self.regViewer.measurementPage.active_rows["imgIndex"])]
                del self.json_data["targetDots"][str(self.regViewer.measurementPage.active_rows["imgIndex"])]
                del self.json_data["truthDots"][str(self.regViewer.measurementPage.active_rows["imgIndex"])]
                del self.json_data["treScore"][str(self.regViewer.measurementPage.active_rows["imgIndex"])]
                del self.json_data["imgName"][str(self.regViewer.measurementPage.active_rows["imgIndex"])]
                # update measurement.json
                with open(self.regViewer.status.folderPath.joinpath('measurement.json'), 'w') as json_file:
                    json.dump(self.json_data, json_file)
            else:
                pass
        else: # save record and clear page
            self.json_data["sourceDots"][str(self.regViewer.measurementPage.active_rows["imgIndex"])] = deepcopy(self.regViewer.measurementPage.active_rows["source_coords"])
            self.json_data["useTransformation"][str(self.regViewer.measurementPage.active_rows["imgIndex"])] = deepcopy(self.regViewer.measurementPage.active_rows["tform_matrix"])
            self.json_data["targetDots"][str(self.regViewer.measurementPage.active_rows["imgIndex"])] = deepcopy(self.regViewer.measurementPage.active_rows["target_coords"])
            self.json_data["truthDots"][str(self.regViewer.measurementPage.active_rows["imgIndex"])] = deepcopy(self.regViewer.measurementPage.active_rows["truth_coords"])
            self.json_data["treScore"][str(self.regViewer.measurementPage.active_rows["imgIndex"])] = deepcopy(self.regViewer.measurementPage.active_rows["tre_score"])
            self.json_data["imgName"][str(self.regViewer.measurementPage.active_rows["imgIndex"])] = deepcopy(self.regViewer.status.imgFileName[self.regViewer.measurementPage.active_rows["imgIndex"]])
            # update measurement.json
            with open(self.regViewer.status.folderPath.joinpath('measurement.json'), 'w') as json_file:
                json.dump(self.json_data, json_file)
            # cleanup page
            for _ in range(len(self.regViewer.measurementPage.active_rows["row_obj"])):
                self.regViewer.measurementPage.active_rows["row_obj"][0].remove_registered_row()

            self.regViewer.measurementPage.active_rows["tform_matrix"] = None
            self.regViewer.measurementPage.active_rows["imgIndex"] = None








